# =============================================================================
# =============================================================================
"""
Traceability layer
=============================

Records WHAT the app does (front → execution) for:
  1) visual inspection in Graylog (GELF UDP, real time, streams/alerts), and
  2) audit + reports: persistence in MongoDB (`user_activity`), app-owned.

Components:
  - `GelfEmitter`            : GELF 1.1 emitter over UDP (no external dependencies).
  - `_MongoSink`             : queue + flusher thread → inserts events into Mongo in batches
                               (does not block the request path).
  - `TraceabilityMiddleware` : PURE ASGI middleware (does not buffer SSE). 1 event per request.
  - `trace_event(...)`       : rich traces from the endpoints (user, action, ...).

Config-driven: `init_traceability(gelf_host, gelf_port, source, enabled, mongo_uri, mongo_db)`.
"""
from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Optional

logger = logging.getLogger("traceability")

# trace_id and user of the current request (set by the middleware, visible downstream).
_trace_id_ctx: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_user_ctx: ContextVar[Optional[str]] = ContextVar("trace_user", default=None)

# Regex to extract business fields from the body (JSON or json_data in form-data).
import re as _re
# The same field arrives in THREE different shapes depending on the endpoint, and until
# all three are covered the per-user audit only sees part of the traffic:
#   1. inside a JSON               {"user": "X"}            → json_data, JSON bodies
#   2. as a multipart field        name="user"\r\n\r\nX      → endpoints with Form(...)
#   3. url-encoded or query string user=X&...               → simple forms and GET
_CAMPOS = {
    "user":        (rb"user",                    150),
    "file_name":   (rb"(?:file_name|filename)",  300),
    "config_name": (rb"config_name",             150),
    "company":     (rb"company",                 150),
}
_SNIFF_JSON  = {k: _re.compile(rb'"' + n + rb'"\s*:\s*"([^"]{1,' + str(m).encode() + rb'})"')
                for k, (n, m) in _CAMPOS.items()}
_SNIFF_MULTI = {k: _re.compile(rb'name="' + n + rb'"[^\r\n]*\r?\n\r?\n([^\r\n]{1,'
                               + str(m).encode() + rb'})')
                for k, (n, m) in _CAMPOS.items()}
_SNIFF_FORM  = {k: _re.compile(rb'(?:^|&)' + n + rb'=([^&\r\n]{1,' + str(m).encode() + rb'})')
                for k, (n, m) in _CAMPOS.items()}

# backward compatibility
_USER_RE = _SNIFF_JSON["user"]
_SNIFF_FIELDS = _SNIFF_JSON


def _sniff_body_fields(body: bytes) -> dict:
    """Extract business fields from the body or the query string (best-effort).

    Tries the three shapes the app uses to carry them. Never raises.
    """
    from urllib.parse import unquote_plus
    out = {}
    for key in _CAMPOS:
        for rx in (_SNIFF_JSON[key], _SNIFF_MULTI[key], _SNIFF_FORM[key]):
            try:
                m = rx.search(body)
            except Exception:
                continue
            if not m:
                continue
            v = unquote_plus(m.group(1).decode("utf-8", "ignore")).strip()
            if not v:
                continue
            # uppercase the fields that the app also uppercases, so that auditing
            # per user does not lose events due to capitalization
            if key in ("user", "company"):
                v = v.upper()
            out[key] = v
            break
    return out


def get_trace_id() -> Optional[str]:
    return _trace_id_ctx.get()


def get_user() -> Optional[str]:
    return _user_ctx.get()


# ─────────────────────────────────────────────────────────────────────────────
# GELF (Graylog)
# ─────────────────────────────────────────────────────────────────────────────
class GelfEmitter:
    """GELF 1.1 emitter over UDP. Best-effort: never breaks the request."""

    def __init__(self, host: str, port: int, source: str = "chatbot", enabled: bool = True):
        self.host = host
        self.port = int(port)
        self.source = source
        self.enabled = bool(enabled)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, short_message: str, level: int = 6, **fields) -> None:
        if not self.enabled:
            return
        msg = {
            "version": "1.1",
            "host": self.source,
            "short_message": str(short_message)[:2000],
            "timestamp": time.time(),
            "level": int(level),
        }
        for k, v in fields.items():
            if v is None:
                continue
            key = k if k.startswith("_") else f"_{k}"
            if key == "_id":
                key = "_id_"
            # GELF only accepts string or number in the additional fields: a bool
            # is SILENTLY DROPPED when indexing. Normalizing it to text keeps the
            # value and leaves it queryable (`tsa:true`, `verified:false`).
            if isinstance(v, bool):
                v = "true" if v else "false"
            if isinstance(v, str) and len(v) > 4000:
                v = v[:4000] + "…"
            msg[key] = v
        try:
            data = json.dumps(msg, ensure_ascii=False, default=str).encode("utf-8")
            self._sock.sendto(data, (self.host, self.port))
        except Exception as e:
            logger.debug("GELF send failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Mongo (app-owned audit + reports)
# ─────────────────────────────────────────────────────────────────────────────
class _MongoSink:
    """In-memory queue + flusher thread that inserts into Mongo in batches."""

    def __init__(self, uri: str, db: str, collection: str = "user_activity",
                 retention_days: int = 90):
        from pymongo import MongoClient
        self._client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self._coll = self._client[db][collection]
        self._retention_days = int(retention_days)
        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=20000)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, name="traceability-flusher", daemon=True)
        self._t.start()
        self._ensure_indexes()

    def _ensure_indexes(self):
        """Indexes for fast audit lookups + retention TTL (best-effort)."""
        try:
            # fast lookup: by user/action/company/path + time, and by trace_id
            self._coll.create_index([("user", 1), ("created_at", -1)], name="user_time")
            self._coll.create_index([("action", 1), ("created_at", -1)], name="action_time")
            self._coll.create_index([("company", 1), ("created_at", -1)], name="empresa_time")
            self._coll.create_index([("path", 1), ("created_at", -1)], name="path_time")
            self._coll.create_index([("trace_id", 1)], name="trace")
            self._coll.create_index([("created_at", -1)], name="recent")
        except Exception as e:
            logger.debug("user_activity indexes: %s", e)
        # TTL: automatically expires documents past the retention window (GDPR: prompts = personal data)
        secs = self._retention_days * 86400
        try:
            self._coll.create_index([("created_at", 1)], name="ttl", expireAfterSeconds=secs)
        except Exception:
            # already exists with a different TTL → update with collMod
            try:
                self._coll.database.command(
                    "collMod", self._coll.name,
                    index={"name": "ttl", "expireAfterSeconds": secs},
                )
            except Exception as e2:
                logger.debug("TTL collMod: %s", e2)

    def put(self, doc: dict) -> None:
        try:
            self._q.put_nowait(doc)
        except queue.Full:
            pass  # drop if the queue is saturated (never blocks the request)

    def _loop(self):
        while not self._stop.is_set():
            batch = []
            try:
                batch.append(self._q.get(timeout=2))
                while len(batch) < 500:
                    batch.append(self._q.get_nowait())
            except queue.Empty:
                pass
            if batch:
                try:
                    self._coll.insert_many(batch, ordered=False)
                except Exception as e:
                    logger.debug("Mongo flush failed (%d events): %s", len(batch), e)


# ─────────────────────────────────────────────────────────────────────────────
# Global state + init
# ─────────────────────────────────────────────────────────────────────────────
_emitter: Optional[GelfEmitter] = None
_sink: Optional[_MongoSink] = None


def init_traceability(gelf_host: str, gelf_port: int, source: str = "chatbot",
                      enabled: bool = True, mongo_uri: Optional[str] = None,
                      mongo_db: Optional[str] = None, retention_days: int = 90) -> None:
    global _emitter, _sink
    _emitter = GelfEmitter(gelf_host, gelf_port, source, enabled)
    logger.info("✅ GELF traceability → %s:%s (enabled=%s, source=%s)",
                gelf_host, gelf_port, enabled, source)
    if mongo_uri and mongo_db:
        try:
            _sink = _MongoSink(mongo_uri, mongo_db, retention_days=retention_days)
            logger.info("✅ Mongo traceability → %s.user_activity (retention %sd)",
                        mongo_db, retention_days)
        except Exception as e:
            logger.warning("Mongo traceability unavailable (non-critical): %s", e)


def _record(short_message: str, level: int, fields: dict) -> None:
    """Emit the event to Graylog (GELF) and enqueue it for Mongo (best-effort)."""
    if _emitter is not None:
        _emitter.send(short_message, level=level, **fields)
    if _sink is not None:
        doc = {"ts": time.time(), "created_at": datetime.utcnow(),
               "short_message": short_message, "level": level}
        doc.update({k: v for k, v in fields.items() if v is not None})
        _sink.put(doc)


def trace_event(action: str, level: int = 6, **fields) -> None:
    """
    Record a business action (Graylog + Mongo). Use it inside the endpoints:
        trace_event("sql_query", user=..., hash=..., db_type=..., rows=...)
    """
    hint = fields.get("endpoint") or fields.get("path") or ""
    base = {"app": "chatbot", "action": action, "trace_id": _trace_id_ctx.get()}
    if not fields.get("user"):          # fallback: user resolved by the middleware
        base["user"] = _user_ctx.get()
    base.update(fields)
    _record(f"[{action}] {hint}".strip(), level, base)


# ─────────────────────────────────────────────────────────────────────────────
# Middleware ASGI (no bufferea streaming)
# ─────────────────────────────────────────────────────────────────────────────
def _header(scope, name: bytes) -> Optional[str]:
    for k, v in scope.get("headers", []):
        if k == name:
            try:
                return v.decode("latin-1")
            except Exception:
                return None
    return None


class TraceabilityMiddleware:
    """PURE ASGI middleware. Wraps `send` only to capture status + the X-Trace-Id header."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        trace_id = _header(scope, b"x-trace-id") or uuid.uuid4().hex
        method = scope.get("method")
        ctype = (_header(scope, b"content-type") or "").lower()
        want_sniff = method in ("POST", "PUT", "PATCH") and ("json" in ctype or "form" in ctype)

        token = _trace_id_ctx.set(trace_id)
        _user_ctx.set(None)

        # Attribution without a body: a GET has nothing to sniff, so we look at
        # (1) the X-User header, an explicit channel for the frontend, and (2) the query
        # string. Without this, 97% of the traffic ends up with no user.
        pre = {}
        qs = scope.get("query_string") or b""
        if qs:
            pre.update(_sniff_body_fields(qs))
        cab_user = _header(scope, b"x-user")
        if cab_user:
            pre["user"] = cab_user.strip().upper()
        cab_emp = _header(scope, b"x-company")
        if cab_emp:
            pre.setdefault("company", cab_emp.strip().upper())
        if pre.get("user"):
            _user_ctx.set(pre["user"])

        start = time.perf_counter()
        status_holder = {"status": 500}
        # PASS-THROUGH sniff: we observe the body as THE APP reads it (we never pre-consume
        # nor block → no deadlock with SSE/streaming responses). Cap 1MB for uploads.
        sniff = {"buf": bytearray(), "done": False, "fields": dict(pre)}

        async def receive_peek():
            msg = await receive()
            if want_sniff and not sniff["done"] and msg.get("type") == "http.request":
                if len(sniff["buf"]) <= 1_000_000:
                    sniff["buf"].extend(msg.get("body", b""))
                if (not msg.get("more_body")) or len(sniff["buf"]) > 1_000_000:
                    sniff["done"] = True
                    sniff["fields"] = {**pre, **_sniff_body_fields(bytes(sniff["buf"]))}
                    if sniff["fields"].get("user"):
                        _user_ctx.set(sniff["fields"]["user"])
            return msg

        async def send_wrapper(message):
            if message.get("type") == "http.response.start":
                status_holder["status"] = message.get("status", 200)
                headers = message.setdefault("headers", [])
                headers.append((b"x-trace-id", trace_id.encode("latin-1")))
            await send(message)

        try:
            await self.app(scope, receive_peek if want_sniff else receive, send_wrapper)
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            try:
                client = scope.get("client")
                status = status_holder["status"]
                _ev = {
                    "app": "chatbot",
                    "action": "http_request",
                    "trace_id": trace_id,
                    "user": sniff["fields"].get("user"),
                    "method": method,
                    "path": scope.get("path"),
                    "status": status,
                    "duration_ms": duration_ms,
                    "client_ip": client[0] if client else None,
                }
                # business detail sniffed from the body (file_name, config_name, company)
                for _k, _v in sniff["fields"].items():
                    if _k != "user" and _k not in _ev:
                        _ev[_k] = _v
                _record(
                    f"{method} {scope.get('path')} → {status}",
                    6 if status < 400 else (4 if status < 500 else 3),
                    _ev,
                )
            except Exception:
                pass
            _trace_id_ctx.reset(token)
