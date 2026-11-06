# =============================================================================
# =============================================================================
"""
UserManager: user CRUD over the Mongo `users` collection.

User document:
    {
        _id, email (unique), name, lastname, password (bcrypt),
        active (bool), roles (list[str]),
        otp, otp_generated_at (epoch s),
        token_pass, token_pass_generated_at (epoch s),
        created_at, updated_at
    }
"""
import time
from typing import List, Optional

from framework.Database.MongoDb.MongoDb import MongoDBManager
from framework.Users._config import get_app_config

COLLECTION = "users"


class UserManager:
    def __init__(self, config=None, collection: str = COLLECTION):
        self.config = config or get_app_config()
        self.mongo = MongoDBManager(collection, self.config)
        self.collection_name = collection

    # ------------------------- indexes -------------------------
    def ensure_indexes(self) -> None:
        """Create the unique index on email (idempotent)."""
        self.mongo.collection.create_index("email", unique=True)

    # ------------------------- reads -------------------------
    def get_by_email(self, email: str) -> Optional[dict]:
        if not email:
            return None
        return self.mongo.find_one_query({"email": email.lower().strip()})

    def exists_by_email(self, email: str) -> bool:
        return self.get_by_email(email) is not None

    def list_all(self) -> List[dict]:
        return self.mongo.find_all({})

    def list_by_company(self, company: str) -> List[dict]:
        return self.mongo.find_all({"company": company})

    @staticmethod
    def to_public(doc: Optional[dict]) -> Optional[dict]:
        """Public projection (without password/otp/tokens)."""
        if not doc:
            return None
        return {
            "id": str(doc.get("_id")) if doc.get("_id") is not None else None,
            "email": doc.get("email"),
            "name": doc.get("name"),
            "lastname": doc.get("lastname"),
            "active": bool(doc.get("active", False)),
            "blocked": bool(doc.get("blocked", False)),
            "company": doc.get("company"),
            "roles": list(doc.get("roles") or []),
            # Admin who created this user (empty for self-registered users). The
            # frontend uses it to know which users this admin may audit.
            "created_by": doc.get("created_by"),
            # Token quota: total assigned and consumed. quota 0/None = unlimited.
            "token_quota": int(doc.get("token_quota") or 0),
            "tokens_used": int(doc.get("tokens_used") or 0),
            # TOTP 2FA status (authenticator app). The secret is NEVER exposed,
            # only whether the user has the second factor enabled.
            "totp_enabled": bool(doc.get("totp_enabled", False)),
            # How many backup codes remain unused (never the hashes).
            "backup_codes_remaining": sum(
                1 for c in (doc.get("backup_codes") or []) if not c.get("used")
            ),
        }

    # ------------------------- writes -------------------------
    def create_user(
        self,
        email: str,
        password_hash: str,
        name: Optional[str] = None,
        lastname: Optional[str] = None,
        roles: Optional[List[str]] = None,
        active: bool = False,
        created_by: Optional[str] = None,
    ) -> str:
        now = time.time()
        doc = {
            "email": email.lower().strip(),
            "name": name,
            "lastname": lastname,
            "password": password_hash,
            "active": active,
            "roles": list(roles or []),
            # Admin who provisioned the account (None for self-registration via /auth/register).
            "created_by": (created_by or "").strip().lower() or None,
            "otp": None,
            "otp_generated_at": None,
            "token_pass": None,
            "token_pass_generated_at": None,
            # TOTP 2FA: secret encrypted at rest (via secrets_vault) and an enabled
            # flag. Enrolled on first login and confirmed with a code.
            "totp_secret": None,
            "totp_enabled": False,
            # Single-use backup codes (hashes only) and trusted devices
            # (hashes only + expiration). Never stored in clear text in the DB.
            "backup_codes": [],
            "trusted_devices": [],
            "created_at": now,
            "updated_at": now,
        }
        return self.mongo.insert_one(doc)

    def _update_by_email(self, email: str, fields: dict) -> None:
        fields = dict(fields)
        fields["updated_at"] = time.time()
        self.mongo.update_one_set({"email": email.lower().strip()}, fields, upsert=False)

    def set_active(self, email: str, active: bool = True) -> None:
        self._update_by_email(email, {"active": active})

    def set_blocked(self, email: str, blocked: bool = True) -> None:
        self._update_by_email(email, {"blocked": blocked})

    def set_roles(self, email: str, roles: List[str]) -> None:
        self._update_by_email(email, {"roles": sorted(set(roles or []))})

    # ------------------------- token quota -------------------------
    def set_token_quota(self, email: str, quota: int) -> None:
        """Set the total number of allowed tokens (0 = unlimited)."""
        self._update_by_email(email, {"token_quota": int(max(0, quota))})

    def reset_tokens_used(self, email: str) -> None:
        self._update_by_email(email, {"tokens_used": 0})

    def add_tokens_used(self, email: str, amount: int) -> int:
        """Atomically increment usage and return the new total used.

        Returns -1 if the document could not be read after updating.
        """
        if not amount or amount <= 0:
            doc = self.get_by_email(email)
            return int(doc.get("tokens_used", 0)) if doc else -1
        self.mongo.collection.update_one(
            {"email": email.lower().strip()},
            {"$inc": {"tokens_used": int(amount)}, "$set": {"updated_at": time.time()}},
        )
        doc = self.get_by_email(email)
        return int(doc.get("tokens_used", 0)) if doc else -1

    def set_otp(self, email: str, otp: str, generated_at: Optional[float] = None) -> None:
        self._update_by_email(
            email,
            {"otp": otp, "otp_generated_at": generated_at if generated_at is not None else time.time()},
        )

    def clear_otp(self, email: str) -> None:
        self._update_by_email(email, {"otp": None, "otp_generated_at": None})

    # ------------------------- 2FA (TOTP) -------------------------
    def set_totp_secret(self, email: str, encrypted_secret: Optional[str], enabled: bool = False) -> None:
        """Store the TOTP secret (already ENCRYPTED by the service layer) and the enabled
        flag. During enrollment `enabled` is False until a code is confirmed."""
        self._update_by_email(email, {"totp_secret": encrypted_secret, "totp_enabled": bool(enabled)})

    def set_totp_enabled(self, email: str, enabled: bool = True) -> None:
        self._update_by_email(email, {"totp_enabled": bool(enabled)})

    def clear_totp(self, email: str) -> None:
        """Disable 2FA and delete the secret."""
        self._update_by_email(email, {"totp_secret": None, "totp_enabled": False})

    def reset_2fa(self, email: str) -> None:
        """Administrative 2FA reset: delete secret, flag and backup codes.
        After this the user re-enrolls TOTP on their next login."""
        self._update_by_email(email, {
            "totp_secret": None,
            "totp_enabled": False,
            "backup_codes": [],
        })

    # ------------------------- backup codes -------------------------
    def set_backup_codes(self, email: str, stored_codes: List[dict]) -> None:
        """Store the list of backup codes (hashes only + `used` flag)."""
        self._update_by_email(email, {"backup_codes": list(stored_codes or [])})

    def mark_backup_code_used(self, email: str, index: int) -> None:
        """Mark the backup code at position `index` as used."""
        user = self.get_by_email(email)
        codes = list((user or {}).get("backup_codes") or [])
        if 0 <= index < len(codes):
            codes[index] = {**codes[index], "used": True}
            self._update_by_email(email, {"backup_codes": codes})

    # ------------------------- trusted devices (remember device) -------------------------
    def add_trusted_device(self, email: str, entry: dict, keep: List[dict] = None) -> None:
        """Add a trusted device (hash + expiration). `keep` lets you pass the
        already-pruned list (without expired ones) so it doesn't grow unbounded."""
        base = list(keep) if keep is not None else list((self.get_by_email(email) or {}).get("trusted_devices") or [])
        base.append(entry)
        self._update_by_email(email, {"trusted_devices": base})

    def set_trusted_devices(self, email: str, devices: List[dict]) -> None:
        self._update_by_email(email, {"trusted_devices": list(devices or [])})

    def clear_trusted_devices(self, email: str) -> None:
        """Revoke all of the user's trusted devices."""
        self._update_by_email(email, {"trusted_devices": []})

    def set_reset_token(self, email: str, token: str, generated_at: Optional[float] = None) -> None:
        self._update_by_email(
            email,
            {
                "token_pass": token,
                "token_pass_generated_at": generated_at if generated_at is not None else time.time(),
            },
        )

    def update_password(self, email: str, password_hash: str) -> None:
        # When the password changes we invalidate the reset token.
        self._update_by_email(
            email,
            {"password": password_hash, "token_pass": None, "token_pass_generated_at": None},
        )

    def update_profile(
        self,
        email: str,
        name: Optional[str] = None,
        lastname: Optional[str] = None,
        new_email: Optional[str] = None,
    ) -> None:
        fields = {}
        if name is not None:
            fields["name"] = name
        if lastname is not None:
            fields["lastname"] = lastname
        if new_email:
            fields["email"] = new_email.lower().strip()
        if fields:
            self._update_by_email(email, fields)

    def add_role(self, email: str, role_name: str) -> None:
        user = self.get_by_email(email)
        roles = set(user.get("roles") or []) if user else set()
        roles.add(role_name)
        self._update_by_email(email, {"roles": sorted(roles)})
