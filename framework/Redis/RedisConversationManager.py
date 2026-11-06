"""
"""
import redis.asyncio as redis
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class RedisConversationManager:
    """
    Conversation history manager using Redis.
    Optimized for Ghost Attention in RAG systems.
    """

    def __init__(
            self,
            host: str = 'localhost',
            port: int = 6380,
            db: int = 0,
            username: Optional[str] = None,
            password: Optional[str] = None,
            max_retries: int = 3,
            socket_timeout: int = 5
    ):
        """
        Initialize the Redis client for conversations.

        Args:
            host: Redis host
            port: Redis port (default 6380)
            db: Redis database number
            password: Password if Redis requires auth
            max_retries: Connection retries
            socket_timeout: Socket timeout in seconds
        """
        self.host = host
        self.port = port
        self.db = db
        self.username = username
        self.password = password
        self.max_retries = max_retries
        self.socket_timeout = socket_timeout
        self.redis_client = None
        self._initialized = False

    async def initialize(self):
        """Initialize the Redis connection asynchronously."""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                username=self.username,
                password=self.password,
                decode_responses=True,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_timeout,
                retry_on_timeout=True,
                max_connections=50
            )

            # Connection test
            await self.redis_client.ping()
            self._initialized = True
            print(f"[Redis] ✓ Connected to {self.host}:{self.port} (db={self.db})")
            return True

        except Exception as e:
            print(f"[Redis] ✗ Connection error: {e}")
            self.redis_client = None
            self._initialized = False
            return False

    def _get_conversation_key(self, conversation_id: str) -> str:
        """Build the Redis key for a conversation."""
        return f"conversation:{conversation_id}"

    def _get_metadata_key(self, conversation_id: str) -> str:
        """Build the Redis key for conversation metadata."""
        return f"conversation:meta:{conversation_id}"

    async def save_message(
            self,
            conversation_id: str,
            user_message: str,
            bot_response: str,
            user: str = None,
            timestamp=None,  # Accepts datetime, str or None
            metadata: Dict = None,
            ttl_hours: int = 24
    ) -> bool:
        """
        Save a message in the conversation history.

        Args:
            conversation_id: Unique conversation ID
            user_message: User message
            bot_response: Bot response
            user: User identifier (optional)
            timestamp: Message timestamp (datetime, ISO str or None)
            metadata: Additional data (optional)
            ttl_hours: Hours before expiring (default: 24h)

        Returns:
            True if it was saved successfully
        """
        if not self._initialized or not self.redis_client:
            print("[Redis] Client not initialized")
            return False

        try:
            # Normalize timestamp to ISO string
            if timestamp is None:
                timestamp_str = datetime.utcnow().isoformat()
            elif isinstance(timestamp, str):
                # Already a string, validate the format
                try:
                    datetime.fromisoformat(timestamp)  # Validate
                    timestamp_str = timestamp
                except:
                    timestamp_str = datetime.utcnow().isoformat()
            elif hasattr(timestamp, 'isoformat'):  # It's a datetime
                timestamp_str = timestamp.isoformat()
            else:
                # Unknown type, use the current timestamp
                timestamp_str = datetime.utcnow().isoformat()

            # Build the message object (all serializable)
            message = {
                "user_message": user_message,
                "bot_response": bot_response,
                "timestamp": timestamp_str,  # String, not datetime
                "usuario": user
            }

            # Add metadata if present
            if metadata:
                message["metadata"] = metadata

            redis_key = self._get_conversation_key(conversation_id)

            # Save the message in a Redis list
            await self.redis_client.rpush(
                redis_key,
                json.dumps(message, ensure_ascii=False)
            )

            # Limit to the last 10 messages
            await self.redis_client.ltrim(redis_key, -10, -1)

            # Set TTL
            await self.redis_client.expire(redis_key, ttl_hours * 3600)

            # Save conversation metadata
            await self._save_conversation_metadata(
                conversation_id,
                user,
                ttl_hours
            )

            # Verify
            count = await self.redis_client.llen(redis_key)
            print(f"[Redis] ✓ Message saved in {conversation_id} ({count} total)")

            return True

        except Exception as e:
            print(f"[Redis] Error saving message: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _save_conversation_metadata(
            self,
            conversation_id: str,
            user: str,
            ttl_hours: int
    ):
        """Save the conversation metadata."""
        try:
            meta_key = self._get_metadata_key(conversation_id)
            metadata = {
                "conversation_id": conversation_id,
                "usuario": user,
                "last_updated": datetime.utcnow().isoformat(),
                "created_at": datetime.utcnow().isoformat()
            }

            # Only update created_at if it does not exist
            existing = await self.redis_client.get(meta_key)
            if existing:
                existing_meta = json.loads(existing)
                metadata["created_at"] = existing_meta.get("created_at")

            await self.redis_client.setex(
                meta_key,
                ttl_hours * 3600,
                json.dumps(metadata, ensure_ascii=False)
            )

        except Exception as e:
            print(f"[Redis] Error saving metadata: {e}")

    async def get_conversation_history(
            self,
            conversation_id: str,
            limit: int = 5
    ) -> List[Dict]:
        """
        Retrieve the conversation history.

        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages to retrieve

        Returns:
            List of messages ordered chronologically
        """
        if not self._initialized or not self.redis_client:
            print("[Redis] Client not initialized")
            return []

        try:
            redis_key = self._get_conversation_key(conversation_id)

            # Get the last N messages
            messages_raw = await self.redis_client.lrange(redis_key, -limit, -1)

            if not messages_raw:
                print(f"[Redis] No history for: {conversation_id}")
                return []

            # Parse JSON
            history = []
            for msg_json in messages_raw:
                try:
                    history.append(json.loads(msg_json))
                except json.JSONDecodeError as e:
                    print(f"[Redis] Error parsing message: {e}")
                    continue

            print(f"[Redis] ✓ Retrieved {len(history)} messages")
            return history

        except Exception as e:
            print(f"[Redis] Error retrieving history: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def get_conversation_metadata(
            self,
            conversation_id: str
    ) -> Optional[Dict]:
        """Retrieve a conversation's metadata."""
        if not self._initialized or not self.redis_client:
            return None

        try:
            meta_key = self._get_metadata_key(conversation_id)
            meta_json = await self.redis_client.get(meta_key)

            if not meta_json:
                return None

            return json.loads(meta_json)

        except Exception as e:
            print(f"[Redis] Error retrieving metadata: {e}")
            return None

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a full conversation."""
        if not self._initialized or not self.redis_client:
            return False

        try:
            redis_key = self._get_conversation_key(conversation_id)
            meta_key = self._get_metadata_key(conversation_id)

            await self.redis_client.delete(redis_key, meta_key)
            print(f"[Redis] ✓ Conversation {conversation_id} deleted")
            return True

        except Exception as e:
            print(f"[Redis] Error deleting conversation: {e}")
            return False

    async def get_message_count(self, conversation_id: str) -> int:
        """Get the number of messages in a conversation."""
        if not self._initialized or not self.redis_client:
            return 0

        try:
            redis_key = self._get_conversation_key(conversation_id)
            count = await self.redis_client.llen(redis_key)
            return count
        except Exception as e:
            print(f"[Redis] Error counting messages: {e}")
            return 0

    async def extend_ttl(
            self,
            conversation_id: str,
            additional_hours: int = 24
    ) -> bool:
        """Extend a conversation's TTL."""
        if not self._initialized or not self.redis_client:
            return False

        try:
            redis_key = self._get_conversation_key(conversation_id)
            meta_key = self._get_metadata_key(conversation_id)

            await self.redis_client.expire(redis_key, additional_hours * 3600)
            await self.redis_client.expire(meta_key, additional_hours * 3600)

            print(f"[Redis] ✓ TTL extended {additional_hours}h for {conversation_id}")
            return True

        except Exception as e:
            print(f"[Redis] Error extending TTL: {e}")
            return False

    async def get_active_conversations(
            self,
            user: str = None,
            limit: int = 100
    ) -> List[str]:
        """
        Get the list of active conversations.

        Args:
            user: Filter by user (optional)
            limit: Results limit

        Returns:
            List of conversation_ids
        """
        if not self._initialized or not self.redis_client:
            return []

        try:
            # Scan all conversation keys
            pattern = "conversation:*"
            conversation_ids = []

            async for key in self.redis_client.scan_iter(match=pattern, count=100):
                if key.startswith("conversation:meta:"):
                    continue

                conv_id = key.replace("conversation:", "")

                # If a user is specified, filter
                if user:
                    meta = await self.get_conversation_metadata(conv_id)
                    if meta and meta.get("usuario") == user:
                        conversation_ids.append(conv_id)
                else:
                    conversation_ids.append(conv_id)

                if len(conversation_ids) >= limit:
                    break

            return conversation_ids

        except Exception as e:
            print(f"[Redis] Error getting active conversations: {e}")
            return []

    async def health_check(self) -> bool:
        """Check that Redis is working."""
        try:
            if not self.redis_client:
                return False
            await self.redis_client.ping()
            return True
        except Exception as e:
            print(f"[Redis] Health check failed: {e}")
            return False

    async def close(self):
        """Close the Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            print("[Redis] ✓ Connection closed")
            self._initialized = False


# ========================================
# USAGE EXAMPLE
# ========================================
async def ejemplo_uso():
    """Example of how to use the class."""

    # 1. Create an instance
    redis_manager = RedisConversationManager(
        host='localhost',
        port=6388,
        password=None  # If you have a password, add it here
    )

    # 2. Initialize
    if not await redis_manager.initialize():
        print("Could not connect to Redis")
        return

    # 3. Save messages
    conversation_id = "conv-12345"

    await redis_manager.save_message(
        conversation_id=conversation_id,
        user_message="What are this month's purchases?",
        bot_response="<HTML table with purchases...>",
        user="user123",
        ttl_hours=24
    )

    await redis_manager.save_message(
        conversation_id=conversation_id,
        user_message="What is the highest price?",
        bot_response="VIRGEN CLASICO OLIOVITA: 12.389 ARS",
        user="user123"
    )

    # 4. Retrieve the history
    history = await redis_manager.get_conversation_history(
        conversation_id=conversation_id,
        limit=5
    )

    print(f"\nRetrieved history: {len(history)} messages")
    for i, msg in enumerate(history, 1):
        print(f"\nMessage {i}:")
        print(f"  User: {msg['user_message'][:50]}...")
        print(f"  Bot: {msg['bot_response'][:50]}...")

    # 5. Close the connection
    await redis_manager.close()


# if __name__ == "__main__":
#     import asyncio
#
#     asyncio.run(ejemplo_uso())
