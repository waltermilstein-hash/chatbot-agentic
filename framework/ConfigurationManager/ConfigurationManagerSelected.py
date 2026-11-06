# =============================================================================
# =============================================================================
"""
Centralized configuration module for the application.
Integrated with the existing YamlFileManager.
Provides an abstraction to manage configurations for:
- Databases: MongoDB, Redis, PostgreSQL, Milvus, Weaviate
- External APIs: OpenAI, Anthropic (Claude), Google Services
- Services: PDF

✨ NEW: support for the Anthropic (Claude) API
✨ NEW: support for access via dot AND bracket notation
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

from dotenv import load_dotenv

from configyml.YamlFileManager import YamlFileManager

load_dotenv()


class MongoDBConfig:
    """Configuration for the MongoDB connection.

    Supports dot and bracket access:
    - config.mongodb.uri
    - config['mongodb']['uri']
    """

    def __init__(self, db: str = 'gen-ai-extractor-pro',
                 uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0"),
                 collection_name: str = '',
                 **kwargs):
        """
        Initialize the MongoDB configuration.

        Args:
            db: Database name
            uri: Connection URI
            collection_name: Collection name (optional)
            **kwargs: Additional fields that will be ignored
        """
        self.db = db
        self.uri = uri
        self.collection_name = collection_name
        # Keep additional fields in case they are needed
        self._extra_fields = kwargs

    def __getitem__(self, key: str):
        """Allows bracket access: config.mongodb['uri']"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Field '{key}' not found in MongoDBConfig. Available fields: db, uri, collection_name")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        result = {
            'db': self.db,
            'uri': self.uri
        }
        if self.collection_name:
            result['collection_name'] = self.collection_name
        # Add extra fields
        result.update(self._extra_fields)
        return result

    def get_connection_string(self) -> str:
        """Get the full connection URI."""
        return self.uri


@dataclass
class RedisConfig:
    """Configuration for the Redis connection.

    Supports dot and bracket access:
    - config.redis.host
    - config['redis']['host']
    """
    host: str = 'localhost'
    port: int = 6380
    decode_responses: bool = True
    encoding: str = 'utf-8'
    username: Optional[str] = None
    password: Optional[str] = None
    db: int = 0

    def __getitem__(self, key: str):
        """Allows bracket access: config.redis['host']"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Field '{key}' not found in RedisConfig")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        config = {
            'host': self.host,
            'port': self.port,
            'decode_responses': self.decode_responses,
            'encoding': self.encoding,
            'db': self.db
        }
        if self.username:
            config['username'] = self.username   # ACL user (stack runs `user default off`)
        if self.password:
            config['password'] = self.password
        return config


@dataclass
class MilvusConfig:
    """Configuration for the Milvus connection.

    Supports dot and bracket access.
    """
    host: str = 'localhost'
    port: str = '19530'

    def __getitem__(self, key: str):
        """Allows bracket access: config.milvus['host']"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Field '{key}' not found in MilvusConfig")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            'host': self.host,
            'port': self.port
        }

    def get_connection_params(self) -> Dict[str, str]:
        """Get the connection parameters."""
        return {
            'host': self.host,
            'port': self.port
        }


@dataclass
class WeaviateConfig:
    """Configuration for the Weaviate connection.

    Supports dot and bracket access.
    """
    url: str = 'http://localhost:8087'

    def __getitem__(self, key: str):
        """Allows bracket access: config.weaviate['url']"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Field '{key}' not found in WeaviateConfig")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {'url': self.url}

    def get_url(self) -> str:
        """Get the connection URL."""
        return self.url


@dataclass
class OpenAIConfig:
    """Configuration for the OpenAI API.

    Supports dot and bracket access.
    """
    key: str = ''
    model_3_5: str = 'gpt-3.5-turbo'
    model_4: str = 'gpt-5.6'

    def __getitem__(self, key: str):
        """Allows bracket access: config.openai['key']"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Field '{key}' not found in OpenAIConfig")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            'key': self.key,
            'model_3.5': self.model_3_5,
            'model_4': self.model_4
        }

    def get_api_key(self) -> str:
        """Get the API key."""
        return self.key


@dataclass
class AnthropicConfig:
    """Configuration for the Anthropic (Claude) API.

    Supports dot and bracket access:
    - config.anthropic.key
    - config['anthropic']['key']

    Available models:
    - claude-opus-4-20250514: Most powerful, best for complex tasks
    - claude-sonnet-4-20250514: Perfect balance (RECOMMENDED)
    - claude-haiku-4-20250514: Fastest and cheapest
    """
    key: str = ''
    api_key: str = ''  # Alias for compatibility
    default_model: str = 'claude-sonnet-4-20250514'
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 120.0

    def __post_init__(self):
        """Post-initialization to handle the alias."""
        # If api_key is set but key is not, use api_key
        if self.api_key and not self.key:
            self.key = self.api_key
        # If key is set but api_key is not, sync them
        elif self.key and not self.api_key:
            self.api_key = self.key

    def __getitem__(self, key: str):
        """Allows bracket access: config.anthropic['key']"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Field '{key}' not found in AnthropicConfig")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            'key': self.key,
            'api_key': self.api_key,
            'default_model': self.default_model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'timeout': self.timeout
        }

    def get_api_key(self) -> str:
        """Get the Anthropic API key."""
        return self.key or self.api_key

    def get_model(self) -> str:
        """Get the default model."""
        return self.default_model


@dataclass
class GoogleConfig:
    """Configuration for Google services.

    Supports dot and bracket access.
    """
    credentials_file: str = 'gen-lang-client-0036662396-1cbb491f62cf.json'
    cx: str = 'gen0'
    api_key_custom_search: str = ''
    api_key_google_map: str = ''
    api_key_gemini: str = ''
    geolocate_url: str = 'https://www.googleapis.com/geolocation/v1/geolocate?key='

    def __getitem__(self, key: str):
        """Allows bracket access: config.google['api_key_gemini']"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Field '{key}' not found in GoogleConfig")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            'credentials_file': self.credentials_file,
            'cx': self.cx,
            'api_key_customSearch': self.api_key_custom_search,
            'api_key_google_map': self.api_key_google_map,
            'api_key_gemini': self.api_key_gemini,
            'geolocate_url': self.geolocate_url
        }

    def get_geolocate_full_url(self) -> str:
        """Get the full geolocation URL."""
        return f"{self.geolocate_url}{self.api_key_google_map}"

    def get_gemini_api_key(self) -> str:
        """Get the Google Gemini API key."""
        return self.api_key_gemini


@dataclass
class AzureConfig:
    """Configuration for Azure services.

    Supports dot and bracket access:
    - config.azure.document_intelligence_endpoint
    - config['azure']['document_intelligence']['endpoint']
    """
    document_intelligence_endpoint: str = ''
    document_intelligence_key: str = ''

    def __getitem__(self, key: str):
        """Allows nested bracket access: config.azure['document_intelligence']['endpoint']"""
        if key == 'document_intelligence':
            # Return a dict for compatibility with legacy code
            return {
                'endpoint': self.document_intelligence_endpoint,
                'key': self.document_intelligence_key
            }
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Field '{key}' not found in AzureConfig")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            'document_intelligence': {
                'endpoint': self.document_intelligence_endpoint,
                'key': self.document_intelligence_key
            }
        }

    def get_document_intelligence_endpoint(self) -> str:
        """Get the Document Intelligence endpoint."""
        return self.document_intelligence_endpoint

    def get_document_intelligence_key(self) -> str:
        """Get the Document Intelligence key."""
        return self.document_intelligence_key

    def get_document_intelligence_credential(self):
        """
        Get the credential for Azure Document Intelligence.
        Requires: from azure.core.credentials import AzureKeyCredential
        """
        try:
            from azure.core.credentials import AzureKeyCredential
            return AzureKeyCredential(self.document_intelligence_key)
        except ImportError:
            raise ImportError(
                "azure-core is not installed. "
                "Install it with: pip install azure-core"
            )


@dataclass
class PDFConfig:
    """Configuration for the PDF service.

    Supports dot and bracket access.
    """
    url: str = 'http://127.0.0.1:8000/download-pdf/'

    def __getitem__(self, key: str):
        """Allows bracket access: config.pdf['url']"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Field '{key}' not found in PDFConfig")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {'url': self.url}

    def get_url(self) -> str:
        """Get the service URL."""
        return self.url


@dataclass
class PostgreSQLConfig:
    """Configuration for the PostgreSQL connection.

    Supports dot and bracket access.
    """
    host: str = 'localhost'
    port: int = 5432
    database: str = 'postgres'
    user: str = 'root'
    password: str = 'root',
    schema: str = 'public'

    def __getitem__(self, key: str):
        """Allows bracket access: config.postgresql['host']"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Field '{key}' not found in PostgreSQLConfig")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.user,
            'password': self.password,
            'schema': self.schema
        }

    def get_connection_string(self) -> str:
        """Build the PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class AppConfig:
    """Main class to manage all application configurations.

    ✨ NEW: supports the Anthropic (Claude) API

    Supports TWO access styles:
    1. Dot notation: config.anthropic.key
    2. Bracket notation: config['anthropic']['key']
    """

    def __init__(
            self,
            config_dict: Optional[Dict[str, Any]] = None,
            mongodb_config: Optional[MongoDBConfig] = None,
            redis_config: Optional[RedisConfig] = None,
            postgresql_config: Optional[PostgreSQLConfig] = None,
            milvus_config: Optional[MilvusConfig] = None,
            weaviate_config: Optional[WeaviateConfig] = None,
            openai_config: Optional[OpenAIConfig] = None,
            anthropic_config: Optional[AnthropicConfig] = None,  # 🆕 NEW
            google_config: Optional[GoogleConfig] = None,
            azure_config: Optional[AzureConfig] = None,
            pdf_config: Optional[PDFConfig] = None
    ):
        if config_dict:
            self._load_from_dict(config_dict)
        else:
            self.mongodb = mongodb_config or MongoDBConfig()
            self.redis = redis_config or RedisConfig()
            self.postgresql = postgresql_config or PostgreSQLConfig()
            self.milvus = milvus_config or MilvusConfig()
            self.weaviate = weaviate_config or WeaviateConfig()
            self.openai = openai_config or OpenAIConfig()
            self.anthropic = anthropic_config or AnthropicConfig()  # 🆕
            self.google = google_config or GoogleConfig()
            self.azure = azure_config or AzureConfig()
            self.pdf = pdf_config or PDFConfig()

    def __getitem__(self, key: str):
        """Allows bracket access: config['anthropic']"""
        key_mappings = {
            'openai_key': 'openai',
        }
        actual_key = key_mappings.get(key, key)
        if hasattr(self, actual_key):
            return getattr(self, actual_key)
        raise KeyError(
            f"Configuration '{key}' not found. "
            f"Available keys: {', '.join(self._get_available_keys())}"
        )

    def _get_available_keys(self):
        """Get the available keys."""
        return ['mongodb', 'redis', 'postgresql', 'milvus', 'weaviate',
                'openai', 'anthropic', 'google', 'azure', 'pdf']

    def _load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load the configurations from a dictionary."""
        self.mongodb = MongoDBConfig(**config_dict.get('mongodb', {}))

        redis_params = config_dict.get('redis', {})
        self.redis = RedisConfig(**redis_params) if redis_params else RedisConfig()

        postgresql_params = config_dict.get('postgresql', {})
        self.postgresql = PostgreSQLConfig(**postgresql_params) if postgresql_params else PostgreSQLConfig()

        self.milvus = MilvusConfig(**config_dict.get('milvus', {}))
        self.weaviate = WeaviateConfig(**config_dict.get('weaviate', {}))

        # OpenAI. P0-3: the OPENAI_APIKEY env var takes precedence over the yml, to
        # stop depending on committed keys (the docker-compose files already export it).
        openai_key = os.getenv('OPENAI_APIKEY') or config_dict.get('openai_key', {}).get('key', '')
        model_config = config_dict.get('config', {})
        self.openai = OpenAIConfig(
            key=openai_key,
            model_3_5=model_config.get('model_3.5', 'gpt-3.5-turbo'),
            model_4=model_config.get('model_4', 'gpt-5.6')
        )

        # 🆕 Anthropic (Claude)
        anthropic_data = config_dict.get('anthropic', {})
        if anthropic_data:
            self.anthropic = AnthropicConfig(
                key=anthropic_data.get('key', ''),
                api_key=anthropic_data.get('api_key', anthropic_data.get('key', '')),
                default_model=anthropic_data.get('default_model', 'claude-sonnet-4-20250514'),
                max_tokens=anthropic_data.get('max_tokens', 4096),
                temperature=anthropic_data.get('temperature', 0.7),
                timeout=anthropic_data.get('timeout', 120.0)
            )
        else:
            self.anthropic = AnthropicConfig()

        # Google
        google_data = config_dict.get('google', {})
        google_credentials = config_dict.get('config', {}).get('google', '')
        # Google API keys are read from .env (precedence); the yaml is a fallback.
        self.google = GoogleConfig(
            credentials_file=google_credentials,
            cx=google_data.get('cx', ''),
            api_key_custom_search=os.getenv('GOOGLE_CUSTOM_SEARCH_KEY', google_data.get('api_key_customSearch', '')),
            api_key_google_map=os.getenv('GOOGLE_MAPS_API_KEY', google_data.get('api_key_google_map', '')),
            api_key_gemini=os.getenv('GEMINI_API_KEY', google_data.get('api_key_gemini', '')),
            geolocate_url=google_data.get('geolocate_url', 'https://www.googleapis.com/geolocation/v1/geolocate?key=')
        )

        # Azure
        azure_data = config_dict.get('azure', {})
        doc_intel = azure_data.get('document_intelligence', {})
        self.azure = AzureConfig(
            document_intelligence_endpoint=doc_intel.get('endpoint', ''),
            document_intelligence_key=doc_intel.get('key', '')
        )

        # PDF
        pdf_data = config_dict.get('pdf', {})
        self.pdf = PDFConfig(url=pdf_data.get('url', 'http://127.0.0.1:8000/download-pdf/'))

    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all configurations."""
        return {
            'mongodb': self.mongodb.to_dict(),
            'redis': self.redis.to_dict(),
            'postgresql': self.postgresql.to_dict(),
            'milvus': self.milvus.to_dict(),
            'weaviate': self.weaviate.to_dict(),
            'openai': self.openai.to_dict(),
            'anthropic': self.anthropic.to_dict(),  # 🆕
            'google': self.google.to_dict(),
            'azure': self.azure.to_dict(),
            'pdf': self.pdf.to_dict()
        }


class ConfigurationManagerSelected:
    """Configuration manager with support for Anthropic."""

    def __init__(self, config_name: Optional[str] = None, lazy_load: bool = False):
        self.config_name = config_name or "manager_extractor"
        self.configurations_map = self._load_configurations_map()
        self.config = None
        if config_name is not None and not lazy_load:
            self.config = self._load_configuration()

    def _load_configurations_map(self) -> Dict[str, str]:
        # Environment override: CHATBOT_CONFIG_YML lets you point to a sanitized
        # config (127.0.0.7) without touching the host's shared config.
        override = os.getenv("CHATBOT_CONFIG_YML")
        if override:
            return {"manager_extractor": override, "default": override}
        return {
            "manager_extractor": "/config/managerextractor-pro/config.yml",
            "default": "/config/managerextractor-pro/config.yml",
        }

    def _load_configuration(self) -> AppConfig:
        # Override: with CHATBOT_CONFIG_YML, ANY profile (application) uses the
        # sanitized config → no need to register each 'application'.
        override = os.getenv("CHATBOT_CONFIG_YML")
        if override:
            config_file = override
            try:
                yaml_manager = YamlFileManager(config_file)
                return AppConfig(config_dict=yaml_manager.load())
            except Exception as e:
                raise RuntimeError(f"Error loading configuration: {str(e)}") from e

        if self.config_name not in self.configurations_map:
            raise ValueError(f"Configuration '{self.config_name}' not found.")

        try:
            config_file = self.configurations_map[self.config_name]
            yaml_manager = YamlFileManager(config_file)
            config_dict = yaml_manager.load()
            return AppConfig(config_dict=config_dict)
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {str(e)}") from e

    def reload_configuration(self, config_name: Optional[str] = None) -> AppConfig:
        if config_name:
            self.config_name = config_name
        self.config = self._load_configuration()
        return self.config

    def load_configuration(self, config_name: str) -> AppConfig:
        return self.reload_configuration(config_name)
