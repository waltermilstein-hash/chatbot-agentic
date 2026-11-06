# =============================================================================
"""
Shared helper to obtain the application's AppConfig (Mongo, etc.).

Centralizes configuration loading so the user/role managers are built the same
way as the rest of the backend:
    ConfigurationManagerSelected().load_configuration("manager_extractor")
"""
import os

from framework.ConfigurationManager.ConfigurationManagerSelected import (
    ConfigurationManagerSelected,
)

# Config app name (YAML path with mongodb.uri/db). Same as the rest of the
# backend endpoints; overridable via environment.
DEFAULT_APP_CONFIG = os.getenv("APP_CONFIG", "manager_extractor")


def get_app_config():
    return ConfigurationManagerSelected().load_configuration(DEFAULT_APP_CONFIG)
