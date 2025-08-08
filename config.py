"""
Configuration module for literature mining project.

This module handles all environment variable loading and provides
a centralized configuration interface.
"""

import os
from typing import Optional
from dotenv import load_dotenv


class Config:
    """Configuration class that loads and validates environment variables."""

    def __init__(self, dotenv_path: Optional[str] = None):
        """
        Initialize configuration by loading environment variables.

        Args:
            dotenv_path: Optional path to .env file. If None, will look for .env in current directory.
        """
        # Load environment variables from .env file
        load_dotenv(dotenv_path)

        # Load and validate required environment variables
        self._email = self._get_required_env("EMAIL")
        self._openai_api_key = self._get_required_env("OPENAI_API_KEY")

        # Optional Elsevier API key for full text access
        self._elsevier_api_key = os.getenv("ELSEVIER_API_KEY")

        # Optional environment variables with defaults
        self._base_dir = os.getenv("BASE_DIR", "articles_data")
        self._max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self._request_timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))
        self._plot_detection_confidence = float(
            os.getenv("PLOT_DETECTION_CONFIDENCE", "0.75")
        )

        # OpenAI specific settings
        self._openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self._openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))
        self._openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))

    def _get_required_env(self, var_name: str) -> str:
        """
        Get required environment variable and raise error if not found.

        Args:
            var_name: Name of the environment variable

        Returns:
            Value of the environment variable

        Raises:
            ValueError: If the environment variable is not set
        """
        value = os.getenv(var_name)
        if not value:
            raise ValueError(
                f"{var_name} environment variable is not set. "
                f"Please set it in your .env file."
            )
        return value

    @property
    def email(self) -> str:
        """Email address for PubMed API access."""
        return self._email

    @property
    def openai_api_key(self) -> str:
        """OpenAI API key for LLM processing."""
        return self._openai_api_key

    @property
    def elsevier_api_key(self) -> Optional[str]:
        """Elsevier API key for full text access."""
        return self._elsevier_api_key

    @property
    def base_dir(self) -> str:
        """Base directory for storing article data."""
        return self._base_dir

    @property
    def max_retries(self) -> int:
        """Maximum number of retries for HTTP requests."""
        return self._max_retries

    @property
    def request_timeout(self) -> int:
        """Request timeout in seconds."""
        return self._request_timeout

    @property
    def plot_detection_confidence(self) -> float:
        """Confidence threshold for plot detection."""
        return self._plot_detection_confidence

    @property
    def openai_model(self) -> str:
        """OpenAI model to use for LLM processing."""
        return self._openai_model

    @property
    def openai_max_tokens(self) -> int:
        """Maximum tokens for OpenAI API calls."""
        return self._openai_max_tokens

    @property
    def openai_temperature(self) -> float:
        """Temperature setting for OpenAI API calls."""
        return self._openai_temperature

    def validate(self) -> bool:
        """
        Validate that all required configuration is present.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required variables
            required_vars = [self.email, self.openai_api_key]

            # Check that values are not empty
            for var in required_vars:
                if not var or not var.strip():
                    return False

            # Validate numeric ranges
            if self.max_retries < 0:
                return False

            if self.request_timeout <= 0:
                return False

            if not (0.0 <= self.plot_detection_confidence <= 1.0):
                return False

            if self.openai_max_tokens <= 0:
                return False

            if not (0.0 <= self.openai_temperature <= 2.0):
                return False

            return True

        except Exception:
            return False

    def get_openai_config(self) -> dict:
        """
        Get OpenAI configuration as a dictionary.

        Returns:
            Dictionary with OpenAI configuration parameters
        """
        return {
            "api_key": self.openai_api_key,
            "model": self.openai_model,
            "max_tokens": self.openai_max_tokens,
            "temperature": self.openai_temperature,
        }

    def print_config(self, hide_secrets: bool = True) -> None:
        """
        Print current configuration (for debugging).

        Args:
            hide_secrets: If True, will mask sensitive information like API keys
        """
        print("Current Configuration:")
        print(f"  Email: {self.email}")

        if hide_secrets:
            api_key_display = (
                f"{self.openai_api_key[:8]}..."
                if len(self.openai_api_key) > 8
                else "***"
            )
            print(f"  OpenAI API Key: {api_key_display}")

            if self.elsevier_api_key:
                elsevier_key_display = (
                    f"{self.elsevier_api_key[:8]}..."
                    if len(self.elsevier_api_key) > 8
                    else "***"
                )
                print(f"  Elsevier API Key: {elsevier_key_display}")
            else:
                print("  Elsevier API Key: Not configured")
        else:
            print(f"  OpenAI API Key: {self.openai_api_key}")
            print(f"  Elsevier API Key: {self.elsevier_api_key or 'Not configured'}")

        print(f"  Base Directory: {self.base_dir}")
        print(f"  Max Retries: {self.max_retries}")
        print(f"  Request Timeout: {self.request_timeout}s")
        print(f"  Plot Detection Confidence: {self.plot_detection_confidence}")
        print(f"  OpenAI Model: {self.openai_model}")
        print(f"  OpenAI Max Tokens: {self.openai_max_tokens}")
        print(f"  OpenAI Temperature: {self.openai_temperature}")


# Global configuration instance
# This will be initialized when the module is imported
_config = None


def get_config(dotenv_path: Optional[str] = None) -> Config:
    """
    Get the global configuration instance.

    Args:
        dotenv_path: Optional path to .env file

    Returns:
        Configuration instance
    """
    global _config
    if _config is None:
        _config = Config(dotenv_path)
    return _config


def reload_config(dotenv_path: Optional[str] = None) -> Config:
    """
    Reload the configuration (useful for testing or config changes).

    Args:
        dotenv_path: Optional path to .env file

    Returns:
        New configuration instance
    """
    global _config
    _config = Config(dotenv_path)
    return _config


# Convenience functions for easy access to common config values
def get_email() -> str:
    """Get email from configuration."""
    return get_config().email


def get_openai_api_key() -> str:
    """Get OpenAI API key from configuration."""
    return get_config().openai_api_key


def get_elsevier_api_key() -> Optional[str]:
    """Get Elsevier API key from configuration."""
    return get_config().elsevier_api_key


def get_base_dir() -> str:
    """Get base directory from configuration."""
    return get_config().base_dir


def get_openai_config() -> dict:
    """Get OpenAI configuration dictionary."""
    return get_config().get_openai_config()
