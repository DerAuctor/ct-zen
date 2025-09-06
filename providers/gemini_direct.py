"""
Gemini Direct Provider for Zen MCP Server
Hybrid implementation combining:
1. Official Gemini REST API (generativelanguage.googleapis.com)
2. Google Code Assist API (cloudcode-pa.googleapis.com) - like gemini-cli
Supports both API Key and OAuth2 authentication with automatic fallback
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base import ModelCapabilities, ModelProvider, ModelResponse, ProviderType, create_temperature_constraint

logger = logging.getLogger(__name__)


class GeminiRateLimitError(Exception):
    """Exception raised when Gemini API rate limit is exceeded"""


class GeminiOAuth2Manager:
    """Handles OAuth2 token management for Gemini API (from Anubis_dev)"""

    def __init__(self):
        self.credentials: Optional[Dict[str, Any]] = None
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.expiry_date: Optional[int] = None

        # OAuth2 Configuration (using environment variables or defaults)
        self.client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "your-client-id-here")
        self.client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "your-client-secret-here")
        self.refresh_url = "https://oauth2.googleapis.com/token"
        self.creds_path = Path.home() / ".gemini" / "oauth_creds.json"

    def load_credentials(self) -> bool:
        """Load OAuth2 credentials from gemini CLI file"""
        try:
            if not self.creds_path.exists():
                logger.debug("OAuth credentials file not found: %s", self.creds_path)
                return False

            with open(self.creds_path, "r", encoding="utf-8") as f:
                self.credentials = json.load(f)

            if self.credentials is not None:
                self.access_token = self.credentials.get("access_token")
                self.refresh_token = self.credentials.get("refresh_token")
                self.expiry_date = self.credentials.get("expiry_date")

            logger.info("OAuth2 credentials loaded successfully")
            return True

        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug("Failed to load OAuth2 credentials: %s", e)
            return False

    def is_token_expired(self) -> bool:
        """Check if the access token is expired"""
        if not self.expiry_date:
            return True
        # Add 5 minute buffer
        return time.time() * 1000 > (self.expiry_date - 300000)

    def refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token"""
        if not self.refresh_token:
            return False

        try:
            response = requests.post(
                self.refresh_url,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": self.refresh_token,
                    "grant_type": "refresh_token",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )

            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data["access_token"]
                # Update expiry (typically 1 hour)
                self.expiry_date = int(time.time() * 1000) + (3600 * 1000)

                # Update credentials file
                if self.credentials is not None:
                    self.credentials["access_token"] = self.access_token
                    self.credentials["expiry_date"] = self.expiry_date

                    with open(self.creds_path, "w", encoding="utf-8") as f:
                        json.dump(self.credentials, f, indent=2)

                logger.info("Access token refreshed successfully")
                return True
        except Exception as e:
            logger.error("Token refresh error: %s", e)
        return False

    def get_valid_token(self) -> Optional[str]:
        """Get a valid access token, refreshing if necessary"""
        if not self.access_token and not self.load_credentials():
            return None

        if self.is_token_expired() and not self.refresh_access_token():
            return None

        return self.access_token


class GeminiDirectProvider(ModelProvider):
    """
    Enhanced Gemini Direct Provider supporting both OAuth2 and API Key authentication
    Combines Anubis_dev Code Assist API with official REST API
    """

    # Supported models with capabilities
    SUPPORTED_MODELS = {
        "gemini-2.5-pro": ModelCapabilities(
            provider=ProviderType.GOOGLE,
            model_name="gemini-2.5-pro",
            friendly_name="Gemini 2.5 Pro",
            context_window=32768,
            max_output_tokens=8192,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=False,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            description="Google's most advanced Gemini model with extended thinking",
            max_thinking_tokens=24000,
            temperature_constraint=create_temperature_constraint("range"),
        ),
        "gemini-2.5-flash": ModelCapabilities(
            provider=ProviderType.GOOGLE,
            model_name="gemini-2.5-flash",
            friendly_name="Gemini 2.5 Flash",
            context_window=32768,
            max_output_tokens=8192,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=False,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            description="Fast Gemini model with thinking capabilities",
            max_thinking_tokens=16000,
            temperature_constraint=create_temperature_constraint("range"),
        ),
        "gemini-2.5-flash-lite": ModelCapabilities(
            provider=ProviderType.GOOGLE,
            model_name="gemini-2.5-flash-lite",
            friendly_name="Gemini 2.5 Flash Lite",
            context_window=32768,
            max_output_tokens=8192,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=False,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            description="Lightweight fast Gemini model",
            temperature_constraint=create_temperature_constraint("range"),
        ),
        "gemini-2.0-flash": ModelCapabilities(
            provider=ProviderType.GOOGLE,
            model_name="gemini-2.0-flash",
            friendly_name="Gemini 2.0 Flash",
            context_window=8192,
            max_output_tokens=4096,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=False,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            description="Previous generation fast Gemini model",
            temperature_constraint=create_temperature_constraint("range"),
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)

        # Authentication setup - support both methods
        self.auth_method = kwargs.get("auth_method", "api_key")  # "api_key" or "oauth2"
        self.oauth_manager = GeminiOAuth2Manager() if self.auth_method == "oauth2" else None

        # API endpoints
        self.rest_base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.code_assist_endpoint = "https://cloudcode-pa.googleapis.com"
        self.code_assist_version = "v1internal"

        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.project_id: Optional[str] = None

    async def _ensure_session(self):
        """Ensure aiohttp session is available"""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get appropriate authentication headers based on auth method"""
        if self.auth_method == "oauth2" and self.oauth_manager:
            token = self.oauth_manager.get_valid_token()
            if not token:
                raise ValueError("Failed to get valid OAuth2 token")
            return {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        else:
            if not self.api_key:
                raise ValueError("API key required for API key authentication")
            return {
                "Content-Type": "application/json",
            }

    async def _discover_project_id(self) -> str:
        """Discover Google Cloud project ID for OAuth2 mode"""
        if self.project_id:
            return self.project_id

        if self.auth_method != "oauth2" or not self.oauth_manager:
            return "default-project"

        try:
            await self._ensure_session()
            headers = await self._get_auth_headers()
            load_data = {
                "cloudaicompanionProject": "default-project",
                "metadata": {"duetProject": "default-project"},
            }

            if self.session is None:
                raise ValueError("Session not initialized")
            async with self.session.post(
                f"{self.code_assist_endpoint}/{self.code_assist_version}:loadCodeAssist",
                headers=headers,
                json=load_data,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    project_id = data.get("cloudaicompanionProject")
                    if project_id:
                        self.project_id = project_id
                        return project_id

        except Exception as e:
            logger.warning("Project discovery failed: %s", e)

        self.project_id = "default-project"
        return "default-project"

    def _messages_to_contents(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert messages to Gemini API format"""
        contents = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                # Gemini doesn't have system role, add as user message
                contents.append({"role": "user", "parts": [{"text": f"System: {content}"}]})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})

        return contents

    async def _call_rest_api(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Call the official Gemini REST API"""
        await self._ensure_session()

        contents = self._messages_to_contents(messages)
        url = f"{self.rest_base_url}/models/{model}:generateContent"

        if self.auth_method == "api_key":
            url += f"?key={self.api_key}"

        generation_config = {
            "temperature": kwargs.get("temperature", 0.7),
            "maxOutputTokens": kwargs.get("max_tokens", 8192),
        }

        # Add thinking budget if supported
        capabilities = self.get_capabilities(model)
        if capabilities.supports_extended_thinking and kwargs.get("thinking_budget"):
            thinking_budget = kwargs.get("thinking_budget", capabilities.max_thinking_tokens)
            if thinking_budget > 0:
                generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}

        payload = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        headers = await self._get_auth_headers()

        if self.session is None:
            raise ValueError("Session not initialized")
        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                if response.status == 429:
                    raise GeminiRateLimitError(f"Rate limit exceeded: {error_text}")
                raise ValueError(f"REST API call failed: {response.status} - {error_text}")

            data = await response.json()
            return self._extract_content_from_response(data)

    async def _call_code_assist_api(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Call the Code Assist API (OAuth2 only)"""
        if self.auth_method != "oauth2" or not self.oauth_manager:
            raise ValueError("Code Assist API requires OAuth2 authentication")

        await self._ensure_session()
        headers = await self._get_auth_headers()
        project_id = await self._discover_project_id()

        contents = self._messages_to_contents(messages)

        request_data = {
            "model": model,
            "project": project_id,
            "request": {
                "contents": contents,
                "generationConfig": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "maxOutputTokens": kwargs.get("max_tokens", 8192),
                },
            },
        }

        if self.session is None:
            raise ValueError("Session not initialized")
        async with self.session.post(
            f"{self.code_assist_endpoint}/{self.code_assist_version}:streamGenerateContent",
            headers=headers,
            json=request_data,
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                if response.status == 429:
                    raise GeminiRateLimitError(f"Rate limit exceeded: {error_text}")
                raise ValueError(f"Code Assist API call failed: {response.status} - {error_text}")

            return await self._process_streaming_response(response)

    async def _process_streaming_response(self, response: aiohttp.ClientResponse) -> str:
        """Process streaming response from Code Assist API"""
        complete_content = ""

        async for line in response.content:
            if line:
                line_str = line.decode("utf-8").strip()
                if line_str and line_str != "[DONE]":
                    try:
                        if line_str.startswith("data: "):
                            line_str = line_str[6:]
                        data = json.loads(line_str)
                        content = self._extract_content_from_response(data)
                        if content:
                            complete_content += content
                    except json.JSONDecodeError:
                        continue

        return complete_content

    def _extract_content_from_response(self, data: Dict[str, Any]) -> str:
        """Extract text content from API response"""
        try:
            # Handle different response structures
            if isinstance(data, list):
                # Handle list response
                for item in data:
                    if isinstance(item, dict):
                        content = self._extract_content_from_response(item)
                        if content:
                            return content
                return ""

            candidates = data.get("candidates", [])
            if candidates and isinstance(candidates, list):
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts and isinstance(parts, list):
                    return parts[0].get("text", "")

            # Fallback for direct content
            if "content" in data:
                parts = data["content"].get("parts", [])
                if parts and isinstance(parts, list):
                    return parts[0].get("text", "")

            # Direct text field
            if "text" in data:
                return data["text"]

            return ""
        except (KeyError, TypeError, IndexError, AttributeError) as e:
            logger.debug(f"Error extracting content: {e}")
            return ""

    async def generate_completion(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> str:
        """
        Generate completion using appropriate API based on auth method
        """
        try:
            # Use Code Assist API for OAuth2, REST API for API key
            if self.auth_method == "oauth2" and self.oauth_manager:
                return await self._call_code_assist_api(model, messages, **kwargs)
            else:
                return await self._call_rest_api(model, messages, **kwargs)

        except GeminiRateLimitError:
            # Fallback to other API if rate limited
            logger.warning("Rate limit hit, attempting fallback...")
            try:
                if self.auth_method == "oauth2" and self.api_key:
                    # Fallback from OAuth2 to API key
                    return await self._call_rest_api(model, messages, **kwargs)
                elif self.auth_method == "api_key" and self.oauth_manager:
                    # Fallback from API key to OAuth2
                    return await self._call_code_assist_api(model, messages, **kwargs)
            except Exception as fallback_error:
                logger.error("Fallback also failed: %s", fallback_error)

            raise  # Re-raise original error

    # Abstract method implementations
    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for a specific model."""
        return self.SUPPORTED_MODELS.get(model_name, self.SUPPORTED_MODELS["gemini-2.0-flash"])

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using the model."""
        # Convert to message format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Run async method in sync context
        try:
            content = asyncio.run(
                self.generate_completion(
                    model_name, messages, temperature=temperature, max_tokens=max_output_tokens, **kwargs
                )
            )

            return ModelResponse(
                content=content,
                model_name=model_name,
                friendly_name=self.get_capabilities(model_name).friendly_name,
                provider=ProviderType.GOOGLE,
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},  # TODO: implement token counting
            )
        except Exception as e:
            raise ValueError(f"Gemini API call failed: {str(e)}") from e

    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens for the given text using the specified model's tokenizer."""
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.GOOGLE

    def validate_model_name(self, model_name: str) -> bool:
        """Validate if the model name is supported by this provider."""
        return model_name in self.SUPPORTED_MODELS

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode."""
        capabilities = self.get_capabilities(model_name)
        return capabilities.supports_extended_thinking

    def close(self):
        """Close the provider and cleanup resources"""
        if self.session and not self.session.closed:
            # Note: This would ideally be async, but base class expects sync
            # In practice, this should be called in an async context
            try:
                import asyncio
                if asyncio.iscoroutinefunction(self.session.close):
                    # If close is async, we need to handle it differently
                    pass
                else:
                    self.session.close()
            except Exception:
                pass  # Best effort cleanup
        self.session = None
