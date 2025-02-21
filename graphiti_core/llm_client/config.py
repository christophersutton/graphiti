"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import logging

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0

logger = logging.getLogger(__name__)


class LLMConfig:
    """
    Configuration class for the Language Learning Model (LLM).

    This class encapsulates the necessary parameters to interact with an LLM API,
    such as OpenAI's GPT models. It stores the API key, model name, and base URL
    for making requests to the LLM service.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = 'http://localhost:4000',
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the LLMConfig with the provided parameters.

        Args:
                api_key (str): The authentication key for accessing the LLM API.
                                                This is required for making authorized requests.

                model (str, optional): The specific LLM model to use for generating responses.
                                                                Defaults to "gpt-4o-mini", which appears to be a custom model name.
                                                                Common values might include "gpt-3.5-turbo" or "gpt-4".

                base_url (str, optional): The base URL of the LLM API service.
                                                                        If not provided, will check for LLM_BASE_URL environment variable.
                                                                        If neither is provided, defaults to None which will use the client's default.
        """
        env_base_url = os.environ.get('LLM_BASE_URL')
        logger.warning(f"LLMConfig initialization:")  # Changed to warning for visibility
        logger.warning(f"  - Environment variables:")
        logger.warning(f"    - LLM_BASE_URL: {env_base_url}")
        logger.warning(f"  - Constructor args:")
        logger.warning(f"    - base_url: {base_url}")
        
        # Be explicit about which value we're using
        if base_url is not None:
            self.base_url = base_url
            logger.warning(f"Using provided base_url: {self.base_url}")
        elif env_base_url is not None:
            self.base_url = env_base_url
            logger.warning(f"Using environment variable LLM_BASE_URL: {self.base_url}")
        else:
            self.base_url = None
            logger.warning("No base_url provided, will use client default")
        
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
