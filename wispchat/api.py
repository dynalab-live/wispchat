import json
import os
import time
import threading
import warnings

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

import openai
import openai.error

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .schema import CompletionOptions, OpenAIResponse, OpenAIResponseChunk, Function


SUPPORTED_API_TYPES = ["openai", "azure"]
API_TYPE_AZURE = "azure"
API_TYPE_OPENAI = "openai"


class WishChat:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        depolyment_id: str = "gpt-3.5-turbo",
        api_type: str = API_TYPE_OPENAI,
        api_key: Optional[str] = os.environ.get("OPENAI_API_KEY"),
        api_base: Optional[str] = os.environ.get("OPENAI_API_BASE"),
        api_version: Optional[str] = os.environ.get("OPENAI_API_VERSION"),
        system_prompt: str = "You are a helpful assistant.",
        enable_logging: bool = False,
    ):
        if api_type not in SUPPORTED_API_TYPES:
            raise ValueError(
                f"Unsupported API type. Supported types are: {', '.join(SUPPORTED_API_TYPES)}"
            )

        if api_type == API_TYPE_AZURE:
            self.api_type = api_type
            openai.api_type = api_type
            openai.api_key = api_key
            openai.api_base = api_base
            openai.api_version = api_version
        else:
            self.api_type = api_type
            openai.api_key = api_key
        self.model_name = model_name
        self.depolyment_id = depolyment_id
        self._system_prompt = system_prompt
        self.enable_logging = enable_logging
        self._local = threading.local()

        if enable_logging:
            logger.add("log_file.json", format="{message}", serialize=True)

    @contextmanager
    def override_system_prompt(self, new_prompt: str):
        """
        Temporarily overrides the system prompt for a specific block of code.
        This is useful for changing the behavior of the model within a specific context.

        Args:
            new_prompt (str): The new system prompt to use within the context.

        Usage:
            with api.override_system_prompt("You are a dog."):
                # Code here will use the new system prompt
        """
        original_prompt = getattr(self._local, "system_prompt", self._system_prompt)
        self._local.system_prompt = new_prompt
        yield
        self._local.system_prompt = original_prompt

    def system_prompt(self, prompt: str):
        """
        Decorator to temporarily override the system prompt for a specific function or method.
        This is useful for changing the behavior of the model for a specific function call.

        Args:
            prompt (str): The new system prompt to use within the decorated function.

        Usage:
            @api.system_prompt("You are a dog.")
            def some_function():
                # Code here will use the new system prompt
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.override_system_prompt(prompt):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def __call__(
        self,
        user_messages: List[str],
        options: Optional[Dict[str, Any]] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> OpenAIResponse:
        """
        Initiates a non-streaming interaction with the OpenAI Chat API.

        Args:
            user_messages (List[str]): The user messages to send to the API.
            options (Optional[Dict[str, Any]]): Optional parameters for the completion.
                Note: The 'stream' option is always set to False in this method and should not be provided.
                If 'stream' is provided in the options and set to True, it will be ignored.
            functions (Optional[List[Dict[str, Any]]]): Optional list of functions to call.
            system_prompt (Optional[str]): A system prompt to guide the model's behavior.

        Returns:
            OpenAIResponse: The response from the API.

        Example:
            response = api(user_messages, options={"max_tokens": 50})
            print(response.first)
        """
        if options and options.get("stream"):
            raise ValueError(
                "The 'stream' option is not allowed in this method. Use the 'stream' method for streaming."
            )

        # Ensure that the 'stream' option is set to False
        if options is None:
            options = {}
        options["stream"] = False

        return self.completion(
            user_messages, options, functions=functions, system_prompt=system_prompt
        )

    def stream(
        self,
        user_messages: List[str],
        options: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> Iterator[OpenAIResponseChunk]:
        """
        Initiates a streaming interaction with the OpenAI Chat API.

        Args:
            user_messages (List[str]): The user messages to send to the API.
            options (Optional[Dict[str, Any]]): Optional parameters for the completion.
                Note: The 'stream' option is always set to True in this method and cannot be overridden.
                If 'stream' is provided in the options and set to False, it will be ignored and a warning will be issued.
            system_prompt (Optional[str]): A system prompt to guide the model's behavior.

        Returns:
            Iterator[OpenAIResponseChunk]: An iterator over the response chunks from the API.

        Example:
            for chunk in api.stream(user_messages, options={"max_tokens": 50}):
                print(chunk.first)
        """
        if options and options.get("stream") is False:
            warnings.warn(
                "The 'stream' option is ignored in 'stream' method; it always operates in streaming mode."
            )

        # Ensure that the 'stream' option is set to True
        if options is None:
            options = {}
        options["stream"] = True

        return self.completion(user_messages, options, system_prompt)

    def completion(
        self,
        user_messages: List[str],
        options: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
    ) -> OpenAIResponse | Iterator[OpenAIResponseChunk]:
        """
        Constructs and sends a completion request to the OpenAI API.

        Args:
            user_messages (List[str]): List of user messages for the interaction.
            options (Optional[Dict[str, Any]]): Optional additional options for the completion. Default is None.
            system_prompt (Optional[str]): Optional system prompt to guide the model's behavior. Default is None.

        Returns:
            OpenAIResponse | Iterator[OpenAIResponseChunk]: An OpenAIResponse object or an iterator of OpenAIResponseChunk objects, depending on the call type (stream or not).

        This method takes user messages, optional completion options, and an optional system prompt.
        It verifies the provided options, constructs the messages (including a system message if a prompt is provided),
        and calls the OpenAI API with these messages.
        """
        verified_options = CompletionOptions(**options) if options else None
        verified_functions = (
            [Function(**function) for function in functions] if functions else None
        )

        system_prompt = system_prompt or getattr(
            self._local, "system_prompt", self._system_prompt
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for content in user_messages:
            messages.append({"role": "user", "content": content})

        response = self._call_openai_api(
            messages, vars(verified_options) if options else {}, functions
        )

        return response

    @retry(
        reraise=True,
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
    )
    def _call_openai_api(
        self,
        messages: List[Dict[str, str]],
        api_params: Dict[str, Any],
        functions: Optional[List[Dict[str, Any]]] = None,
    ) -> OpenAIResponse | Iterator[OpenAIResponseChunk]:
        if functions:
            function_call = "auto"
        else:
            function_call = "none"
        start_time = time.perf_counter()
        if functions:
            api_params["functions"] = functions
            api_params["function_call"] = function_call

        if self.api_type == API_TYPE_AZURE:
            response = openai.ChatCompletion.create(
                deployment_id=self.depolyment_id, messages=messages, **api_params
            )
        else:
            response = openai.ChatCompletion.create(
                model=self.model_name, messages=messages, **api_params
            )
        if api_params.get("stream"):
            return self._convert_to_response_chunks(response)
        else:
            response_data = OpenAIResponse(**response)
            self._log_response(start_time, messages, api_params, response_data)
            return response_data

    @staticmethod
    def _convert_to_response_chunks(openai_generator: Iterator):
        for chunk in openai_generator:
            chunk_object = OpenAIResponseChunk(**chunk)
            yield chunk_object

    def _log_response(self, start_time, messages, api_params, response_data):
        if not self.enable_logging:
            return

        end_time = time.perf_counter()
        response_time = end_time - start_time
        log_info = {
            "timestamp": start_time,
            "request": {
                "model_name": self.model_name,
                "messages": messages,
                "options": api_params,
            },
            "response": response_data.dict(),
            "analysis": {"response_time": response_time},
            "errors": None,
        }
        logger.info(log_info)
