import pytest

from unittest.mock import patch, MagicMock

from loguru import logger as loguru_logger

from wispchat.api import WishChat
from wispchat.schema import OpenAIResponse, OpenAIResponseChunk


@pytest.fixture
def api():
    return WishChat(enable_logging=True)


def test_override_system_tip(api):
    with api.override_system_tip("New tip") as system_tip:
        assert api._local.system_tip == "New tip"
    assert getattr(api._local, "system_tip", api.system_tip) == api.system_tip


def test_with_system_tip_decorator(api):
    @api.with_system_tip("New tip")
    def some_func():
        return api._local.system_tip

    assert some_func() == "New tip"


def test_call_non_stream(api):
    mock_response = MagicMock()  # Assuming OpenAIResponse
    with patch.object(api, "completion", return_value=mock_response) as mock_method:
        response = api(["message"])
        assert response == mock_response


def test_call_stream(api):
    mock_response_chunk = MagicMock()  # Assuming OpenAIResponseChunk
    with patch.object(
        api, "stream", return_value=iter([mock_response_chunk])
    ) as mock_method:
        response = list(api.stream(["message"]))
        assert response[0] == mock_response_chunk


def test_completion(api):
    options = {"stream": True}
    mock_response = MagicMock()  # Assuming OpenAIResponse
    with patch.object(
        api, "_call_openai_api", return_value=mock_response
    ) as mock_method:
        response = api.completion(["message"], options)
        assert response == mock_response


def test_call_openai_api(api):
    messages = [{"role": "user", "content": "message"}]
    api_params = {"stream": False}
    mock_create_response = {
        "id": "some_id",
        "object": "chat.completion",
        "created": 123,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {"role": "system", "content": "content"},
                "finish_reason": "stop",
            }
        ],  # Added finish_reason
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    with patch(
        "openai.ChatCompletion.create", return_value=mock_create_response
    ) as mock_create:
        response = api._call_openai_api(messages, api_params)
        assert isinstance(response, OpenAIResponse)


def test_convert_to_response_chunks(api):
    chunks = iter(
        [
            {
                "id": "some_id",
                "object": "chat.completion",
                "created": 123,
                "model": "gpt-3.5-turbo",
                "choices": [{"index": 0, "delta": {}}],
            }
        ]
    )
    converted_chunks = list(api._convert_to_response_chunks(chunks))
    assert isinstance(converted_chunks[0], OpenAIResponseChunk)


def test_log_response(api):
    with patch.object(loguru_logger, "info") as mock_logger:
        api._log_response(0, [], {}, MagicMock())  # Assuming OpenAIResponse
        mock_logger.assert_called_once()
