from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CompletionOptions(BaseModel):
    """
    A class that defines completion options for the OpenAI Chat API
    """

    temperature: float = 0.0
    top_p: float = 1.0
    n: int = Field(default=1, ge=1)
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = Field(default_factory=dict)


from typing import Any, Union


# Define JSON Schema Object Model
class JSONSchema(BaseModel):
    type: str
    properties: Optional[Dict[str, Any]]
    items: Optional[Any]
    additionalProperties: Optional[Union[bool, Dict[str, Any]]]
    required: Optional[List[str]]
    enum: Optional[List[Any]]
    default: Optional[Any]
    description: Optional[str]
    format: Optional[str]


class Function(BaseModel):
    name: str = Field(..., max_length=64, regex=r"^[a-zA-Z0-9_-]+$")
    description: Optional[str] = None
    parameters: JSONSchema


class FunctionCall(BaseModel):
    name: str
    arguments: str


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None


class ChunkChoice(BaseModel):
    index: int
    delta: ChunkDelta
    finish_reason: Optional[str] = None


class OpenAIResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

    @property
    def contents(self):
        return [choice.message.content for choice in self.choices]

    @property
    def first(self):
        # Note: n >= 1, choices >= 1
        return self.choices[0].message.content

    @property
    def first_choice(self):
        return self.choices[0]


class OpenAIResponseChunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChunkChoice]

    @property
    def first(self):
        return self.choices[0].delta.content
