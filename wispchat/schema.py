from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CompletionOptions(BaseModel):
    """
    A class that defines completion options for the OpenAI Chat API
    """

    temperature: float = 0.0
    top_p: float = 1.0
    n: int = 1
    stop: Optional[List[str]] = None
    max_tokens: int = 16
    stream: bool = False
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = Field(default_factory=dict)


class Message(BaseModel):
    role: str
    content: str


class FunctionCall(BaseModel):
    name: str
    arguments: str


class Choice(BaseModel):
    index: int
    message: Message
    function_call: Optional[FunctionCall] = None
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
        return self.choices[0].message.content if self.choices else None

    @property
    def first_choice(self):
        return self.choices[0] if self.choices else None


class OpenAIResponseChunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChunkChoice]

    @property
    def first(self):
        return self.choices[0].delta.content if self.choices else None
