# WispChat
WispChat is a Pythonic client library for the OpenAI GPT API. Without introducing any extra abstractions, it maintains complete consistency with the OpenAI API's parameters and response structure. If you're already familiar with the GPT API, you'll find that WispChat seamlessly aligns with your existing knowledge (Python language structure, OpenAI API), with no unexpected deviations.

## Installation
`pip install wispchat`

## Features
- Pythonic Interface: Enables interactions with the OpenAI API in a Python-friendly way.
- Complete Consistency: Parameters and response structure are fully consistent with the OpenAI API.
- No Extra Abstractions: Does not build any abstractions beyond existing knowledge.
- Retry Mechanism: WispChat uses an adaptive retry strategy to automatically attempt to resend requests in case of timeouts, API errors, connection errors, etc.
- Logging: Configure the `enable_logging` option to enable detailed logging for debugging and monitoring.

## Usage
### Initialization
```python
from wispchat import ChatAPI

api = ChatAPI(model_name="gpt-3.5-turbo", api_key="YOUR_API_KEY")
```

### Non-Streaming Interaction
You can interact with the OpenAI API in a non-streaming manner as follows:
```python
response = api(["Hello, how are you?"], options={"max_tokens": 50})
print(response.first)
```

### Streaming Interaction
To interact in a streaming manner, you can use the stream method:
```python
for chunk in api.stream(["Hello, how are you?"], options={"max_tokens": 50}):
    print(chunk.first)
```

### Temporarily Override System Prompts
WispChat offers four different scope methods to temporarily change system prompts.

`Global Scope`: Set global system prompts when initializing the ChatAPI object.

`Context Scope`: Use the `override_system_tip` context manager to override system prompts within a specific code block.

```python
with api.override_system_tip("You are a dog."):
    response = api(["Woof!"])
```
`Decorator Scope`: Use the `with_system_tip` decorator to override system prompts within a specific function or method.

```python
@api.with_system_tip("You are a dog.")
def some_function():
    response = api(["Woof!"])
```
`Function Scope`: Override system prompts within a specific call using parameters.

```python
response = api(["Hello, how are you?"], system_tip="You are a dog.")
```
These methods allow you to flexibly manage system prompts and ensure that the correct prompts guide the model's behavior in different contexts.

### Return Results
WispChat returns an OpenAIResponse object after interacting with the OpenAI API. Here are some convenient methods provided by WispChat:

#### Methods
`contents`: Returns a list containing all choices' contents when request parameter n > 1.

```python
print(response.contents)  # Prints all choices' contents when n > 1
```

`first`: Returns the content of the first choice in the response choices list.
```python
print(response.first)  # Prints the content of the first choice
```

`first_choice`: Returns the first choice in the response choices list.
```python
print(response.first_choice)  # Prints the first choice
```

#### Examples
```python
response = api(["Hello, how are you?"], options={"max_tokens": 50, "n": 2})
print(response.first)      # Prints the content of the first choice
print(response.contents)   # Prints all choices' contents when n > 1
```
These methods allow you to easily access and manipulate the response contents with the OpenAI chat model.

## API Description
ChatAPI Class
```python
__init__(model_name, api_key, system_tip, enable_logging): # Initializes the object.
```
```python
override_system_tip(new_tip): # Context manager for temporarily overriding system prompts.
```
```python
with_system_tip(tip): # Decorator for temporarily overriding system prompts for specific functions.
```
```python
__call__(user_messages, options, system_tip): # Non-streaming call method.
```
```python
stream(user_messages, options, system_tip): # Streaming call method.
```

## Planned Features
WispChat is under active development and plans to introduce the following features in future releases:

### Multi-turn Dialogue Support (Chat Functionality)
Description: Allows users to have multi-turn dialogues with the OpenAI GPT model, similar to the interaction with ChatGPT.
Status: Planned
### Function Call Support
Description: Adds support for Function Calls, enabling users to interact with the model more flexibly and call specific functions.
Status: Planned

## How to Contribute
We welcome anyone to contribute to WispChat's development. Here are the steps to get started:

1. Clone the repository: git clone https://github.com/your_username/WispChat.git
2. Create a branch: git checkout -b feature/your_feature_name
3. Make changes and commit: git commit -m "Add your_feature_name"
4. Push to GitHub: git push origin feature/your_feature_name
5. Create a Pull Request
Ensure you adhere to the project's coding style and testing requirements.
