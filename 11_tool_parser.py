import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tool Parser

    This guide demonstrates how to use SGLang’s [Function calling](https://platform.openai.com/docs/guides/function-calling) functionality.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Currently supported parsers:

    | Parser | Supported Models | Notes |
    |---|---|---|
    | `deepseekv3` | DeepSeek-v3 (e.g., `deepseek-ai/DeepSeek-V3-0324`) | Recommend adding `--chat-template ./examples/chat_template/tool_chat_template_deepseekv3.jinja` to launch command. |
    | `deepseekv31` | DeepSeek-V3.1 and DeepSeek-V3.2 (e.g. `deepseek-ai/DeepSeek-V3.1`, `deepseek-ai/DeepSeek-V3.2-Exp`) | Recommend adding `--chat-template ./examples/chat_template/tool_chat_template_deepseekv31.jinja` (Or ..deepseekv32.jinja for DeepSeek-V3.2) to launch command. |
    | `glm` | GLM series (e.g. `zai-org/GLM-4.6`) | |
    | `gpt-oss` | GPT-OSS (e.g., `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `lmsys/gpt-oss-120b-bf16`, `lmsys/gpt-oss-20b-bf16`) | The gpt-oss tool parser filters out analysis channel events and only preserves normal text. This can cause the content to be empty when explanations are in the analysis channel. To work around this, complete the tool round by returning tool results as `role="tool"` messages, which enables the model to generate the final content. |
    | `kimi_k2` | `moonshotai/Kimi-K2-Instruct` | |
    | `llama3` | Llama 3.1 / 3.2 / 3.3 (e.g. `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`) | |
    | `llama4` | Llama 4 (e.g. `meta-llama/Llama-4-Scout-17B-16E-Instruct`) | |
    | `mistral` | Mistral (e.g. `mistralai/Mistral-7B-Instruct-v0.3`, `mistralai/Mistral-Nemo-Instruct-2407`, `mistralai/Mistral-7B-v0.3`) | |
    | `pythonic` | Llama-3.2 / Llama-3.3 / Llama-4 | Model outputs function calls as Python code. Requires `--tool-call-parser pythonic` and is recommended to use with a specific chat template. |
    | `qwen` | Qwen series (e.g. `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3-VL-30B-A3B-Thinking`) except Qwen3-Coder| |
    | `qwen3_coder` | Qwen3-Coder (e.g. `Qwen/Qwen3-Coder-30B-A3B-Instruct`) | |
    | `step3` | Step-3 | |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## OpenAI Compatible API
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Launching the Server
    """)
    return


@app.cell
def _():
    import json
    from sglang.test.doc_patch import launch_server_cmd
    from sglang.utils import wait_for_server, print_highlight, terminate_process
    from openai import OpenAI

    server_process, port = launch_server_cmd(
        "python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --tool-call-parser qwen25 --host 0.0.0.0 --log-level warning"  # qwen25
    )
    wait_for_server(f"http://localhost:{port}")
    return (
        OpenAI,
        json,
        launch_server_cmd,
        port,
        print_highlight,
        server_process,
        terminate_process,
        wait_for_server,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that `--tool-call-parser` defines the parser used to interpret responses.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define Tools for Function Call
    Below is a Python snippet that shows how to define a tool as a dictionary. The dictionary includes a tool name, a description, and property defined Parameters.
    """)
    return


@app.cell
def _():
    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for, e.g. 'San Francisco'",
                        },
                        "state": {
                            "type": "string",
                            "description": "the two-letter abbreviation for the state that the city is"
                            " in, e.g. 'CA' which would mean 'California'",
                        },
                        "unit": {
                            "type": "string",
                            "description": "The unit to fetch the temperature in",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city", "state", "unit"],
                },
            },
        }
    ]
    return (tools,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define Messages
    """)
    return


@app.cell
def _():
    def get_messages():
        return [
            {
                "role": "user",
                "content": "What's the weather like in Boston today? Output a reasoning before act, then use the tools to help you.",
            }
        ]


    messages = get_messages()
    return get_messages, messages


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Initialize the Client
    """)
    return


@app.cell
def _(OpenAI, port):
    # Initialize OpenAI-like client
    client = OpenAI(api_key="None", base_url=f"http://0.0.0.0:{port}/v1")
    model_name = client.models.list().data[0].id
    return client, model_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###  Non-Streaming Request
    """)
    return


@app.cell
def _(client, messages, model_name, print_highlight, tools):
    # Non-streaming mode test
    response_non_stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        top_p=0.95,
        max_tokens=1024,
        stream=False,  # Non-streaming
        tools=tools,
    )
    print_highlight("Non-stream response:")
    print_highlight(response_non_stream)
    print_highlight("==== content ====")
    print_highlight(response_non_stream.choices[0].message.content)
    print_highlight("==== tool_calls ====")
    print_highlight(response_non_stream.choices[0].message.tool_calls)
    return (response_non_stream,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Handle Tools
    When the engine determines it should call a particular tool, it will return arguments or partial arguments through the response. You can parse these arguments and later invoke the tool accordingly.
    """)
    return


@app.cell
def _(print_highlight, response_non_stream):
    name_non_stream = response_non_stream.choices[0].message.tool_calls[0].function.name
    arguments_non_stream = (
        response_non_stream.choices[0].message.tool_calls[0].function.arguments
    )

    print_highlight(f"Final streamed function call name: {name_non_stream}")
    print_highlight(f"Final streamed function call arguments: {arguments_non_stream}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Streaming Request
    """)
    return


@app.cell
def _(client, messages, model_name, print_highlight, tools):
    print_highlight('Streaming response:')
    _response_stream = client.chat.completions.create(model=model_name, messages=messages, temperature=0, top_p=0.95, max_tokens=1024, stream=True, tools=tools)
    _texts = ''
    tool_calls = []
    _name = ''
    _arguments = ''
    for _chunk in _response_stream:
        if _chunk.choices[0].delta.content:
            _texts = _texts + _chunk.choices[0].delta.content
        if _chunk.choices[0].delta.tool_calls:
            tool_calls.append(_chunk.choices[0].delta.tool_calls[0])
    print_highlight('==== Text ====')
    print_highlight(_texts)
    print_highlight('==== Tool Call ====')
    for _tool_call in tool_calls:
        print_highlight(_tool_call)
    return (tool_calls,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Handle Tools
    When the engine determines it should call a particular tool, it will return arguments or partial arguments through the response. You can parse these arguments and later invoke the tool accordingly.
    """)
    return


@app.cell
def _(print_highlight, tool_calls):
    # Parse and combine function call arguments
    _arguments = []
    for _tool_call in tool_calls:
        if _tool_call.function.name:
            print_highlight(f'Streamed function call name: {_tool_call.function.name}')
        if _tool_call.function.arguments:
            _arguments.append(_tool_call.function.arguments)
    full_arguments = ''.join(_arguments)
    # Combine all fragments into a single JSON string
    print_highlight(f'streamed function call arguments: {full_arguments}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define a Tool Function
    """)
    return


@app.cell
def _():
    # This is a demonstration, define real function according to your usage.
    def get_current_weather(city: str, state: str, unit: "str"):
        return (
            f"The weather in {city}, {state} is 85 degrees {unit}. It is "
            "partly cloudly, with highs in the 90's."
        )


    available_tools = {"get_current_weather": get_current_weather}
    return (available_tools,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Execute the Tool
    """)
    return


@app.cell
def _(available_tools, json, messages, print_highlight, response_non_stream):
    messages.append(response_non_stream.choices[0].message)
    _tool_call = messages[-1].tool_calls[0]
    # Call the corresponding tool function
    tool_name = _tool_call.function.name
    tool_to_call = available_tools[tool_name]
    _result = tool_to_call(**json.loads(_tool_call.function.arguments))
    print_highlight(f'Function call result: {_result}')
    messages.append({'role': 'tool', 'tool_call_id': _tool_call.id, 'content': str(_result), 'name': tool_name})
    # messages.append({"role": "tool", "content": result, "name": tool_name})
    print_highlight(f'Updated message history: {messages}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Send Results Back to Model
    """)
    return


@app.cell
def _(client, messages, model_name, print_highlight, tools):
    final_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        top_p=0.95,
        stream=False,
        tools=tools,
    )
    print_highlight("Non-stream response:")
    print_highlight(final_response)

    print_highlight("==== Text ====")
    print_highlight(final_response.choices[0].message.content)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Native API and SGLang Runtime (SRT)
    """)
    return


@app.cell
def _(get_messages, port, print_highlight, tools):
    from transformers import AutoTokenizer
    import requests
    _tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
    messages_1 = get_messages()
    input = _tokenizer.apply_chat_template(messages_1, tokenize=False, add_generation_prompt=True, tools=tools)
    gen_url = f'http://localhost:{port}/generate'
    gen_data = {'text': input, 'sampling_params': {'skip_special_tokens': False, 'max_new_tokens': 1024, 'temperature': 0, 'top_p': 0.95}}
    gen_response = requests.post(gen_url, json=gen_data).json()['text']
    print_highlight('==== Response ====')
    print_highlight(gen_response)
    parse_url = f'http://localhost:{port}/parse_function_call'
    function_call_input = {'text': gen_response, 'tool_call_parser': 'qwen25', 'tools': tools}
    function_call_response = requests.post(parse_url, json=function_call_input)
    function_call_response_json = function_call_response.json()
    print_highlight('==== Text ====')
    print(function_call_response_json['normal_text'])
    print_highlight('==== Calls ====')
    print('function name: ', function_call_response_json['calls'][0]['name'])
    print('function arguments: ', function_call_response_json['calls'][0]['parameters'])
    return (messages_1,)


@app.cell
def _(server_process, terminate_process):
    terminate_process(server_process)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Offline Engine API
    """)
    return


@app.cell
def _(messages_1, print_highlight, tools):
    import sglang as sgl
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.managers.io_struct import Tool, Function
    llm = sgl.Engine(model_path='Qwen/Qwen2.5-7B-Instruct')
    _tokenizer = llm.tokenizer_manager.tokenizer
    input_ids = _tokenizer.apply_chat_template(messages_1, tokenize=True, add_generation_prompt=True, tools=tools)
    sampling_params = {'max_new_tokens': 1024, 'temperature': 0, 'top_p': 0.95, 'skip_special_tokens': False}
    _result = llm.generate(input_ids=input_ids, sampling_params=sampling_params)
    generated_text = _result['text']
    print_highlight('=== Offline Engine Output Text ===')
    print_highlight(generated_text)

    def convert_dict_to_tool(tool_dict: dict) -> Tool:
        function_dict = tool_dict.get('function', {})
        return Tool(type=tool_dict.get('type', 'function'), function=Function(name=function_dict.get('name'), description=function_dict.get('description'), parameters=function_dict.get('parameters')))
    tools_1 = [convert_dict_to_tool(raw_tool) for raw_tool in tools]
    parser = FunctionCallParser(tools=tools_1, tool_call_parser='qwen25')
    (normal_text, calls) = parser.parse_non_stream(generated_text)
    print_highlight('=== Parsing Result ===')
    print('Normal text portion:', normal_text)
    print_highlight('Function call portion:')
    for call in calls:
        print_highlight(f'  - tool name: {call.name}')
        print_highlight(f'    parameters: {call.parameters}')
    return (llm,)


@app.cell
def _(llm):
    llm.shutdown()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tool Choice Mode

    SGLang supports OpenAI's `tool_choice` parameter to control when and which tools the model should call. This feature is implemented using EBNF (Extended Backus-Naur Form) grammar to ensure reliable tool calling behavior.

    ### Supported Tool Choice Options

    - **`tool_choice="required"`**: Forces the model to call at least one tool
    - **`tool_choice={"type": "function", "function": {"name": "specific_function"}}`**: Forces the model to call a specific function

    ### Backend Compatibility

    Tool choice is fully supported with the **Xgrammar backend**, which is the default grammar backend (`--grammar-backend xgrammar`). However, it may not be fully supported with other backends such as `outlines`.

    ### Example: Required Tool Choice
    """)
    return


@app.cell
def _(OpenAI, launch_server_cmd, print_highlight, wait_for_server):
    (server_process_tool_choice, port_tool_choice) = launch_server_cmd('python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --tool-call-parser qwen25 --host 0.0.0.0  --log-level warning')
    wait_for_server(f'http://localhost:{port_tool_choice}')
    client_tool_choice = OpenAI(api_key='None', base_url=f'http://0.0.0.0:{port_tool_choice}/v1')
    model_name_tool_choice = client_tool_choice.models.list().data[0].id
    # Start a new server session for tool choice examples
    messages_required = [{'role': 'user', 'content': 'Hello, what is the capital of France?'}]
    tools_2 = [{'type': 'function', 'function': {'name': 'get_current_weather', 'description': 'Get the current weather in a given location', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string', 'description': "The city to find the weather for, e.g. 'San Francisco'"}, 'unit': {'type': 'string', 'description': 'The unit to fetch the temperature in', 'enum': ['celsius', 'fahrenheit']}}, 'required': ['city', 'unit']}}}]
    response_required = client_tool_choice.chat.completions.create(model=model_name_tool_choice, messages=messages_required, temperature=0, max_tokens=1024, tools=tools_2, tool_choice='required')
    print_highlight("Response with tool_choice='required':")
    print('Content:', response_required.choices[0].message.content)
    # Initialize client for tool choice examples
    # Example with tool_choice="required" - forces the model to call a tool
    # Define tools
    print('Tool calls:', response_required.choices[0].message.tool_calls)  # Force the model to call a tool
    return (
        client_tool_choice,
        model_name_tool_choice,
        server_process_tool_choice,
        tools_2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: Specific Function Choice
    """)
    return


@app.cell
def _(client_tool_choice, model_name_tool_choice, print_highlight, tools_2):
    messages_specific = [{'role': 'user', 'content': 'What are the most attactive places in France?'}]
    response_specific = client_tool_choice.chat.completions.create(model=model_name_tool_choice, messages=messages_specific, temperature=0, max_tokens=1024, tools=tools_2, tool_choice={'type': 'function', 'function': {'name': 'get_current_weather'}})
    print_highlight('Response with specific function choice:')
    print('Content:', response_specific.choices[0].message.content)
    print('Tool calls:', response_specific.choices[0].message.tool_calls)
    if response_specific.choices[0].message.tool_calls:
        _tool_call = response_specific.choices[0].message.tool_calls[0]
        print_highlight(f'Called function: {_tool_call.function.name}')
        print_highlight(f'Arguments: {_tool_call.function.arguments}')
    return


@app.cell
def _(server_process_tool_choice, terminate_process):
    terminate_process(server_process_tool_choice)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pythonic Tool Call Format (Llama-3.2 / Llama-3.3 / Llama-4)

    Some Llama models (such as Llama-3.2-1B, Llama-3.2-3B, Llama-3.3-70B, and Llama-4) support a "pythonic" tool call format, where the model outputs function calls as Python code, e.g.:

    ```python
    [get_current_weather(city="San Francisco", state="CA", unit="celsius")]
    ```

    - The output is a Python list of function calls, with arguments as Python literals (not JSON).
    - Multiple tool calls can be returned in the same list:
    ```python
    [get_current_weather(city="San Francisco", state="CA", unit="celsius"),
     get_current_weather(city="New York", state="NY", unit="fahrenheit")]
    ```

    For more information, refer to Meta’s documentation on  [Zero shot function calling](https://github.com/meta-llama/llama-models/blob/main/models/llama4/prompt_format.md#zero-shot-function-calling---system-message).

    Note that this feature is still under development on Blackwell.

    ### How to enable
    - Launch the server with `--tool-call-parser pythonic`
    - You may also specify --chat-template with the improved template for the model (e.g., `--chat-template=examples/chat_template/tool_chat_template_llama4_pythonic.jinja`).
    This is recommended because the model expects a special prompt format to reliably produce valid pythonic tool call outputs. The template ensures that the prompt structure (e.g., special tokens, message boundaries like `<|eom|>`, and function call delimiters) matches what the model was trained or fine-tuned on. If you do not use the correct chat template, tool calling may fail or produce inconsistent results.

    #### Forcing Pythonic Tool Call Output Without a Chat Template
    If you don't want to specify a chat template, you must give the model extremely explicit instructions in your messages to enforce pythonic output. For example, for `Llama-3.2-1B-Instruct`, you need:
    """)
    return


@app.cell
def _(launch_server_cmd, print_highlight, terminate_process, wait_for_server):
    import openai
    (server_process_1, port_1) = launch_server_cmd(' python3 -m sglang.launch_server --model-path meta-llama/Llama-3.2-1B-Instruct --tool-call-parser pythonic --tp 1  --log-level warning')
    wait_for_server(f'http://localhost:{port_1}')
    tools_3 = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Get the current weather for a given location.', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'The name of the city or location.'}}, 'required': ['location']}}}, {'type': 'function', 'function': {'name': 'get_tourist_attractions', 'description': 'Get a list of top tourist attractions for a given city.', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string', 'description': 'The name of the city to find attractions for.'}}, 'required': ['city']}}}]

    def get_messages_1():
        return [{'role': 'system', 'content': 'You are a travel assistant. When asked to call functions, ALWAYS respond ONLY with a python list of function calls, using this format: [func_name1(param1=value1, param2=value2), func_name2(param=value)]. Do NOT use JSON, do NOT use variables, do NOT use any other format. Here is an example:\n[get_weather(location="Paris"), get_tourist_attractions(city="Paris")]'}, {'role': 'user', 'content': "I'm planning a trip to Tokyo next week. What's the weather like and what are some top tourist attractions? Propose parallel tool calls at once, using the python list of function calls format as shown above."}]
    messages_2 = get_messages_1()
    client_1 = openai.Client(base_url=f'http://localhost:{port_1}/v1', api_key='xxxxxx')
    model_name_1 = client_1.models.list().data[0].id
    response_non_stream_1 = client_1.chat.completions.create(model=model_name_1, messages=messages_2, temperature=0, top_p=0.9, stream=False, tools=tools_3)
    print_highlight('Non-stream response:')
    print_highlight(response_non_stream_1)
    _response_stream = client_1.chat.completions.create(model=model_name_1, messages=messages_2, temperature=0, top_p=0.9, stream=True, tools=tools_3)
    _texts = ''
    tool_calls_1 = []
    _name = ''
    _arguments = ''
    for _chunk in _response_stream:
        if _chunk.choices[0].delta.content:
            _texts = _texts + _chunk.choices[0].delta.content
        if _chunk.choices[0].delta.tool_calls:
            tool_calls_1.append(_chunk.choices[0].delta.tool_calls[0])
    print_highlight('Streaming Response:')
    print_highlight('==== Text ====')
    print_highlight(_texts)
    print_highlight('==== Tool Call ====')
    for _tool_call in tool_calls_1:
        print_highlight(_tool_call)
    terminate_process(server_process_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Note:**
    > The model may still default to JSON if it was heavily finetuned on that format. Prompt engineering (including examples) is the only way to increase the chance of pythonic output if you are not using a chat template.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to support a new model?
    1. Update the TOOLS_TAG_LIST in sglang/srt/function_call_parser.py with the model’s tool tags. Currently supported tags include:
    ```
    	TOOLS_TAG_LIST = [
    	    “<|plugin|>“,
    	    “<function=“,
    	    “<tool_call>“,
    	    “<|python_tag|>“,
    	    “[TOOL_CALLS]”
    	]
    ```
    2. Create a new detector class in sglang/srt/function_call_parser.py that inherits from BaseFormatDetector. The detector should handle the model’s specific function call format. For example:
    ```
        class NewModelDetector(BaseFormatDetector):
    ```
    3. Add the new detector to the MultiFormatParser class that manages all the format detectors.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
