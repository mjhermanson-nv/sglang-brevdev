import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Structured Outputs For Reasoning Models

    When working with reasoning models that use special tokens like `<think>...</think>` to denote reasoning sections, you might want to allow free-form text within these sections while still enforcing grammar constraints on the rest of the output.

    SGLang provides a feature to disable grammar restrictions within reasoning sections. This is particularly useful for models that need to perform complex reasoning steps before providing a structured output.

    To enable this feature, use the `--reasoning-parser` flag which decide the think_end_token, such as `</think>`, when launching the server. You can also specify the reasoning parser using the `--reasoning-parser` flag.

    ## Supported Models

    Currently, SGLang supports the following reasoning models:
    - [DeepSeek R1 series](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d): The reasoning content is wrapped with `<think>` and `</think>` tags.
    - [QwQ](https://huggingface.co/Qwen/QwQ-32B): The reasoning content is wrapped with `<think>` and `</think>` tags.


    ## Usage

    ## OpenAI Compatible API
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Specify the `--grammar-backend`, `--reasoning-parser` option.
    """)
    return


@app.cell
def _():
    import openai
    import os

    from sglang.test.doc_patch import launch_server_cmd
    from sglang.utils import wait_for_server, print_highlight, terminate_process

    os.environ["TOKENIZERS_PARALLELISM"] = "false"


    server_process, port = launch_server_cmd(
        "python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --host 0.0.0.0 --reasoning-parser deepseek-r1 --attention-backend triton --log-level warning"
    )

    wait_for_server(f"http://localhost:{port}")
    client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")
    return client, port, print_highlight, server_process, terminate_process


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### JSON

    you can directly define a JSON schema or use [Pydantic](https://docs.pydantic.dev/latest/) to define and validate the response.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Using Pydantic**
    """)
    return


@app.cell
def _(client, print_highlight):
    from pydantic import BaseModel, Field

    class _CapitalInfo(BaseModel):
    # Define the schema using Pydantic
        name: str = Field(..., pattern='^\\w+$', description='Name of the capital city')
        population: int = Field(..., description='Population of the capital city')
    _response = client.chat.completions.create(model='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', messages=[{'role': 'assistant', 'content': 'Give me the information and population of the capital of France in the JSON format.'}], temperature=0, max_tokens=2048, response_format={'type': 'json_schema', 'json_schema': {'name': 'foo', 'schema': _CapitalInfo.model_json_schema()}})
    print_highlight(f'reasoing_content: {_response.choices[0].message.reasoning_content}\n\ncontent: {_response.choices[0].message.content}')  # convert the pydantic model to json schema
    return BaseModel, Field


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **JSON Schema Directly**
    """)
    return


@app.cell
def _(client, print_highlight):
    import json
    _json_schema = json.dumps({'type': 'object', 'properties': {'name': {'type': 'string', 'pattern': '^[\\w]+$'}, 'population': {'type': 'integer'}}, 'required': ['name', 'population']})
    _response = client.chat.completions.create(model='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', messages=[{'role': 'assistant', 'content': 'Give me the information and population of the capital of France in the JSON format.'}], temperature=0, max_tokens=2048, response_format={'type': 'json_schema', 'json_schema': {'name': 'foo', 'schema': json.loads(_json_schema)}})
    print_highlight(f'reasoing_content: {_response.choices[0].message.reasoning_content}\n\ncontent: {_response.choices[0].message.content}')
    return (json,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EBNF
    """)
    return


@app.cell
def _(client, print_highlight):
    ebnf_grammar = '\nroot ::= city | description\ncity ::= "London" | "Paris" | "Berlin" | "Rome"\ndescription ::= city " is " status\nstatus ::= "the capital of " country\ncountry ::= "England" | "France" | "Germany" | "Italy"\n'
    _response = client.chat.completions.create(model='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', messages=[{'role': 'system', 'content': 'You are a helpful geography bot.'}, {'role': 'assistant', 'content': 'Give me the information and population of the capital of France in the JSON format.'}], temperature=0, max_tokens=2048, extra_body={'ebnf': ebnf_grammar})
    print_highlight(f'reasoing_content: {_response.choices[0].message.reasoning_content}\n\ncontent: {_response.choices[0].message.content}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regular expression
    """)
    return


@app.cell
def _(client, print_highlight):
    _response = client.chat.completions.create(model='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', messages=[{'role': 'assistant', 'content': 'What is the capital of France?'}], temperature=0, max_tokens=2048, extra_body={'regex': '(Paris|London)'})
    print_highlight(f'reasoing_content: {_response.choices[0].message.reasoning_content}\n\ncontent: {_response.choices[0].message.content}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Structural Tag
    """)
    return


@app.cell
def _(client, print_highlight):
    tool_get_current_weather = {'type': 'function', 'function': {'name': 'get_current_weather', 'description': 'Get the current weather in a given location', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string', 'description': "The city to find the weather for, e.g. 'San Francisco'"}, 'state': {'type': 'string', 'description': "the two-letter abbreviation for the state that the city is in, e.g. 'CA' which would mean 'California'"}, 'unit': {'type': 'string', 'description': 'The unit to fetch the temperature in', 'enum': ['celsius', 'fahrenheit']}}, 'required': ['city', 'state', 'unit']}}}
    tool_get_current_date = {'type': 'function', 'function': {'name': 'get_current_date', 'description': 'Get the current date and time for a given timezone', 'parameters': {'type': 'object', 'properties': {'timezone': {'type': 'string', 'description': "The timezone to fetch the current date and time for, e.g. 'America/New_York'"}}, 'required': ['timezone']}}}
    schema_get_current_weather = tool_get_current_weather['function']['parameters']
    schema_get_current_date = tool_get_current_date['function']['parameters']

    def get_messages():
        return [{'role': 'system', 'content': f"""\n# Tool Instructions\n- Always execute python code in messages that you share.\n- When looking for real time information use relevant functions if available else fallback to brave_search\nYou have access to the following functions:\nUse the function 'get_current_weather' to: Get the current weather in a given location\n{tool_get_current_weather['function']}\nUse the function 'get_current_date' to: Get the current date and time for a given timezone\n{tool_get_current_date['function']}\nIf a you choose to call a function ONLY reply in the following format:\n<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}\nwhere\nstart_tag => `<function`\nparameters => a JSON dict with the function argument name as key and function argument value as value.\nend_tag => `</function>`\nHere is an example,\n<function=example_function_name>{{"example_name": "example_value"}}</function>\nReminder:\n- Function calls MUST follow the specified format\n- Required parameters MUST be specified\n- Only call one function at a time\n- Put the entire function call reply on one line\n- Always add your sources when using search results to answer the user query\nYou are a helpful assistant."""}, {'role': 'assistant', 'content': 'You are in New York. Please get the current date and time, and the weather.'}]
    messages = get_messages()
    _response = client.chat.completions.create(model='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', messages=messages, response_format={'type': 'structural_tag', 'max_new_tokens': 2048, 'structures': [{'begin': '<function=get_current_weather>', 'schema': schema_get_current_weather, 'end': '</function>'}, {'begin': '<function=get_current_date>', 'schema': schema_get_current_date, 'end': '</function>'}], 'triggers': ['<function=']})
    print_highlight(f'reasoing_content: {_response.choices[0].message.reasoning_content}\n\ncontent: {_response.choices[0].message.content}')
    return schema_get_current_date, schema_get_current_weather


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Native API and SGLang Runtime (SRT)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### JSON
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Using Pydantic**
    """)
    return


@app.cell
def _(BaseModel, Field, json, port, print_highlight):
    import requests
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')

    class _CapitalInfo(BaseModel):
        name: str = Field(..., pattern='^\\w+$', description='Name of the capital city')
        population: int = Field(..., description='Population of the capital city')
    messages_1 = [{'role': 'assistant', 'content': 'Give me the information and population of the capital of France in the JSON format.'}]
    _text = tokenizer.apply_chat_template(messages_1, tokenize=False, add_generation_prompt=True)
    _response = requests.post(f'http://localhost:{port}/generate', json={'text': _text, 'sampling_params': {'temperature': 0, 'max_new_tokens': 2048, 'json_schema': json.dumps(_CapitalInfo.model_json_schema())}})
    print(_response.json())
    reasoing_content = _response.json()['text'].split('</think>')[0]
    content = _response.json()['text'].split('</think>')[1]
    print_highlight(f'reasoing_content: {reasoing_content}\n\ncontent: {content}')
    return messages_1, requests, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **JSON Schema Directly**
    """)
    return


@app.cell
def _(json, messages_1, port, print_highlight, requests, tokenizer):
    _json_schema = json.dumps({'type': 'object', 'properties': {'name': {'type': 'string', 'pattern': '^[\\w]+$'}, 'population': {'type': 'integer'}}, 'required': ['name', 'population']})
    _text = tokenizer.apply_chat_template(messages_1, tokenize=False, add_generation_prompt=True)
    _response = requests.post(f'http://localhost:{port}/generate', json={'text': _text, 'sampling_params': {'temperature': 0, 'max_new_tokens': 2048, 'json_schema': _json_schema}})
    print_highlight(_response.json())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EBNF
    """)
    return


@app.cell
def _(port, requests):
    _response = requests.post(f'http://localhost:{port}/generate', json={'text': 'Give me the information of the capital of France.', 'sampling_params': {'max_new_tokens': 2048, 'temperature': 0, 'n': 3, 'ebnf': 'root ::= city | description\ncity ::= "London" | "Paris" | "Berlin" | "Rome"\ndescription ::= city " is " status\nstatus ::= "the capital of " country\ncountry ::= "England" | "France" | "Germany" | "Italy"'}, 'stream': False, 'return_logprob': False})
    print(_response.json())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regular expression
    """)
    return


@app.cell
def _(port, requests):
    _response = requests.post(f'http://localhost:{port}/generate', json={'text': 'Paris is the capital of', 'sampling_params': {'temperature': 0, 'max_new_tokens': 2048, 'regex': '(France|England)'}})
    print(_response.json())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Structural Tag
    """)
    return


@app.cell
def _(
    json,
    messages_1,
    port,
    print_highlight,
    requests,
    schema_get_current_date,
    schema_get_current_weather,
    tokenizer,
):
    _text = tokenizer.apply_chat_template(messages_1, tokenize=False, add_generation_prompt=True)
    payload = {'text': _text, 'sampling_params': {'max_new_tokens': 2048, 'structural_tag': json.dumps({'type': 'structural_tag', 'structures': [{'begin': '<function=get_current_weather>', 'schema': schema_get_current_weather, 'end': '</function>'}, {'begin': '<function=get_current_date>', 'schema': schema_get_current_date, 'end': '</function>'}], 'triggers': ['<function=']})}}
    _response = requests.post(f'http://localhost:{port}/generate', json=payload)
    print_highlight(_response.json())
    return


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
def _():
    import sglang as sgl

    llm = sgl.Engine(
        model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        reasoning_parser="deepseek-r1",
        grammar_backend="xgrammar",
    )
    return (llm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### JSON
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Using Pydantic**
    """)
    return


@app.cell
def _(BaseModel, Field, json, llm):
    _prompts = ['Give me the information of the capital of China in the JSON format.', 'Give me the information of the capital of France in the JSON format.', 'Give me the information of the capital of Ireland in the JSON format.']

    class _CapitalInfo(BaseModel):
        name: str = Field(..., pattern='^\\w+$', description='Name of the capital city')
        population: int = Field(..., description='Population of the capital city')
    _sampling_params = {'temperature': 0, 'top_p': 0.95, 'max_new_tokens': 2048, 'json_schema': json.dumps(_CapitalInfo.model_json_schema())}
    _outputs = llm.generate(_prompts, _sampling_params)
    for (_prompt, _output) in zip(_prompts, _outputs):
        print('===============================')
        print(f"Prompt: {_prompt}\nGenerated text: {_output['text']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **JSON Schema Directly**
    """)
    return


@app.cell
def _(json, llm):
    _prompts = ['Give me the information of the capital of China in the JSON format.', 'Give me the information of the capital of France in the JSON format.', 'Give me the information of the capital of Ireland in the JSON format.']
    _json_schema = json.dumps({'type': 'object', 'properties': {'name': {'type': 'string', 'pattern': '^[\\w]+$'}, 'population': {'type': 'integer'}}, 'required': ['name', 'population']})
    _sampling_params = {'temperature': 0, 'max_new_tokens': 2048, 'json_schema': _json_schema}
    _outputs = llm.generate(_prompts, _sampling_params)
    for (_prompt, _output) in zip(_prompts, _outputs):
        print('===============================')
        print(f"Prompt: {_prompt}\nGenerated text: {_output['text']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EBNF
    """)
    return


@app.cell
def _(llm):
    _prompts = ['Give me the information of the capital of France.', 'Give me the information of the capital of Germany.', 'Give me the information of the capital of Italy.']
    _sampling_params = {'temperature': 0.8, 'top_p': 0.95, 'ebnf': 'root ::= city | description\ncity ::= "London" | "Paris" | "Berlin" | "Rome"\ndescription ::= city " is " status\nstatus ::= "the capital of " country\ncountry ::= "England" | "France" | "Germany" | "Italy"'}
    _outputs = llm.generate(_prompts, _sampling_params)
    for (_prompt, _output) in zip(_prompts, _outputs):
        print('===============================')
        print(f"Prompt: {_prompt}\nGenerated text: {_output['text']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regular expression
    """)
    return


@app.cell
def _(llm):
    _prompts = ['Please provide information about London as a major global city:', 'Please provide information about Paris as a major global city:']
    _sampling_params = {'temperature': 0.8, 'top_p': 0.95, 'regex': '(France|England)'}
    _outputs = llm.generate(_prompts, _sampling_params)
    for (_prompt, _output) in zip(_prompts, _outputs):
        print('===============================')
        print(f"Prompt: {_prompt}\nGenerated text: {_output['text']}")
    return


@app.cell
def _(
    json,
    llm,
    messages_1,
    schema_get_current_date,
    schema_get_current_weather,
    tokenizer,
):
    _text = tokenizer.apply_chat_template(messages_1, tokenize=False, add_generation_prompt=True)
    _prompts = [_text]
    _sampling_params = {'temperature': 0.8, 'top_p': 0.95, 'max_new_tokens': 2048, 'structural_tag': json.dumps({'type': 'structural_tag', 'structures': [{'begin': '<function=get_current_weather>', 'schema': schema_get_current_weather, 'end': '</function>'}, {'begin': '<function=get_current_date>', 'schema': schema_get_current_date, 'end': '</function>'}], 'triggers': ['<function=']})}
    _outputs = llm.generate(_prompts, _sampling_params)
    for (_prompt, _output) in zip(_prompts, _outputs):
        print('===============================')
        print(f"Prompt: {_prompt}\nGenerated text: {_output['text']}")
    return


@app.cell
def _(llm):
    llm.shutdown()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
