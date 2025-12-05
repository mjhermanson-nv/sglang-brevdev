import marimo

__generated_with = "0.18.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Structured Outputs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can specify a JSON schema, [regular expression](https://en.wikipedia.org/wiki/Regular_expression) or [EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) to constrain the model output. The model output will be guaranteed to follow the given constraints. Only one constraint parameter (`json_schema`, `regex`, or `ebnf`) can be specified for a request.

    SGLang supports three grammar backends:

    - [XGrammar](https://github.com/mlc-ai/xgrammar)(default): Supports JSON schema, regular expression, and EBNF constraints.
    - [Outlines](https://github.com/dottxt-ai/outlines): Supports JSON schema and regular expression constraints.
    - [Llguidance](https://github.com/guidance-ai/llguidance): Supports JSON schema, regular expression, and EBNF constraints.

    We suggest using XGrammar for its better performance and utility. XGrammar currently uses the [GGML BNF format](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md). For more details, see [XGrammar technical overview](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar).

    To use Outlines, simply add `--grammar-backend outlines` when launching the server.
    To use llguidance, add `--grammar-backend llguidance`  when launching the server.
    If no backend is specified, XGrammar will be used as the default.

    For better output quality, **It's advisable to explicitly include instructions in the prompt to guide the model to generate the desired format.** For example, you can specify, 'Please generate the output in the following JSON format: ...'.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## OpenAI Compatible API
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Hugging Face Authentication

    Some models require a Hugging Face token for access. Please enter your token below.
    """)
    return


@app.cell
def _(mo):
    import os
    import getpass

    # Prompt for Hugging Face token via stdin
    # Check if token is already set in environment
    existing_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

    if existing_token:
        mo.md(f"✅ Hugging Face token found in environment (length: {len(existing_token)})")
        mo.md("**To change it, run the next cell.**")
        hf_token = existing_token
    else:
        mo.md("**Please enter your Hugging Face token:**")
        mo.md("(Token format: starts with 'hf_' and is typically 40-50 characters)")
        # Use getpass for secure input (hides the token as you type)
        try:
            hf_token = getpass.getpass("Hugging Face Token: ")
            if hf_token and hf_token.strip():
                hf_token = hf_token.strip()
                # Validate token
                if len(hf_token) > 100:
                    mo.md(f"⚠️ **Warning:** Token appears too long ({len(hf_token)} chars). Please verify.")
                else:
                    try:
                        hf_token.encode('ascii')
                        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
                        os.environ["HF_TOKEN"] = hf_token
                        mo.md(f"✅ **Token saved!** (length: {len(hf_token)})")
                    except UnicodeEncodeError:
                        mo.md("⚠️ **Error:** Token contains invalid characters.")
                        hf_token = None
            else:
                mo.md("⚠️ No token entered. Model access may fail if it's gated.")
                hf_token = None
        except Exception as e:
            mo.md(f"⚠️ **Error reading token:** {e}")
            hf_token = None

    return hf_token, os


@app.cell
def _(hf_token, os):
    import openai
    import subprocess

    from sglang.test.doc_patch import launch_server_cmd
    from sglang.utils import wait_for_server, print_highlight, terminate_process

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Use token from stdin prompt or environment
    token_value = hf_token or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

    if token_value and token_value.strip():
        token_value = token_value.strip()
        # Ensure environment variables are set (must be set before subprocess is created)
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token_value
        os.environ["HF_TOKEN"] = token_value
        print(f"✅ Using Hugging Face token (length: {len(token_value)})")

        # Also try to authenticate with huggingface-cli if available
        # This ensures the token is cached for the subprocess
        try:
            result = subprocess.run(
                ["huggingface-cli", "login", "--token", token_value, "--add-to-git-credential"],
                input="",
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
                env=os.environ.copy()  # Ensure environment is passed
            )
            if result.returncode == 0:
                print("✅ Authenticated with huggingface-cli")
        except FileNotFoundError:
            print("ℹ️  huggingface-cli not found, using environment variables only")
        except subprocess.TimeoutExpired:
            print("⚠️  huggingface-cli login timed out, using environment variables")
        except Exception as e:
            print(f"⚠️  Could not login via huggingface-cli: {e}")
            print("   Using environment variables instead.")
    else:
        print("⚠️  No Hugging Face token provided. Model access may fail if it's gated.")

    # Use a smaller model to avoid GPU memory issues
    # Meta-Llama-3.1-8B-Instruct requires ~16GB GPU memory
    # If you need the 8B model, make sure to free GPU memory first
    model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # Check GPU memory and warn if needed
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            free_memory_gb = int(result.stdout.strip().split('\n')[0]) / 1024
            if free_memory_gb < 20:
                print(f"⚠️  WARNING: Only {free_memory_gb:.1f} GB GPU memory free.")
                print("   The 8B model requires ~16GB. Consider:")
                print("   1. Killing old SGLang processes: pkill -f 'sglang::scheduler'")
                print("   2. Using a smaller model")
    except Exception:
        pass  # Ignore if nvidia-smi fails

    # Use triton backend to avoid nvcc compilation issues
    # FlashInfer requires nvcc (CUDA compiler) which may not be available
    # Also disable CUDA graph to avoid FlashInfer compilation during graph capture
    server_process, port = launch_server_cmd(
        f"python3 -m sglang.launch_server --model-path {model_path} --host 0.0.0.0 --attention-backend triton --sampling-backend pytorch --disable-cuda-graph --log-level warning"
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
    _response = client.chat.completions.create(model='meta-llama/Meta-Llama-3.1-8B-Instruct', messages=[{'role': 'user', 'content': 'Please generate the information of the capital of France in the JSON format.'}], temperature=0, max_tokens=128, response_format={'type': 'json_schema', 'json_schema': {'name': 'foo', 'schema': _CapitalInfo.model_json_schema()}})
    response_content = _response.choices[0].message.content
    _capital_info = _CapitalInfo.model_validate_json(response_content)
    # validate the JSON response by the pydantic model
    print_highlight(f'Validated response: {_capital_info.model_dump_json()}')  # convert the pydantic model to json schema
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
    _response = client.chat.completions.create(model='meta-llama/Meta-Llama-3.1-8B-Instruct', messages=[{'role': 'user', 'content': 'Give me the information of the capital of France in the JSON format.'}], temperature=0, max_tokens=128, response_format={'type': 'json_schema', 'json_schema': {'name': 'foo', 'schema': json.loads(_json_schema)}})
    print_highlight(_response.choices[0].message.content)
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
    _response = client.chat.completions.create(model='meta-llama/Meta-Llama-3.1-8B-Instruct', messages=[{'role': 'system', 'content': 'You are a helpful geography bot.'}, {'role': 'user', 'content': 'Give me the information of the capital of France.'}], temperature=0, max_tokens=32, extra_body={'ebnf': ebnf_grammar})
    print_highlight(_response.choices[0].message.content)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regular expression
    """)
    return


@app.cell
def _(client, print_highlight):
    _response = client.chat.completions.create(model='meta-llama/Meta-Llama-3.1-8B-Instruct', messages=[{'role': 'user', 'content': 'What is the capital of France?'}], temperature=0, max_tokens=128, extra_body={'regex': '(Paris|London)'})
    print_highlight(_response.choices[0].message.content)
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
        return [{'role': 'system', 'content': f"""\n# Tool Instructions\n- Always execute python code in messages that you share.\n- When looking for real time information use relevant functions if available else fallback to brave_search\nYou have access to the following functions:\nUse the function 'get_current_weather' to: Get the current weather in a given location\n{tool_get_current_weather['function']}\nUse the function 'get_current_date' to: Get the current date and time for a given timezone\n{tool_get_current_date['function']}\nIf a you choose to call a function ONLY reply in the following format:\n<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}\nwhere\nstart_tag => `<function`\nparameters => a JSON dict with the function argument name as key and function argument value as value.\nend_tag => `</function>`\nHere is an example,\n<function=example_function_name>{{"example_name": "example_value"}}</function>\nReminder:\n- Function calls MUST follow the specified format\n- Required parameters MUST be specified\n- Only call one function at a time\n- Put the entire function call reply on one line\n- Always add your sources when using search results to answer the user query\nYou are a helpful assistant."""}, {'role': 'user', 'content': 'You are in New York. Please get the current date and time, and the weather.'}]
    messages = get_messages()
    _response = client.chat.completions.create(model='meta-llama/Meta-Llama-3.1-8B-Instruct', messages=messages, response_format={'type': 'structural_tag', 'structures': [{'begin': '<function=get_current_weather>', 'schema': schema_get_current_weather, 'end': '</function>'}, {'begin': '<function=get_current_date>', 'schema': schema_get_current_date, 'end': '</function>'}], 'triggers': ['<function=']})
    print_highlight(_response.choices[0].message.content)
    return messages, schema_get_current_date, schema_get_current_weather


@app.cell
def _(
    client,
    messages,
    print_highlight,
    schema_get_current_date,
    schema_get_current_weather,
):
    # Support for XGrammar latest structural tag format
    # https://xgrammar.mlc.ai/docs/tutorials/structural_tag.html
    _response = client.chat.completions.create(model='meta-llama/Meta-Llama-3.1-8B-Instruct', messages=messages, response_format={'type': 'structural_tag', 'format': {'type': 'triggered_tags', 'triggers': ['<function='], 'tags': [{'begin': '<function=get_current_weather>', 'content': {'type': 'json_schema', 'json_schema': schema_get_current_weather}, 'end': '</function>'}, {'begin': '<function=get_current_date>', 'content': {'type': 'json_schema', 'json_schema': schema_get_current_date}, 'end': '</function>'}], 'at_least_one': False, 'stop_after_first': False}})
    print_highlight(_response.choices[0].message.content)
    return


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
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')

    class _CapitalInfo(BaseModel):
        name: str = Field(..., pattern='^\\w+$', description='Name of the capital city')
        population: int = Field(..., description='Population of the capital city')
    messages_1 = [{'role': 'user', 'content': 'Here is the information of the capital of France in the JSON format.\n'}]
    text = tokenizer.apply_chat_template(messages_1, tokenize=False, add_generation_prompt=True)
    _response = requests.post(f'http://localhost:{port}/generate', json={'text': text, 'sampling_params': {'temperature': 0, 'max_new_tokens': 64, 'json_schema': json.dumps(_CapitalInfo.model_json_schema())}})
    print_highlight(_response.json())
    response_data = json.loads(_response.json()['text'])
    _capital_info = _CapitalInfo.model_validate(response_data)
    print_highlight(f'Validated response: {_capital_info.model_dump_json()}')
    return AutoTokenizer, requests, text, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **JSON Schema Directly**
    """)
    return


@app.cell
def _(json, port, print_highlight, requests, text):
    _json_schema = json.dumps({'type': 'object', 'properties': {'name': {'type': 'string', 'pattern': '^[\\w]+$'}, 'population': {'type': 'integer'}}, 'required': ['name', 'population']})
    _response = requests.post(f'http://localhost:{port}/generate', json={'text': text, 'sampling_params': {'temperature': 0, 'max_new_tokens': 64, 'json_schema': _json_schema}})
    # JSON
    print_highlight(_response.json())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EBNF
    """)
    return


@app.cell
def _(port, print_highlight, requests, tokenizer):
    messages_2 = [{'role': 'user', 'content': 'Give me the information of the capital of France.'}]
    text_1 = tokenizer.apply_chat_template(messages_2, tokenize=False, add_generation_prompt=True)
    _response = requests.post(f'http://localhost:{port}/generate', json={'text': text_1, 'sampling_params': {'max_new_tokens': 128, 'temperature': 0, 'n': 3, 'ebnf': 'root ::= city | description\ncity ::= "London" | "Paris" | "Berlin" | "Rome"\ndescription ::= city " is " status\nstatus ::= "the capital of " country\ncountry ::= "England" | "France" | "Germany" | "Italy"'}, 'stream': False, 'return_logprob': False})
    print_highlight(_response.json())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regular expression
    """)
    return


@app.cell
def _(port, print_highlight, requests, tokenizer):
    messages_3 = [{'role': 'user', 'content': 'Paris is the capital of'}]
    text_2 = tokenizer.apply_chat_template(messages_3, tokenize=False, add_generation_prompt=True)
    _response = requests.post(f'http://localhost:{port}/generate', json={'text': text_2, 'sampling_params': {'temperature': 0, 'max_new_tokens': 64, 'regex': '(France|England)'}})
    print_highlight(_response.json())
    return (messages_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Structural Tag
    """)
    return


@app.cell
def _(
    AutoTokenizer,
    json,
    messages_3,
    port,
    print_highlight,
    requests,
    schema_get_current_date,
    schema_get_current_weather,
):
    tokenizer_1 = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
    text_3 = tokenizer_1.apply_chat_template(messages_3, tokenize=False, add_generation_prompt=True)
    _payload = {'text': text_3, 'sampling_params': {'structural_tag': json.dumps({'type': 'structural_tag', 'structures': [{'begin': '<function=get_current_weather>', 'schema': schema_get_current_weather, 'end': '</function>'}, {'begin': '<function=get_current_date>', 'schema': schema_get_current_date, 'end': '</function>'}], 'triggers': ['<function=']})}}
    _response = requests.post(f'http://localhost:{port}/generate', json=_payload)
    print_highlight(_response.json())
    return text_3, tokenizer_1


@app.cell
def _(
    json,
    port,
    print_highlight,
    requests,
    schema_get_current_date,
    schema_get_current_weather,
    text_3,
):
    _payload = {'text': text_3, 'sampling_params': {'structural_tag': json.dumps({'type': 'structural_tag', 'format': {'type': 'triggered_tags', 'triggers': ['<function='], 'tags': [{'begin': '<function=get_current_weather>', 'content': {'type': 'json_schema', 'json_schema': schema_get_current_weather}, 'end': '</function>'}, {'begin': '<function=get_current_date>', 'content': {'type': 'json_schema', 'json_schema': schema_get_current_date}, 'end': '</function>'}], 'at_least_one': False, 'stop_after_first': False}})}}
    _response = requests.post(f'http://localhost:{port}/generate', json=_payload)
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
    import nest_asyncio

    nest_asyncio.apply()

    # Initialize engine with triton backend to avoid nvcc compilation issues
    # FlashInfer requires nvcc (CUDA compiler) which may not be available
    # Also disable CUDA graph to avoid FlashInfer compilation during graph capture
    llm = sgl.Engine(
        model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
        grammar_backend="xgrammar",
        attention_backend="triton",
        disable_cuda_graph=True
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
def _(BaseModel, Field, json, llm, print_highlight):
    prompts = ['Give me the information of the capital of China in the JSON format.', 'Give me the information of the capital of France in the JSON format.', 'Give me the information of the capital of Ireland in the JSON format.']

    class _CapitalInfo(BaseModel):
        name: str = Field(..., pattern='^\\w+$', description='Name of the capital city')
        population: int = Field(..., description='Population of the capital city')
    _sampling_params = {'temperature': 0.1, 'top_p': 0.95, 'json_schema': json.dumps(_CapitalInfo.model_json_schema())}
    _outputs = llm.generate(prompts, _sampling_params)
    for (_prompt, _output) in zip(prompts, _outputs):
        print_highlight('===============================')
        print_highlight(f'Prompt: {_prompt}')
        _capital_info = _CapitalInfo.model_validate_json(_output['text'])
        print_highlight(f'Validated output: {_capital_info.model_dump_json()}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **JSON Schema Directly**
    """)
    return


@app.cell
def _(json, llm, print_highlight):
    prompts_1 = ['Give me the information of the capital of China in the JSON format.', 'Give me the information of the capital of France in the JSON format.', 'Give me the information of the capital of Ireland in the JSON format.']
    _json_schema = json.dumps({'type': 'object', 'properties': {'name': {'type': 'string', 'pattern': '^[\\w]+$'}, 'population': {'type': 'integer'}}, 'required': ['name', 'population']})
    _sampling_params = {'temperature': 0.1, 'top_p': 0.95, 'json_schema': _json_schema}
    _outputs = llm.generate(prompts_1, _sampling_params)
    for (_prompt, _output) in zip(prompts_1, _outputs):
        print_highlight('===============================')
        print_highlight(f"Prompt: {_prompt}\nGenerated text: {_output['text']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EBNF
    """)
    return


@app.cell
def _(llm, print_highlight):
    prompts_2 = ['Give me the information of the capital of France.', 'Give me the information of the capital of Germany.', 'Give me the information of the capital of Italy.']
    _sampling_params = {'temperature': 0.8, 'top_p': 0.95, 'ebnf': 'root ::= city | description\ncity ::= "London" | "Paris" | "Berlin" | "Rome"\ndescription ::= city " is " status\nstatus ::= "the capital of " country\ncountry ::= "England" | "France" | "Germany" | "Italy"'}
    _outputs = llm.generate(prompts_2, _sampling_params)
    for (_prompt, _output) in zip(prompts_2, _outputs):
        print_highlight('===============================')
        print_highlight(f"Prompt: {_prompt}\nGenerated text: {_output['text']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regular expression
    """)
    return


@app.cell
def _(llm, print_highlight):
    prompts_3 = ['Please provide information about London as a major global city:', 'Please provide information about Paris as a major global city:']
    _sampling_params = {'temperature': 0.8, 'top_p': 0.95, 'regex': '(France|England)'}
    _outputs = llm.generate(prompts_3, _sampling_params)
    for (_prompt, _output) in zip(prompts_3, _outputs):
        print_highlight('===============================')
        print_highlight(f"Prompt: {_prompt}\nGenerated text: {_output['text']}")
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
    llm,
    messages_3,
    print_highlight,
    schema_get_current_date,
    schema_get_current_weather,
    tokenizer_1,
):
    text_4 = tokenizer_1.apply_chat_template(messages_3, tokenize=False, add_generation_prompt=True)
    prompts_4 = [text_4]
    _sampling_params = {'temperature': 0.8, 'top_p': 0.95, 'structural_tag': json.dumps({'type': 'structural_tag', 'structures': [{'begin': '<function=get_current_weather>', 'schema': schema_get_current_weather, 'end': '</function>'}, {'begin': '<function=get_current_date>', 'schema': schema_get_current_date, 'end': '</function>'}], 'triggers': ['<function=']})}
    _outputs = llm.generate(prompts_4, _sampling_params)
    for (_prompt, _output) in zip(prompts_4, _outputs):
        print_highlight('===============================')
        print_highlight(f"Prompt: {_prompt}\nGenerated text: {_output['text']}")
    return (prompts_4,)


@app.cell
def _(
    json,
    llm,
    print_highlight,
    prompts_4,
    schema_get_current_date,
    schema_get_current_weather,
):
    _sampling_params = {'temperature': 0.8, 'top_p': 0.95, 'structural_tag': json.dumps({'type': 'structural_tag', 'format': {'type': 'triggered_tags', 'triggers': ['<function='], 'tags': [{'begin': '<function=get_current_weather>', 'content': {'type': 'json_schema', 'json_schema': schema_get_current_weather}, 'end': '</function>'}, {'begin': '<function=get_current_date>', 'content': {'type': 'json_schema', 'json_schema': schema_get_current_date}, 'end': '</function>'}], 'at_least_one': False, 'stop_after_first': False}})}
    _outputs = llm.generate(prompts_4, _sampling_params)
    for (_prompt, _output) in zip(prompts_4, _outputs):
        print_highlight('===============================')
        print_highlight(f"Prompt: {_prompt}\nGenerated text: {_output['text']}")
    return


@app.cell
def _(llm):
    llm.shutdown()
    return


if __name__ == "__main__":
    app.run()
