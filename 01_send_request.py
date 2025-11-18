import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sending Requests
    This notebook provides a quick-start guide to use SGLang in chat completions after installation.

    - For Vision Language Models, see [OpenAI APIs - Vision](openai_api_vision.ipynb).
    - For Embedding Models, see [OpenAI APIs - Embedding](openai_api_embeddings.ipynb) and [Encode (embedding model)](native_api.html#Encode-(embedding-model)).
    - For Reward Models, see [Classify (reward model)](native_api.html#Classify-(reward-model)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Launch A Server
    """)
    return


@app.cell
def _():
    from sglang.test.doc_patch import launch_server_cmd
    from sglang.utils import wait_for_server, print_highlight, terminate_process

    # This is equivalent to running the following command in your terminal
    # python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0

    server_process, port = launch_server_cmd(
        """
    python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct \
     --host 0.0.0.0 --attention-backend triton --log-level warning
    """
    )

    wait_for_server(f"http://localhost:{port}")
    return port, print_highlight, server_process, terminate_process


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using cURL
    """)
    return


@app.cell
def _(port, print_highlight):
    import subprocess, json
    curl_command = f"""\ncurl -s http://localhost:{port}/v1/chat/completions   -H "Content-Type: application/json"   -d '{{"model": "qwen/qwen2.5-0.5b-instruct", "messages": [{{"role": "user", "content": "What is the capital of France?"}}]}}'\n"""
    _response = json.loads(subprocess.check_output(curl_command, shell=True))
    print_highlight(_response)
    return (json,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using Python Requests
    """)
    return


@app.cell
def _(port, print_highlight):
    import requests
    url = f'http://localhost:{port}/v1/chat/completions'
    _data = {'model': 'qwen/qwen2.5-0.5b-instruct', 'messages': [{'role': 'user', 'content': 'What is the capital of France?'}]}
    _response = requests.post(url, json=_data)
    print_highlight(_response.json())
    return (requests,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using OpenAI Python Client
    """)
    return


@app.cell
def _(port, print_highlight):
    import openai
    _client = openai.Client(base_url=f'http://127.0.0.1:{port}/v1', api_key='None')
    _response = _client.chat.completions.create(model='qwen/qwen2.5-0.5b-instruct', messages=[{'role': 'user', 'content': 'List 3 countries and their capitals.'}], temperature=0, max_tokens=64)
    print_highlight(_response)
    return (openai,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Streaming
    """)
    return


@app.cell
def _(openai, port):
    _client = openai.Client(base_url=f'http://127.0.0.1:{port}/v1', api_key='None')
    _response = _client.chat.completions.create(model='qwen/qwen2.5-0.5b-instruct', messages=[{'role': 'user', 'content': 'List 3 countries and their capitals.'}], temperature=0, max_tokens=64, stream=True)
    for _chunk in _response:
        if _chunk.choices[0].delta.content:
            print(_chunk.choices[0].delta.content, end='', flush=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using Native Generation APIs

    You can also use the native `/generate` endpoint with requests, which provides more flexibility. An API reference is available at [Sampling Parameters](sampling_params.md).
    """)
    return


@app.cell
def _(port, print_highlight, requests):
    _response = requests.post(f'http://localhost:{port}/generate', json={'text': 'The capital of France is', 'sampling_params': {'temperature': 0, 'max_new_tokens': 32}})
    print_highlight(_response.json())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Streaming
    """)
    return


@app.cell
def _(json, port, requests):
    _response = requests.post(f'http://localhost:{port}/generate', json={'text': 'The capital of France is', 'sampling_params': {'temperature': 0, 'max_new_tokens': 32}, 'stream': True}, stream=True)
    prev = 0
    for _chunk in _response.iter_lines(decode_unicode=False):
        _chunk = _chunk.decode('utf-8')
        if _chunk and _chunk.startswith('data:'):
            if _chunk == 'data: [DONE]':
                break
            _data = json.loads(_chunk[5:].strip('\n'))
            output = _data['text']
            print(output[prev:], end='', flush=True)
            prev = len(output)
    return


@app.cell
def _(server_process, terminate_process):
    terminate_process(server_process)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
