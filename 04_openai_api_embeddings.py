import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OpenAI APIs - Embedding

    SGLang provides OpenAI-compatible APIs to enable a smooth transition from OpenAI services to self-hosted local models.
    A complete reference for the API is available in the [OpenAI API Reference](https://platform.openai.com/docs/guides/embeddings).

    This tutorial covers the embedding APIs for embedding models. For a list of the supported models see the [corresponding overview page](../supported_models/embedding_models.md)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Launch A Server

    Launch the server in your terminal and wait for it to initialize. Remember to add `--is-embedding` to the command.
    """)
    return


@app.cell
def _():
    from sglang.test.doc_patch import launch_server_cmd
    from sglang.utils import wait_for_server, print_highlight, terminate_process

    embedding_process, port = launch_server_cmd(
        """
    python3 -m sglang.launch_server --model-path Alibaba-NLP/gte-Qwen2-1.5B-instruct \
        --host 0.0.0.0 --is-embedding --attention-backend triton --log-level warning
    """
    )

    wait_for_server(f"http://localhost:{port}")
    return embedding_process, port, print_highlight, terminate_process


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using cURL
    """)
    return


@app.cell
def _(port, print_highlight):
    import subprocess, json
    text = 'Once upon a time'
    curl_text = f"""curl -s http://localhost:{port}/v1/embeddings   -H "Content-Type: application/json"   -d '{{"model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "input": "{text}"}}'"""
    result = subprocess.check_output(curl_text, shell=True)
    print(result)
    _text_embedding = json.loads(result)['data'][0]['embedding']
    print_highlight(f'Text embedding (first 10): {_text_embedding[:10]}')
    return json, subprocess


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using Python Requests
    """)
    return


@app.cell
def _(port, print_highlight):
    import requests
    text_1 = 'Once upon a time'
    _response = requests.post(f'http://localhost:{port}/v1/embeddings', json={'model': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct', 'input': text_1})
    _text_embedding = _response.json()['data'][0]['embedding']
    print_highlight(f'Text embedding (first 10): {_text_embedding[:10]}')
    return (text_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using OpenAI Python Client
    """)
    return


@app.cell
def _(port, print_highlight, text_1):
    import openai
    client = openai.Client(base_url=f'http://127.0.0.1:{port}/v1', api_key='None')
    _response = client.embeddings.create(model='Alibaba-NLP/gte-Qwen2-1.5B-instruct', input=text_1)
    embedding = _response.data[0].embedding[:10]
    print_highlight(f'Text embedding (first 10): {embedding}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using Input IDs

    SGLang also supports `input_ids` as input to get the embedding.
    """)
    return


@app.cell
def _(json, port, print_highlight, subprocess, text_1):
    import os
    from transformers import AutoTokenizer
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-1.5B-instruct')
    input_ids = tokenizer.encode(text_1)
    curl_ids = f"""curl -s http://localhost:{port}/v1/embeddings   -H "Content-Type: application/json"   -d '{{"model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "input": {json.dumps(input_ids)}}}'"""
    input_ids_embedding = json.loads(subprocess.check_output(curl_ids, shell=True))['data'][0]['embedding']
    print_highlight(f'Input IDs embedding (first 10): {input_ids_embedding[:10]}')
    return


@app.cell
def _(embedding_process, terminate_process):
    terminate_process(embedding_process)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-Modal Embedding Model
    Please refer to [Multi-Modal Embedding Model](../supported_models/embedding_models.md)
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
