import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OpenAI APIs - Vision

    SGLang provides OpenAI-compatible APIs to enable a smooth transition from OpenAI services to self-hosted local models.
    A complete reference for the API is available in the [OpenAI API Reference](https://platform.openai.com/docs/guides/vision).
    This tutorial covers the vision APIs for vision language models.

    SGLang supports various vision language models such as Llama 3.2, LLaVA-OneVision, Qwen2.5-VL, Gemma3 and [more](../supported_models/multimodal_language_models.md).

    As an alternative to the OpenAI API, you can also use the [SGLang offline engine](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/offline_batch_inference_vlm.py).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Launch A Server

    Launch the server in your terminal and wait for it to initialize.
    """)
    return


@app.cell
def _():
    from sglang.test.doc_patch import launch_server_cmd
    from sglang.utils import wait_for_server, print_highlight, terminate_process

    vision_process, port = launch_server_cmd(
        """
    python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct --log-level warning
    """
    )

    wait_for_server(f"http://localhost:{port}")
    return port, print_highlight, terminate_process, vision_process


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using cURL

    Once the server is up, you can send test requests using curl or requests.
    """)
    return


@app.cell
def _(port, print_highlight):
    import subprocess
    curl_command = f"""\ncurl -s http://localhost:{port}/v1/chat/completions \\\n  -H "Content-Type: application/json" \\\n  -d '{{\n    "model": "Qwen/Qwen2.5-VL-7B-Instruct",\n    "messages": [\n      {{\n        "role": "user",\n        "content": [\n          {{\n            "type": "text",\n            "text": "What’s in this image?"\n          }},\n          {{\n            "type": "image_url",\n            "image_url": {{\n              "url": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"\n            }}\n          }}\n        ]\n      }}\n    ],\n    "max_tokens": 300\n  }}'\n"""
    _response = subprocess.check_output(curl_command, shell=True).decode()
    print_highlight(_response)
    _response = subprocess.check_output(curl_command, shell=True).decode()
    print_highlight(_response)
    return


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
    data = {'model': 'Qwen/Qwen2.5-VL-7B-Instruct', 'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'What’s in this image?'}, {'type': 'image_url', 'image_url': {'url': 'https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true'}}]}], 'max_tokens': 300}
    _response = requests.post(url, json=data)
    print_highlight(_response.text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using OpenAI Python Client
    """)
    return


@app.cell
def _(port, print_highlight):
    from openai import OpenAI
    _client = OpenAI(base_url=f'http://localhost:{port}/v1', api_key='None')
    _response = _client.chat.completions.create(model='Qwen/Qwen2.5-VL-7B-Instruct', messages=[{'role': 'user', 'content': [{'type': 'text', 'text': 'What is in this image?'}, {'type': 'image_url', 'image_url': {'url': 'https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true'}}]}], max_tokens=300)
    print_highlight(_response.choices[0].message.content)
    return (OpenAI,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multiple-Image Inputs

    The server also supports multiple images and interleaved text and images if the model supports it.
    """)
    return


@app.cell
def _(OpenAI, port, print_highlight):
    _client = OpenAI(base_url=f'http://localhost:{port}/v1', api_key='None')
    _response = _client.chat.completions.create(model='Qwen/Qwen2.5-VL-7B-Instruct', messages=[{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true'}}, {'type': 'image_url', 'image_url': {'url': 'https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png'}}, {'type': 'text', 'text': 'I have two very different images. They are not related at all. Please describe the first image in one sentence, and then describe the second image in another sentence.'}]}], temperature=0)
    print_highlight(_response.choices[0].message.content)
    return


@app.cell
def _(terminate_process, vision_process):
    terminate_process(vision_process)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
