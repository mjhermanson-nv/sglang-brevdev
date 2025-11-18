import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SGLang Native APIs

    Apart from the OpenAI compatible APIs, the SGLang Runtime also provides its native server APIs. We introduce the following APIs:

    - `/generate` (text generation model)
    - `/get_model_info`
    - `/get_server_info`
    - `/health`
    - `/health_generate`
    - `/flush_cache`
    - `/update_weights`
    - `/encode`(embedding model)
    - `/v1/rerank`(cross encoder rerank model)
    - `/classify`(reward model)
    - `/start_expert_distribution_record`
    - `/stop_expert_distribution_record`
    - `/dump_expert_distribution_record`
    - `/tokenize`
    - `/detokenize`
    - A full list of these APIs can be found at [http_server.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py)

    We mainly use `requests` to test these APIs in the following examples. You can also use `curl`.
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

    server_process, port = launch_server_cmd(
        "python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0 --attention-backend triton --log-level warning"
    )

    wait_for_server(f"http://localhost:{port}")
    return (
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
    ## Generate (text generation model)
    Generate completions. This is similar to the `/v1/completions` in OpenAI API. Detailed parameters can be found in the [sampling parameters](sampling_params.md).
    """)
    return


@app.cell
def _(port, print_highlight):
    import requests
    _url = f'http://localhost:{port}/generate'
    _data = {'text': 'What is the capital of France?'}
    _response = requests.post(_url, json=_data)
    print_highlight(_response.json())
    return (requests,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get Model Info

    Get the information of the model.

    - `model_path`: The path/name of the model.
    - `is_generation`: Whether the model is used as generation model or embedding model.
    - `tokenizer_path`: The path/name of the tokenizer.
    - `preferred_sampling_params`: The default sampling params specified via `--preferred-sampling-params`. `None` is returned in this example as we did not explicitly configure it in server args.
    - `weight_version`: This field contains the version of the model weights. This is often used to track changes or updates to the model’s trained parameters.
    - `has_image_understanding`: Whether the model has image-understanding capability.
    - `has_audio_understanding`: Whether the model has audio-understanding capability.
    """)
    return


@app.cell
def _(port, print_highlight, requests):
    _url = f'http://localhost:{port}/get_model_info'
    _response = requests.get(_url)
    _response_json = _response.json()
    print_highlight(_response_json)
    assert _response_json['model_path'] == 'qwen/qwen2.5-0.5b-instruct'
    assert _response_json['is_generation'] is True
    assert _response_json['tokenizer_path'] == 'qwen/qwen2.5-0.5b-instruct'
    assert _response_json['preferred_sampling_params'] is None
    assert _response_json.keys() == {'model_path', 'is_generation', 'tokenizer_path', 'preferred_sampling_params', 'weight_version', 'has_image_understanding', 'has_audio_understanding'}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get Server Info
    Gets the server information including CLI arguments, token limits, and memory pool sizes.
    - Note: `get_server_info` merges the following deprecated endpoints:
      - `get_server_args`
      - `get_memory_pool_size`
      - `get_max_total_num_tokens`
    """)
    return


@app.cell
def _(port, print_highlight, requests):
    _url = f'http://localhost:{port}/get_server_info'
    _response = requests.get(_url)
    print_highlight(_response.text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Health Check
    - `/health`: Check the health of the server.
    - `/health_generate`: Check the health of the server by generating one token.
    """)
    return


@app.cell
def _(port, print_highlight, requests):
    _url = f'http://localhost:{port}/health_generate'
    _response = requests.get(_url)
    print_highlight(_response.text)
    return


@app.cell
def _(port, print_highlight, requests):
    _url = f'http://localhost:{port}/health'
    _response = requests.get(_url)
    print_highlight(_response.text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Flush Cache

    Flush the radix cache. It will be automatically triggered when the model weights are updated by the `/update_weights` API.
    """)
    return


@app.cell
def _(port, print_highlight, requests):
    _url = f'http://localhost:{port}/flush_cache'
    _response = requests.post(_url)
    print_highlight(_response.text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Update Weights From Disk

    Update model weights from disk without restarting the server. Only applicable for models with the same architecture and parameter size.

    SGLang support `update_weights_from_disk` API for continuous evaluation during training (save checkpoint to disk and update weights from disk).
    """)
    return


@app.cell
def _(port, print_highlight, requests):
    # successful update with same architecture and size
    _url = f'http://localhost:{port}/update_weights_from_disk'
    _data = {'model_path': 'qwen/qwen2.5-0.5b-instruct'}
    _response = requests.post(_url, json=_data)
    print_highlight(_response.text)
    assert _response.json()['success'] is True
    assert _response.json()['message'] == 'Succeeded to update model weights.'
    return


@app.cell
def _(port, print_highlight, requests):
    # failed update with different parameter size or wrong name
    _url = f'http://localhost:{port}/update_weights_from_disk'
    _data = {'model_path': 'qwen/qwen2.5-0.5b-instruct-wrong'}
    _response = requests.post(_url, json=_data)
    _response_json = _response.json()
    print_highlight(_response_json)
    assert _response_json['success'] is False
    assert _response_json['message'] == 'Failed to get weights iterator: qwen/qwen2.5-0.5b-instruct-wrong (repository not found).'
    return


@app.cell
def _(server_process, terminate_process):
    terminate_process(server_process)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Encode (embedding model)

    Encode text into embeddings. Note that this API is only available for [embedding models](openai_api_embeddings.ipynb) and will raise an error for generation models.
    Therefore, we launch a new server to server an embedding model.
    """)
    return


@app.cell
def _(launch_server_cmd, wait_for_server):
    (embedding_process, port_1) = launch_server_cmd('\npython3 -m sglang.launch_server --model-path Alibaba-NLP/gte-Qwen2-1.5B-instruct     --host 0.0.0.0 --is-embedding --attention-backend triton --log-level warning\n')
    wait_for_server(f'http://localhost:{port_1}')
    return embedding_process, port_1


@app.cell
def _(port_1, print_highlight, requests):
    _url = f'http://localhost:{port_1}/encode'
    _data = {'model': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct', 'text': 'Once upon a time'}
    _response = requests.post(_url, json=_data)
    _response_json = _response.json()
    print_highlight(f"Text embedding (first 10): {_response_json['embedding'][:10]}")
    return


@app.cell
def _(embedding_process, terminate_process):
    terminate_process(embedding_process)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## v1/rerank (cross encoder rerank model)
    Rerank a list of documents given a query using a cross-encoder model. Note that this API is only available for cross encoder model like [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) with `attention-backend` `triton` and `torch_native`.
    """)
    return


@app.cell
def _(launch_server_cmd, wait_for_server):
    (reranker_process, port_2) = launch_server_cmd('\npython3 -m sglang.launch_server --model-path BAAI/bge-reranker-v2-m3     --host 0.0.0.0 --disable-radix-cache --chunked-prefill-size -1 --attention-backend triton --is-embedding --log-level warning\n')
    wait_for_server(f'http://localhost:{port_2}')
    return port_2, reranker_process


@app.cell
def _(port_2, print_highlight, requests):
    _url = f'http://localhost:{port_2}/v1/rerank'
    _data = {'model': 'BAAI/bge-reranker-v2-m3', 'query': 'what is panda?', 'documents': ['hi', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']}
    _response = requests.post(_url, json=_data)
    _response_json = _response.json()
    for item in _response_json:
        print_highlight(f"Score: {item['score']:.2f} - Document: '{item['document']}'")
    return


@app.cell
def _(reranker_process, terminate_process):
    terminate_process(reranker_process)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classify (reward model)

    SGLang Runtime also supports reward models. Here we use a reward model to classify the quality of pairwise generations.
    """)
    return


@app.cell
def _(launch_server_cmd, wait_for_server):
    # Note that SGLang now treats embedding models and reward models as the same type of models.
    # This will be updated in the future.
    (reward_process, port_3) = launch_server_cmd('\npython3 -m sglang.launch_server --model-path Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 --host 0.0.0.0 --is-embedding --attention-backend triton --log-level warning\n')
    wait_for_server(f'http://localhost:{port_3}')
    return port_3, reward_process


@app.cell
def _(port_3, print_highlight, requests):
    from transformers import AutoTokenizer
    PROMPT = 'What is the range of the numeric output of a sigmoid node in a neural network?'
    RESPONSE1 = 'The output of a sigmoid node is bounded between -1 and 1.'
    RESPONSE2 = 'The output of a sigmoid node is bounded between 0 and 1.'
    CONVS = [[{'role': 'user', 'content': PROMPT}, {'role': 'assistant', 'content': RESPONSE1}], [{'role': 'user', 'content': PROMPT}, {'role': 'assistant', 'content': RESPONSE2}]]
    tokenizer = AutoTokenizer.from_pretrained('Skywork/Skywork-Reward-Llama-3.1-8B-v0.2')
    prompts = tokenizer.apply_chat_template(CONVS, tokenize=False)
    _url = f'http://localhost:{port_3}/classify'
    _data = {'model': 'Skywork/Skywork-Reward-Llama-3.1-8B-v0.2', 'text': prompts}
    responses = requests.post(_url, json=_data).json()
    for _response in responses:
        print_highlight(f"reward: {_response['embedding'][0]}")
    return


@app.cell
def _(reward_process, terminate_process):
    terminate_process(reward_process)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Capture expert selection distribution in MoE models

    SGLang Runtime supports recording the number of times an expert is selected in a MoE model run for each expert in the model. This is useful when analyzing the throughput of the model and plan for optimization.

    *Note: We only print out the first 10 lines of the csv below for better readability. Please adjust accordingly if you want to analyze the results more deeply.*
    """)
    return


@app.cell
def _(launch_server_cmd, wait_for_server):
    (expert_record_server_process, port_4) = launch_server_cmd('python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B --host 0.0.0.0 --expert-distribution-recorder-mode stat --attention-backend triton --log-level warning')
    wait_for_server(f'http://localhost:{port_4}')
    return expert_record_server_process, port_4


@app.cell
def _(port_4, print_highlight, requests):
    _response = requests.post(f'http://localhost:{port_4}/start_expert_distribution_record')
    print_highlight(_response)
    _url = f'http://localhost:{port_4}/generate'
    _data = {'text': 'What is the capital of France?'}
    _response = requests.post(_url, json=_data)
    print_highlight(_response.json())
    _response = requests.post(f'http://localhost:{port_4}/stop_expert_distribution_record')
    print_highlight(_response)
    _response = requests.post(f'http://localhost:{port_4}/dump_expert_distribution_record')
    print_highlight(_response)
    return


@app.cell
def _(expert_record_server_process, terminate_process):
    terminate_process(expert_record_server_process)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tokenize/Detokenize Example (Round Trip)

    This example demonstrates how to use the /tokenize and /detokenize endpoints together. We first tokenize a string, then detokenize the resulting IDs to reconstruct the original text. This workflow is useful when you need to handle tokenization externally but still leverage the server for detokenization.
    """)
    return


@app.cell
def _(launch_server_cmd, wait_for_server):
    (tokenizer_free_server_process, port_5) = launch_server_cmd('\npython3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --attention-backend triton\n')
    wait_for_server(f'http://localhost:{port_5}')
    return port_5, tokenizer_free_server_process


@app.cell
def _(port_5, print_highlight, requests):
    base_url = f'http://localhost:{port_5}'
    tokenize_url = f'{base_url}/tokenize'
    detokenize_url = f'{base_url}/detokenize'
    model_name = 'qwen/qwen2.5-0.5b-instruct'
    input_text = 'SGLang provides efficient tokenization endpoints.'
    print_highlight(f"Original Input Text:\n'{input_text}'")
    tokenize_payload = {'model': model_name, 'prompt': input_text, 'add_special_tokens': False}
    try:
        tokenize_response = requests.post(tokenize_url, json=tokenize_payload)
        tokenize_response.raise_for_status()
        tokenization_result = tokenize_response.json()
    # --- tokenize the input text ---
        token_ids = tokenization_result.get('tokens')
        if not token_ids:
            raise ValueError('Tokenization returned empty tokens.')
        print_highlight(f'\nTokenized Output (IDs):\n{token_ids}')
        print_highlight(f"Token Count: {tokenization_result.get('count')}")
        print_highlight(f"Max Model Length: {tokenization_result.get('max_model_len')}")
        detokenize_payload = {'model': model_name, 'tokens': token_ids, 'skip_special_tokens': True}
        detokenize_response = requests.post(detokenize_url, json=detokenize_payload)
        detokenize_response.raise_for_status()
        detokenization_result = detokenize_response.json()
        reconstructed_text = detokenization_result.get('text')
        print_highlight(f"\nDetokenized Output (Text):\n'{reconstructed_text}'")
        if input_text == reconstructed_text:
            print_highlight('\nRound Trip Successful: Original and reconstructed text match.')
        else:
            print_highlight('\nRound Trip Mismatch: Original and reconstructed text differ.')
    except requests.exceptions.RequestException as e:
        print_highlight(f'\nHTTP Request Error: {e}')
    except Exception as e:  # --- detokenize the obtained token IDs ---
        print_highlight(f'\nAn error occurred: {e}')
    return


@app.cell
def _(terminate_process, tokenizer_free_server_process):
    terminate_process(tokenizer_free_server_process)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
