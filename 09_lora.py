import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LoRA Serving

    ⚠️ **Important: HuggingFace Authentication Required**

    This notebook uses `meta-llama/Meta-Llama-3.1-8B-Instruct` which is a gated model requiring HuggingFace authentication.

    **To access gated models:**
    1. Visit https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
    2. Accept the model license and request access
    3. Generate a token at https://huggingface.co/settings/tokens
       - **Important**: If using a fine-grained token, enable "public gated repositories" permission
       - Or use a classic token (which has this permission by default)
    4. Enter your token in the cell below

    **Token Requirements:**
    - Must have access to the gated model (request access first)
    - Fine-grained tokens need "public gated repositories" permission enabled
    - Classic tokens work automatically

    Alternatively, you can modify the examples to use non-gated models like `Qwen/Qwen2.5-7B-Instruct` and compatible LoRA adapters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    hf_token_input = mo.ui.text(
        label="HuggingFace Token",
        placeholder="hf_...",
        kind="password",
        full_width=True
    )
    hf_token_input
    return (hf_token_input,)


@app.cell
def _(hf_token_input):
    import os

    if hf_token_input.value:
        token = hf_token_input.value
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token

        # Also login programmatically to huggingface_hub
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            print(f"✓ HuggingFace token set and authenticated")
            print(f"⚠️  Note: If you get 403 errors, ensure your token has 'public gated repositories' permission")
            print(f"   (Fine-grained tokens need this enabled; classic tokens have it by default)")
        except ImportError:
            print(f"✓ HuggingFace token set (install huggingface_hub for programmatic login)")
        except Exception as e:
            print(f"⚠️  Token set but login failed: {e}")
            print(f"   Make sure your token has access to gated repositories")
    elif os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        print("✓ Using existing HuggingFace token from environment")
    else:
        print("⚠️  No HuggingFace token set. Please enter your token above.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    SGLang enables the use of [LoRA adapters](https://arxiv.org/abs/2106.09685) with a base model. By incorporating techniques from [S-LoRA](https://arxiv.org/pdf/2311.03285) and [Punica](https://arxiv.org/pdf/2310.18547), SGLang can efficiently support multiple LoRA adapters for different sequences within a single batch of inputs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arguments for LoRA Serving
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following server arguments are relevant for multi-LoRA serving:

    * `enable_lora`: Enable LoRA support for the model. This argument is automatically set to True if `--lora-paths` is provided for backward compatibility.

    * `lora_paths`: The list of LoRA adapters to load. Each adapter must be specified in one of the following formats: <PATH> | <NAME>=<PATH> | JSON with schema {"lora_name":str,"lora_path":str,"pinned":bool}.

    * `max_loras_per_batch`: Maximum number of adaptors used by each batch. This argument can affect the amount of GPU memory reserved for multi-LoRA serving, so it should be set to a smaller value when memory is scarce. Defaults to be 8.

    * `max_loaded_loras`: If specified, it limits the maximum number of LoRA adapters loaded in CPU memory at a time. The value must be greater than or equal to `max-loras-per-batch`.

    * `lora_eviction_policy`: LoRA adapter eviction policy when GPU memory pool is full. `lru`: Least Recently Used (default, better cache efficiency). `fifo`: First-In-First-Out.

    * `lora_backend`: The backend of running GEMM kernels for Lora modules. Currently we support Triton LoRA backend (`triton`) and Chunked SGMV backend (`csgmv`). In the future, faster backend built upon Cutlass or Cuda kernels will be added.

    * `max_lora_rank`: The maximum LoRA rank that should be supported. If not specified, it will be automatically inferred from the adapters provided in `--lora-paths`. This argument is needed when you expect to dynamically load adapters of larger LoRA rank after server startup.

    * `lora_target_modules`: The union set of all target modules where LoRA should be applied (e.g., `q_proj`, `k_proj`, `gate_proj`). If not specified, it will be automatically inferred from the adapters provided in `--lora-paths`. This argument is needed when you expect to dynamically load adapters of different target modules after server startup. You can also set it to `all` to enable LoRA for all supported modules. However, enabling LoRA on additional modules introduces a minor performance overhead. If your application is performance-sensitive, we recommend only specifying the modules for which you plan to load adapters.

    * `--max-lora-chunk-size`: Maximum chunk size for the ChunkedSGMV LoRA backend. Only used when --lora-backend is 'csgmv'. Choosing a larger value might improve performance. Please tune this value based on your hardware and workload as needed. Defaults to 16.

    * `tp_size`: LoRA serving along with Tensor Parallelism is supported by SGLang. `tp_size` controls the number of GPUs for tensor parallelism. More details on the tensor sharding strategy can be found in [S-Lora](https://arxiv.org/pdf/2311.03285) paper.

    From client side, the user needs to provide a list of strings as input batch, and a list of adaptor names that each input sequence corresponds to.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Usage

    ### Serving Single Adaptor
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Note:** SGLang supports LoRA adapters through two APIs:

    1. **OpenAI-Compatible API** (`/v1/chat/completions`, `/v1/completions`): Use the `model:adapter-name` syntax. See [OpenAI API with LoRA](../basic_usage/openai_api_completions.ipynb#Using-LoRA-Adapters) for examples.

    2. **Native API** (`/generate`): Pass `lora_path` in the request body (shown below).
    """)
    return


@app.cell
def _():
    import json
    import requests

    from sglang.test.doc_patch import launch_server_cmd
    from sglang.utils import wait_for_server, terminate_process
    return launch_server_cmd, requests, terminate_process, wait_for_server


@app.cell
def _(launch_server_cmd, wait_for_server):
    server_process, port = launch_server_cmd(
        """
    python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
        --enable-lora \
        --lora-paths lora0=algoprog/fact-generation-llama-3.1-8b-instruct-lora \
        --max-loras-per-batch 1 \
        --attention-backend triton \
        --log-level warning \
    """
    )

    wait_for_server(f"http://localhost:{port}")
    return port, server_process


@app.cell
def _(port, requests):
    url = f'http://127.0.0.1:{port}'
    _json_data = {'text': ['List 3 countries and their capitals.', 'List 3 countries and their capitals.'], 'sampling_params': {'max_new_tokens': 32, 'temperature': 0}, 'lora_path': ['lora0', None]}
    _response = requests.post(url + '/generate', json=_json_data)
    print(f"Output 0: {_response.json()[0]['text']}")
    print(f"Output 1: {_response.json()[1]['text']}")  # The first input uses lora0, and the second input uses the base model
    return


@app.cell
def _(server_process, terminate_process):
    terminate_process(server_process)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Serving Multiple Adaptors
    """)
    return


@app.cell
def _(launch_server_cmd, wait_for_server):
    (server_process_1, port_1) = launch_server_cmd('\npython3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct     --enable-lora     --lora-paths lora0=algoprog/fact-generation-llama-3.1-8b-instruct-lora     lora1=Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16     --max-loras-per-batch 2     --attention-backend triton     --log-level warning ')
    wait_for_server(f'http://localhost:{port_1}')
    return port_1, server_process_1


@app.cell
def _(port_1, requests):
    url_1 = f'http://127.0.0.1:{port_1}'
    _json_data = {'text': ['List 3 countries and their capitals.', 'List 3 countries and their capitals.'], 'sampling_params': {'max_new_tokens': 32, 'temperature': 0}, 'lora_path': ['lora0', 'lora1']}
    _response = requests.post(url_1 + '/generate', json=_json_data)
    print(f"Output 0: {_response.json()[0]['text']}")
    print(f"Output 1: {_response.json()[1]['text']}")
    return


@app.cell
def _(server_process_1, terminate_process):
    terminate_process(server_process_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dynamic LoRA loading
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Instead of specifying all adapters during server startup via `--lora-paths`. You can also load & unload LoRA adapters dynamically via the `/load_lora_adapter` and `/unload_lora_adapter` API.

    When using dynamic LoRA loading, it's recommended to explicitly specify both `--max-lora-rank` and `--lora-target-modules` at startup. For backward compatibility, SGLang will infer these values from `--lora-paths` if they are not explicitly provided. However, in that case, you would have to ensure that all dynamically loaded adapters share the same shape (rank and target modules) as those in the initial `--lora-paths` or are strictly "smaller".
    """)
    return


@app.cell
def _(launch_server_cmd, wait_for_server):
    lora0 = 'Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16'  # rank - 4, target modules - q_proj, k_proj, v_proj, o_proj, gate_proj
    lora1 = 'algoprog/fact-generation-llama-3.1-8b-instruct-lora'  # rank - 64, target modules - q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    lora0_new = 'philschmid/code-llama-3-1-8b-text-to-sql-lora'  # rank - 256, target modules - q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    (server_process_2, port_2) = launch_server_cmd('\n    python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct     --enable-lora     --cuda-graph-max-bs 2     --max-loras-per-batch 2     --max-lora-rank 256\n    --lora-target-modules all\n    --attention-backend triton\n    --log-level warning\n    ')
    url_2 = f'http://127.0.0.1:{port_2}'
    # The `--target-lora-modules` param below is technically not needed, as the server will infer it from lora0 which already has all the target modules specified.
    # We are adding it here just to demonstrate usage.
    wait_for_server(url_2)
    return lora0, lora0_new, lora1, port_2, server_process_2, url_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load adapter lora0
    """)
    return


@app.cell
def _(lora0, requests, url_2):
    _response = requests.post(url_2 + '/load_lora_adapter', json={'lora_name': 'lora0', 'lora_path': lora0})
    if _response.status_code == 200:
        print('LoRA adapter loaded successfully.', _response.json())
    else:
        print('Failed to load LoRA adapter.', _response.json())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load adapter lora1:
    """)
    return


@app.cell
def _(lora1, requests, url_2):
    _response = requests.post(url_2 + '/load_lora_adapter', json={'lora_name': 'lora1', 'lora_path': lora1})
    if _response.status_code == 200:
        print('LoRA adapter loaded successfully.', _response.json())
    else:
        print('Failed to load LoRA adapter.', _response.json())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check inference output:
    """)
    return


@app.cell
def _(port_2, requests):
    url_3 = f'http://127.0.0.1:{port_2}'
    _json_data = {'text': ['List 3 countries and their capitals.', 'List 3 countries and their capitals.'], 'sampling_params': {'max_new_tokens': 32, 'temperature': 0}, 'lora_path': ['lora0', 'lora1']}
    _response = requests.post(url_3 + '/generate', json=_json_data)
    print(f"Output from lora0: \n{_response.json()[0]['text']}\n")
    print(f"Output from lora1 (updated): \n{_response.json()[1]['text']}\n")
    return (url_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Unload lora0 and replace it with a different adapter:
    """)
    return


@app.cell
def _(lora0_new, requests, url_3):
    _response = requests.post(url_3 + '/unload_lora_adapter', json={'lora_name': 'lora0'})
    _response = requests.post(url_3 + '/load_lora_adapter', json={'lora_name': 'lora0', 'lora_path': lora0_new})
    if _response.status_code == 200:
        print('LoRA adapter loaded successfully.', _response.json())
    else:
        print('Failed to load LoRA adapter.', _response.json())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check output again:
    """)
    return


@app.cell
def _(port_2, requests):
    url_4 = f'http://127.0.0.1:{port_2}'
    _json_data = {'text': ['List 3 countries and their capitals.', 'List 3 countries and their capitals.'], 'sampling_params': {'max_new_tokens': 32, 'temperature': 0}, 'lora_path': ['lora0', 'lora1']}
    _response = requests.post(url_4 + '/generate', json=_json_data)
    print(f"Output from lora0: \n{_response.json()[0]['text']}\n")
    print(f"Output from lora1 (updated): \n{_response.json()[1]['text']}\n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### OpenAI-compatible API usage

    You can use LoRA adapters via the OpenAI-compatible APIs by specifying the adapter in the `model` field using the `base-model:adapter-name` syntax (for example, `qwen/qwen2.5-0.5b-instruct:adapter_a`). For more details and examples, see the “Using LoRA Adapters” section in the OpenAI API documentation: [openai_api_completions.ipynb](../basic_usage/openai_api_completions.ipynb).
    """)
    return


@app.cell
def _(server_process_2, terminate_process):
    terminate_process(server_process_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### LoRA GPU Pinning
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Another advanced option is to specify adapters as `pinned` during loading. When an adapter is pinned, it is permanently assigned to one of the available GPU pool slots (as configured by `--max-loras-per-batch`) and will not be evicted from GPU memory during runtime. Instead, it remains resident until it is explicitly unloaded.

    This can improve performance in scenarios where the same adapter is frequently used across requests, by avoiding repeated memory transfers and reinitialization overhead. However, since GPU pool slots are limited, pinning adapters reduces the flexibility of the system to dynamically load other adapters on demand. If too many adapters are pinned, it may lead to degraded performance, or in the most extreme case (`Number of pinned adapters == max-loras-per-batch`), halt all unpinned requests. Therefore, currently SGLang limits maximal number of pinned adapters to `max-loras-per-batch - 1` to prevent unexpected starvations.

    In the example below, we start a server with `lora1` loaded as pinned, `lora2` and `lora3` loaded as regular (unpinned) adapters. Please note that, we intentionally specify `lora2` and `lora3` in two different formats to demonstrate that both are supported.
    """)
    return


@app.cell
def _(launch_server_cmd, wait_for_server):
    (server_process_3, port_3) = launch_server_cmd('\n    python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct     --enable-lora     --cuda-graph-max-bs 8     --max-loras-per-batch 3     --max-lora-rank 256     --lora-target-modules all     --lora-paths         {"lora_name":"lora0","lora_path":"Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16","pinned":true}         {"lora_name":"lora1","lora_path":"algoprog/fact-generation-llama-3.1-8b-instruct-lora"}         lora2=philschmid/code-llama-3-1-8b-text-to-sql-lora\n    --attention-backend triton\n    --log-level warning\n    ')
    url_5 = f'http://127.0.0.1:{port_3}'
    wait_for_server(url_5)
    return port_3, server_process_3, url_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also specify adapter as pinned during dynamic adapter loading. In the example below, we reload `lora2` as pinned adapter:
    """)
    return


@app.cell
def _(requests, url_5):
    _response = requests.post(url_5 + '/unload_lora_adapter', json={'lora_name': 'lora1'})
    _response = requests.post(url_5 + '/load_lora_adapter', json={'lora_name': 'lora1', 'lora_path': 'algoprog/fact-generation-llama-3.1-8b-instruct-lora', 'pinned': True})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Verify that the results are expected:
    """)
    return


@app.cell
def _(port_3, requests):
    url_6 = f'http://127.0.0.1:{port_3}'
    _json_data = {'text': ['List 3 countries and their capitals.', 'List 3 countries and their capitals.', 'List 3 countries and their capitals.'], 'sampling_params': {'max_new_tokens': 32, 'temperature': 0}, 'lora_path': ['lora0', 'lora1', 'lora2']}
    _response = requests.post(url_6 + '/generate', json=_json_data)
    print(f"Output from lora0 (pinned): \n{_response.json()[0]['text']}\n")
    print(f"Output from lora1 (pinned): \n{_response.json()[1]['text']}\n")
    print(f"Output from lora2 (not pinned): \n{_response.json()[2]['text']}\n")
    return


@app.cell
def _(server_process_3, terminate_process):
    terminate_process(server_process_3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing LoRA Backend

    SGLang supports two LoRA backends that you can choose from using the `--lora-backend` argument:

    - `triton`: Default basic Triton-based backend.
    - `csgmv`: Chunked SGMV backend optimized for high concurrency scenarios.

    The `csgmv` backend was recently introduced to improve performance especially at high-concurrency scenarios. Our benchmark shows that it achieves 20% to 80% latency improvements over the basic triton backend.
    Currently it is at preview phase, we expect to make it our the default LoRA backend in future release. Before that, you can adopt it by manually setting the `--lora-backend` server config.
    """)
    return


@app.cell
def _(launch_server_cmd):
    (server_process_4, port_4) = launch_server_cmd('\n    python3 -m sglang.launch_server     --model-path meta-llama/Meta-Llama-3.1-8B-Instruct     --enable-lora     --lora-backend csgmv     --max-loras-per-batch 16     --lora-paths lora1=path/to/lora1 lora2=path/to/lora2     --attention-backend triton\n    ')
    return (server_process_4,)


@app.cell
def _(server_process_4, terminate_process):
    terminate_process(server_process_4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Future Works

    The development roadmap for LoRA-related features can be found in this [issue](https://github.com/sgl-project/sglang/issues/2929). Other features, including Embedding Layer, Unified Paging, Cutlass backend are still under development.
    """)
    return


if __name__ == "__main__":
    app.run()
