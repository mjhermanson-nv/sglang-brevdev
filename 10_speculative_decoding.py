import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Speculative Decoding

    SGLang now provides an EAGLE-based (EAGLE-2/EAGLE-3) speculative decoding option. Our implementation aims to maximize speed and efficiency and is considered to be among the fastest in open-source LLM engines.

    ⚠️ **Important: HuggingFace Authentication Required**

    This notebook uses gated Meta Llama models requiring HuggingFace authentication.

    **To access gated models:**
    1. Visit the model pages and accept the licenses:
       - https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
       - https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
       - https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
    2. Generate a token at https://huggingface.co/settings/tokens
       - **Important**: If using a fine-grained token, enable "public gated repositories" permission
       - Or use a classic token (which has this permission by default)
    3. Enter your token in the cell below

    **Token Requirements:**
    - Must have access to the gated models (request access first)
    - Fine-grained tokens need "public gated repositories" permission enabled
    - Classic tokens work automatically

    ### Performance Highlights

    Please see below for the huge improvements on throughput for LLaMA-Instruct 3.1 8B tested on MT bench that can be achieved via EAGLE3 decoding.
    For further details please see the [EAGLE3 paper](https://arxiv.org/pdf/2503.01840).

    | Method | Throughput (tokens/s) |
    |--------|----------------|
    | SGLang (w/o speculative, 1x H100) | 158.34 tokens/s |
    | SGLang + EAGLE-2 (1x H100) | 244.10 tokens/s |
    | SGLang + EAGLE-3 (1x H100) | 373.25 tokens/s |
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    hf_token_input = mo.ui.text(
        label="HuggingFace Token",
        placeholder="hf_...",
        kind="password",
        full_width=True
    )
    hf_token_input
    return hf_token_input, mo


@app.cell
def _(hf_token_input):
    import os

    # Set environment variable to allow longer context lengths for speculative decoding
    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

    if hf_token_input.value:
        token = hf_token_input.value
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token

        # Also login programmatically to huggingface_hub
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            print(f"✓ HuggingFace token set and authenticated")
            print(f"✓ Context length override enabled for speculative decoding")
            print(f"⚠️  Note: If you get 403 errors, ensure your token has 'public gated repositories' permission")
            print(f"   (Fine-grained tokens need this enabled; classic tokens have it by default)")
        except ImportError:
            print(f"✓ HuggingFace token set (install huggingface_hub for programmatic login)")
            print(f"✓ Context length override enabled for speculative decoding")
        except Exception as e:
            print(f"⚠️  Token set but login failed: {e}")
            print(f"   Make sure your token has access to gated repositories")
    elif os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        print("✓ Using existing HuggingFace token from environment")
        print(f"✓ Context length override enabled for speculative decoding")
    else:
        print("⚠️  No HuggingFace token set. Please enter your token above.")
        print(f"✓ Context length override enabled for speculative decoding")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EAGLE Decoding

    To enable EAGLE speculative decoding the following parameters are relevant:
    * `speculative_draft_model_path`: Specifies draft model. This parameter is required.
    * `speculative_num_steps`: Depth of autoregressive drafting. Increases speculation range but risks rejection cascades. Default is 5.
    * `speculative_eagle_topk`: Branching factor per step. Improves candidate diversity, will lead to higher acceptance rate, but more lead to higher memory/compute consumption. Default is 4.
    * `speculative_num_draft_tokens`: Maximum parallel verification capacity. Allows deeper tree evaluation but will lead to higher GPU memory usage. Default is 8.

    These parameters are the same for EAGLE-2 and EAGLE-3.

    You can find the best combinations of these parameters with [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py).

    In the documentation below, we set `--cuda-graph-max-bs` to be a small value for faster engine startup. For your own workloads, please tune the above parameters together with `--cuda-graph-max-bs`, `--max-running-requests`, `--mem-fraction-static` for the best performance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EAGLE-2 decoding

    You can enable EAGLE-2 decoding by setting `--speculative-algorithm EAGLE` and choosing an appropriate model.
    """)
    return


@app.cell
def _():
    from sglang.test.doc_patch import launch_server_cmd
    from sglang.utils import wait_for_server, print_highlight, terminate_process

    import openai
    return (
        launch_server_cmd,
        openai,
        print_highlight,
        terminate_process,
        wait_for_server,
    )


@app.cell
def _(launch_server_cmd, wait_for_server):
    server_process, port = launch_server_cmd(
        """
    python3 -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf  --speculative-algorithm EAGLE \
        --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B --speculative-num-steps 3 \
        --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 --cuda-graph-max-bs 8 --log-level warning
    """
    )

    wait_for_server(f"http://localhost:{port}")
    return port, server_process


@app.cell
def _(openai, port, print_highlight):
    _client = openai.Client(base_url=f'http://127.0.0.1:{port}/v1', api_key='None')
    _response = _client.chat.completions.create(model='meta-llama/Llama-2-7b-chat-hf', messages=[{'role': 'user', 'content': 'List 3 countries and their capitals.'}], temperature=0, max_tokens=64)
    print_highlight(f'Response: {_response}')
    return


@app.cell
def _(server_process, terminate_process):
    terminate_process(server_process)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EAGLE-2 Decoding with `torch.compile`

    You can also enable `torch.compile` for further optimizations and optionally set `--torch-compile-max-bs`:
    """)
    return


@app.cell
def _(launch_server_cmd, wait_for_server):
    (server_process_1, port_1) = launch_server_cmd('\npython3 -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf  --speculative-algorithm EAGLE     --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B --speculative-num-steps 5         --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 --mem-fraction 0.6             --enable-torch-compile --torch-compile-max-bs 2 --log-level warning\n')
    wait_for_server(f'http://localhost:{port_1}')
    return port_1, server_process_1


@app.cell
def _(openai, port_1, print_highlight):
    _client = openai.Client(base_url=f'http://127.0.0.1:{port_1}/v1', api_key='None')
    _response = _client.chat.completions.create(model='meta-llama/Llama-2-7b-chat-hf', messages=[{'role': 'user', 'content': 'List 3 countries and their capitals.'}], temperature=0, max_tokens=64)
    print_highlight(f'Response: {_response}')
    return


@app.cell
def _(server_process_1, terminate_process):
    terminate_process(server_process_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EAGLE-2 Decoding via Frequency-Ranked Speculative Sampling

    By employing a truncated high-frequency token vocabulary in the draft model, Eagle speculative decoding reduces `lm_head` computational overhead while accelerating the pipeline without quality degradation. For more details, checkout [the paper](https://arxiv.org/pdf/arXiv:2502.14856).

    In our implementation, set `--speculative-token-map` to enable the optimization. You can get the high-frequency token in FR-Spec from [this model](https://huggingface.co/thunlp/LLaMA3-Instruct-8B-FR-Spec). Or you can obtain high-frequency token by directly downloading these token from [this repo](https://github.com/thunlp/FR-Spec/tree/main?tab=readme-ov-file#prepare-fr-spec-vocabulary-subset).

    Thanks for the contribution from [Weilin Zhao](https://github.com/Achazwl) and [Zhousx](https://github.com/Zhou-sx).
    """)
    return


@app.cell
def _(launch_server_cmd, wait_for_server):
    (server_process_2, port_2) = launch_server_cmd('\npython3 -m sglang.launch_server --model meta-llama/Meta-Llama-3-8B-Instruct --speculative-algorithm EAGLE     --speculative-draft-model-path lmsys/sglang-EAGLE-LLaMA3-Instruct-8B --speculative-num-steps 5     --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 --speculative-token-map thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt     --mem-fraction 0.7 --cuda-graph-max-bs 2 --dtype float16  --log-level warning\n')
    wait_for_server(f'http://localhost:{port_2}')
    return port_2, server_process_2


@app.cell
def _(openai, port_2, print_highlight):
    _client = openai.Client(base_url=f'http://127.0.0.1:{port_2}/v1', api_key='None')
    _response = _client.chat.completions.create(model='meta-llama/Meta-Llama-3-8B-Instruct', messages=[{'role': 'user', 'content': 'List 3 countries and their capitals.'}], temperature=0, max_tokens=64)
    print_highlight(f'Response: {_response}')
    return


@app.cell
def _(server_process_2, terminate_process):
    terminate_process(server_process_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EAGLE-3 Decoding

    You can enable EAGLE-3 decoding by setting `--speculative-algorithm EAGLE3` and choosing an appropriate model.
    """)
    return


@app.cell
def _(launch_server_cmd, wait_for_server):
    (server_process_3, port_3) = launch_server_cmd('\npython3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct  --speculative-algorithm EAGLE3     --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B --speculative-num-steps 5         --speculative-eagle-topk 8 --speculative-num-draft-tokens 32 --mem-fraction 0.6         --cuda-graph-max-bs 2 --dtype float16 --log-level warning\n')
    wait_for_server(f'http://localhost:{port_3}')
    return port_3, server_process_3


@app.cell
def _(openai, port_3, print_highlight):
    _client = openai.Client(base_url=f'http://127.0.0.1:{port_3}/v1', api_key='None')
    _response = _client.chat.completions.create(model='meta-llama/Meta-Llama-3.1-8B-Instruct', messages=[{'role': 'user', 'content': 'List 3 countries and their capitals.'}], temperature=0, max_tokens=64)
    print_highlight(f'Response: {_response}')
    return


@app.cell
def _(server_process_3, terminate_process):
    terminate_process(server_process_3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi Token Prediction

    We support [MTP(Multi-Token Prediction)](https://arxiv.org/pdf/2404.19737) in SGLang by using speculative decoding. We use Xiaomi/MiMo-7B-RL model as example here (deepseek mtp usage refer to [deepseek doc](../basic_usage/deepseek.md#multi-token-prediction))

    ⚠️ **Note**: The MiMo-7B-RL model requires approximately 3.7 GB of disk space. Ensure you have sufficient free space before running this example. If you encounter "No space left on device" errors, consider:
    - Freeing up disk space
    - Using a smaller model for testing
    - Skipping this example if disk space is limited
    """)
    return


@app.cell
def _(launch_server_cmd, wait_for_server):
    (server_process_4, port_4) = launch_server_cmd('\n    python3 -m sglang.launch_server --model-path XiaomiMiMo/MiMo-7B-RL --host 0.0.0.0 --trust-remote-code     --speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2     --mem-fraction 0.5 --log-level warning\n')
    wait_for_server(f'http://localhost:{port_4}')
    return port_4, server_process_4


@app.cell
def _(port_4, print_highlight):
    import requests
    url = f'http://localhost:{port_4}/v1/chat/completions'
    data = {'model': 'XiaomiMiMo/MiMo-7B-RL', 'messages': [{'role': 'user', 'content': 'What is the capital of France?'}]}
    _response = requests.post(url, json=data)
    print_highlight(_response.json())
    return


@app.cell
def _(server_process_4, terminate_process):
    terminate_process(server_process_4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    EAGLE process is as follows:

    - Within EAGLE the draft model predicts the next feature vector, i.e. the last hidden state of the original LLM, using the feature sequence $(f_1, ..., f_k)$ and the token sequence $(t_2, ..., t_{k+1})$.
    - The next token is then sampled from $p_{k+2}=\text{LMHead}(f_{k+1})$. Afterwards, the two sequences are extended in a tree style—branching out multiple potential continuations, with the branching factor per step controlled by the `speculative_eagle_topk` parameter—to ensure a more coherent connection of context, and are given as input again.
    - EAGLE-2 additionally uses the draft model to evaluate how probable certain branches in the draft tree are, dynamically stopping the expansion of unlikely branches. After the expansion phase, reranking is employed to select only the top `speculative_num_draft_tokens` final nodes as draft tokens.
    - EAGLE-3 removes the feature prediction objective, incorporates low and mid-layer features, and is trained in an on-policy manner.

    This enhances drafting accuracy by operating on the features instead of tokens for more regular inputs and passing the tokens from the next timestep additionally to minimize randomness effects from sampling. Furthermore the dynamic adjustment of the draft tree and selection of reranked final nodes increases acceptance rate of draft tokens further. For more details see [EAGLE-2](https://arxiv.org/abs/2406.16858) and [EAGLE-3](https://arxiv.org/abs/2503.01840) paper.


    For guidance how to train your own EAGLE model please see the [EAGLE repo](https://github.com/SafeAILab/EAGLE/tree/main?tab=readme-ov-file#train).
    """)
    return


if __name__ == "__main__":
    app.run()
