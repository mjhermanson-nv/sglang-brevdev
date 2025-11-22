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
    # Offline Engine API

    SGLang provides a direct inference engine without the need for an HTTP server, especially for use cases where additional HTTP server adds unnecessary complexity or overhead. Here are two general use cases:

    - Offline Batch Inference
    - Custom Server on Top of the Engine

    This document focuses on the offline batch inference, demonstrating four different inference modes:

    - Non-streaming synchronous generation
    - Streaming synchronous generation
    - Non-streaming asynchronous generation
    - Streaming asynchronous generation

    Additionally, you can easily build a custom server on top of the SGLang offline engine. A detailed example working in a python script can be found in [custom_server](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/custom_server.py).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nest Asyncio
    Note that if you want to use **Offline Engine** in ipython or some other nested loop code, you need to add the following code:
    ```python
    import nest_asyncio

    nest_asyncio.apply()

    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Advanced Usage

    The engine supports [vlm inference](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/offline_batch_inference_vlm.py) as well as [extracting hidden states](https://github.com/sgl-project/sglang/blob/main/examples/runtime/hidden_states).

    Please see [the examples](https://github.com/sgl-project/sglang/tree/main/examples/runtime/engine) for further use cases.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Offline Batch Inference

    SGLang offline engine supports batch inference with efficient scheduling.
    """)
    return


@app.cell
def _():
    # launch the offline engine
    import asyncio

    import sglang as sgl
    import sglang.test.doc_patch
    from sglang.utils import async_stream_and_merge, stream_and_merge

    llm = sgl.Engine(model_path="qwen/qwen2.5-0.5b-instruct")
    return async_stream_and_merge, asyncio, llm, stream_and_merge


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Non-streaming Synchronous Generation
    """)
    return


@app.cell
def _(llm):
    _prompts = ['Hello, my name is', 'The president of the United States is', 'The capital of France is', 'The future of AI is']
    _sampling_params = {'temperature': 0.8, 'top_p': 0.95}
    outputs = llm.generate(_prompts, _sampling_params)
    for (_prompt, output) in zip(_prompts, outputs):
        print('===============================')
        print(f"Prompt: {_prompt}\nGenerated text: {output['text']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Streaming Synchronous Generation
    """)
    return


@app.cell
def _(llm, stream_and_merge):
    _prompts = ['Write a short, neutral self-introduction for a fictional character. Hello, my name is', 'Provide a concise factual statement about France’s capital city. The capital of France is', 'Explain possible future trends in artificial intelligence. The future of AI is']
    _sampling_params = {'temperature': 0.2, 'top_p': 0.9}
    print('\n=== Testing synchronous streaming generation with overlap removal ===\n')
    for _prompt in _prompts:
        print(f'Prompt: {_prompt}')
        merged_output = stream_and_merge(llm, _prompt, _sampling_params)
        print('Generated text:', merged_output)
        print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Non-streaming Asynchronous Generation
    """)
    return


@app.cell
def _(asyncio, llm):
    _prompts = ['Write a short, neutral self-introduction for a fictional character. Hello, my name is', 'Provide a concise factual statement about France’s capital city. The capital of France is', 'Explain possible future trends in artificial intelligence. The future of AI is']
    _sampling_params = {'temperature': 0.8, 'top_p': 0.95}
    print('\n=== Testing asynchronous batch generation ===')

    async def _main():
        outputs = await llm.async_generate(_prompts, _sampling_params)
        for (_prompt, output) in zip(_prompts, outputs):
            print(f'\nPrompt: {_prompt}')
            print(f"Generated text: {output['text']}")
    asyncio.run(_main())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Streaming Asynchronous Generation
    """)
    return


@app.cell
def _(async_stream_and_merge, asyncio, llm):
    _prompts = ['Write a short, neutral self-introduction for a fictional character. Hello, my name is', 'Provide a concise factual statement about France’s capital city. The capital of France is', 'Explain possible future trends in artificial intelligence. The future of AI is']
    _sampling_params = {'temperature': 0.8, 'top_p': 0.95}
    print('\n=== Testing asynchronous streaming generation (no repeats) ===')

    async def _main():
        for _prompt in _prompts:
            print(f'\nPrompt: {_prompt}')
            print('Generated text: ', end='', flush=True)
            async for cleaned_chunk in async_stream_and_merge(llm, _prompt, _sampling_params):
                print(cleaned_chunk, end='', flush=True)
            print()
    asyncio.run(_main())  # Replace direct calls to async_generate with our custom overlap-aware version  # New line after each prompt
    return


@app.cell
def _(llm):
    llm.shutdown()
    return


if __name__ == "__main__":
    app.run()
