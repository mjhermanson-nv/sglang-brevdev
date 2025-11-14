import marimo

__generated_with = "0.17.8"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Query Vision Language Model

    ⚠️ **GPU Memory Warning**: This notebook loads large vision models. If you encounter out-of-memory errors:
    - Close other notebooks that are running SGLang servers
    - Reduce `mem_fraction_static` if other processes are using GPU memory
    - Ensure you have sufficient GPU memory available (recommended: 24GB+ for Qwen-VL, 80GB+ for Llama 4)

    ⚠️ **Important: HuggingFace Authentication Required**

    This notebook uses gated Meta Llama models requiring HuggingFace authentication.

    **To access gated models:**
    1. Visit the model pages and accept the licenses:
       - https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
    2. Generate a token at https://huggingface.co/settings/tokens
       - **Important**: If using a fine-grained token, enable "public gated repositories" permission
       - Or use a classic token (which has this permission by default)
    3. Enter your token in the cell below

    **Token Requirements:**
    - Must have access to the gated models (request access first)
    - Fine-grained tokens need "public gated repositories" permission enabled
    - Classic tokens work automatically
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
    ## Querying Qwen-VL
    """)
    return


@app.cell
def _():
    import gc
    import nest_asyncio

    nest_asyncio.apply()  # Run this first.

    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    chat_template = "qwen2-vl"
    return chat_template, gc, model_path, nest_asyncio


@app.cell
def _(chat_template):
    # Lets create a prompt.

    from io import BytesIO
    import requests
    from PIL import Image

    from sglang.srt.parser.conversation import chat_templates

    image = Image.open(
        BytesIO(
            requests.get(
                "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
            ).content
        )
    )

    conv = chat_templates[chat_template].copy()
    conv.append_message(conv.roles[0], f"What's shown here: {conv.image_token}?")
    conv.append_message(conv.roles[1], "")
    conv.image_data = [image]

    print(conv.get_prompt())
    image
    return BytesIO, Image, chat_templates, conv, image, requests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Query via the offline Engine API
    """)
    return


@app.cell
def _(chat_template, model_path):
    from sglang import Engine
    import torch

    # Check available GPU memory and adjust mem_fraction_static accordingly
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        # Use 40% if >40GB GPU, otherwise use 50% to leave room for other processes
        mem_fraction = 0.4 if total_memory > 40 else 0.5
        print(f"GPU memory: {total_memory:.1f} GB, using mem_fraction_static={mem_fraction}")
    else:
        mem_fraction = 0.8

    llm = Engine(
        model_path=model_path, chat_template=chat_template, mem_fraction_static=mem_fraction
    )
    return Engine, llm, torch


@app.cell
def _(conv, image, llm):
    _out = llm.generate(prompt=conv.get_prompt(), image_data=[image])
    print(_out['text'])
    return


@app.cell
def _(llm):
    # Cleanup: shutdown the engine to free GPU memory
    llm.shutdown()
    print("✓ Qwen-VL engine shutdown")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Query via the offline Engine API, but send precomputed embeddings
    """)
    return


@app.cell
def _(model_path, torch):
    # Compute the image embeddings using Huggingface.

    from transformers import AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

    # Load vision model with memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear cache before loading
    vision = (
        Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path).eval().visual.cuda()
    )
    return AutoProcessor, processor, vision


@app.cell
def _(
    Engine,
    chat_template,
    conv,
    gc,
    image,
    model_path,
    processor,
    torch,
    vision,
):
    # Recreate engine for this section (previous one was shut down)
    if torch.cuda.is_available():
        total_memory_2 = torch.cuda.get_device_properties(0).total_memory / 1e9
        mem_fraction_2 = 0.4 if total_memory_2 > 40 else 0.5
    else:
        mem_fraction_2 = 0.8

    llm_2 = Engine(model_path=model_path, chat_template=chat_template, mem_fraction_static=mem_fraction_2)

    _processed_prompt = processor(images=[image], text=conv.get_prompt(), return_tensors='pt')
    _input_ids = _processed_prompt['input_ids'][0].detach().cpu().tolist()
    _precomputed_embeddings = vision(_processed_prompt['pixel_values'].cuda(), _processed_prompt['image_grid_thw'].cuda())
    _mm_item = dict(modality='IMAGE', image_grid_thw=_processed_prompt['image_grid_thw'], precomputed_embeddings=_precomputed_embeddings)
    _out = llm_2.generate(input_ids=_input_ids, image_data=[_mm_item])
    print(_out['text'])

    # Cleanup
    del vision
    torch.cuda.empty_cache()
    gc.collect()
    llm_2.shutdown()
    print("✓ Vision model and engine cleaned up")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Querying Llama 4 (Vision)
    """)
    return


@app.cell
def _(nest_asyncio):
    nest_asyncio.apply()
    model_path_1 = 'meta-llama/Llama-4-Scout-17B-16E-Instruct'
    chat_template_1 = 'llama-4'  # Run this first.
    return chat_template_1, model_path_1


@app.cell
def _(BytesIO, Image, chat_template_1, chat_templates, requests):
    # Lets create a prompt.
    image_1 = Image.open(BytesIO(requests.get('https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true').content))
    conv_1 = chat_templates[chat_template_1].copy()
    conv_1.append_message(conv_1.roles[0], f"What's shown here: {conv_1.image_token}?")
    conv_1.append_message(conv_1.roles[1], '')
    conv_1.image_data = [image_1]
    print(conv_1.get_prompt())
    print(f'Image size: {image_1.size}')
    image_1
    return conv_1, image_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Query via the offline Engine API
    """)
    return


@app.cell
def _(Engine, model_path_1, torch):
    from sglang.test.test_utils import is_in_ci
    if not is_in_ci():
        # Check GPU count - Llama 4 with tp_size=4 requires 4 GPUs
        # If only 1 GPU available, use tp_size=1 and reduce memory fraction
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus >= 4:
            tp_size = 4
            mem_fraction_llama4 = 0.8
        else:
            tp_size = 1
            # Use less memory fraction when single GPU to leave room
            mem_fraction_llama4 = 0.4 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory / 1e9 > 40 else 0.5
            print(f"⚠️  Only {num_gpus} GPU(s) available. Using tp_size=1 (requires 80GB+ GPU for Llama 4)")

        llm_1 = Engine(
            model_path=model_path_1, 
            trust_remote_code=True, 
            enable_multimodal=True, 
            mem_fraction_static=mem_fraction_llama4, 
            tp_size=tp_size, 
            attention_backend='fa3', 
            context_length=65536
        )
    return is_in_ci, llm_1


@app.cell
def _(conv_1, image_1, is_in_ci, llm_1):
    if not is_in_ci():
        _out = llm_1.generate(prompt=conv_1.get_prompt(), image_data=[image_1])
        print(_out['text'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Query via the offline Engine API, but send precomputed embeddings
    """)
    return


@app.cell
def _(AutoProcessor, is_in_ci, model_path_1, torch):
    if not is_in_ci():
        from transformers import Llama4ForConditionalGeneration  # Compute the image embeddings using Huggingface.
        # Clear cache before loading large model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        processor_1 = AutoProcessor.from_pretrained(model_path_1, use_fast=True)
        model = Llama4ForConditionalGeneration.from_pretrained(model_path_1, torch_dtype='auto').eval()
        vision_1 = model.vision_model.cuda()
        multi_modal_projector = model.multi_modal_projector.cuda()
    return multi_modal_projector, processor_1, vision_1


@app.cell
def _(
    conv_1,
    gc,
    image_1,
    is_in_ci,
    llm_1,
    multi_modal_projector,
    processor_1,
    torch,
    vision_1,
):
    if not is_in_ci():
        _processed_prompt = processor_1(images=[image_1], text=conv_1.get_prompt(), return_tensors='pt')
        print(f"""processed_prompt["pixel_values"].shape={_processed_prompt['pixel_values'].shape!r}""")
        _input_ids = _processed_prompt['input_ids'][0].detach().cpu().tolist()
        image_outputs = vision_1(_processed_prompt['pixel_values'].to('cuda'), output_hidden_states=False)
        image_features = image_outputs.last_hidden_state
        vision_flat = image_features.view(-1, image_features.size(-1))
        _precomputed_embeddings = multi_modal_projector(vision_flat)
        _mm_item = dict(modality='IMAGE', precomputed_embeddings=_precomputed_embeddings)
        _out = llm_1.generate(input_ids=_input_ids, image_data=[_mm_item])
        print(_out['text'])

        # Cleanup
        del vision_1, multi_modal_projector
        torch.cuda.empty_cache()
        gc.collect()
        llm_1.shutdown()
        print("✓ Llama 4 models and engine cleaned up")
    return


if __name__ == "__main__":
    app.run()
