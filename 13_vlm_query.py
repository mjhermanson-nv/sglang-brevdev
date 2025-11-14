import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Query Vision Language Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Querying Qwen-VL
    """)
    return


@app.cell
def _():
    import nest_asyncio

    nest_asyncio.apply()  # Run this first.

    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    chat_template = "qwen2-vl"
    return chat_template, model_path, nest_asyncio


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

    llm = Engine(
        model_path=model_path, chat_template=chat_template, mem_fraction_static=0.8
    )
    return Engine, llm


@app.cell
def _(conv, image, llm):
    _out = llm.generate(prompt=conv.get_prompt(), image_data=[image])
    print(_out['text'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Query via the offline Engine API, but send precomputed embeddings
    """)
    return


@app.cell
def _(model_path):
    # Compute the image embeddings using Huggingface.

    from transformers import AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    vision = (
        Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path).eval().visual.cuda()
    )
    return AutoProcessor, processor, vision


@app.cell
def _(conv, image, llm, processor, vision):
    _processed_prompt = processor(images=[image], text=conv.get_prompt(), return_tensors='pt')
    _input_ids = _processed_prompt['input_ids'][0].detach().cpu().tolist()
    _precomputed_embeddings = vision(_processed_prompt['pixel_values'].cuda(), _processed_prompt['image_grid_thw'].cuda())
    _mm_item = dict(modality='IMAGE', image_grid_thw=_processed_prompt['image_grid_thw'], precomputed_embeddings=_precomputed_embeddings)
    _out = llm.generate(input_ids=_input_ids, image_data=[_mm_item])
    print(_out['text'])
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
def _(Engine, model_path_1):
    from sglang.test.test_utils import is_in_ci
    if not is_in_ci():
        llm_1 = Engine(model_path=model_path_1, trust_remote_code=True, enable_multimodal=True, mem_fraction_static=0.8, tp_size=4, attention_backend='fa3', context_length=65536)
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
def _(AutoProcessor, is_in_ci, model_path_1):
    if not is_in_ci():
        from transformers import Llama4ForConditionalGeneration  # Compute the image embeddings using Huggingface.
        processor_1 = AutoProcessor.from_pretrained(model_path_1, use_fast=True)
        model = Llama4ForConditionalGeneration.from_pretrained(model_path_1, torch_dtype='auto').eval()
        vision_1 = model.vision_model.cuda()
        multi_modal_projector = model.multi_modal_projector.cuda()
    return multi_modal_projector, processor_1, vision_1


@app.cell
def _(
    conv_1,
    image_1,
    is_in_ci,
    llm_1,
    multi_modal_projector,
    processor_1,
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
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
