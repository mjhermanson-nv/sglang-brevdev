# SGLang Notebooks for Marimo

[![ Click here to deploy.](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy?launchableID=env-35Ob8Jbi077TirDLjC2Yfqo72uh)

Interactive SGLang tutorials converted to **Marimo** notebooks. Learn how to use SGLang for high-performance LLM serving with these step-by-step guides.

## What is SGLang?

[SGLang](https://github.com/sgl-project/sglang) is a fast serving system for large language models (LLMs) that provides:

- 🚀 **High Performance** - Optimized inference with RadixAttention, PagedAttention, and other optimizations
- 🔌 **OpenAI-Compatible APIs** - Drop-in replacement for OpenAI API endpoints
- 🎯 **Flexible** - Support for chat completions, vision models, embeddings, and more
- ⚡ **Efficient** - Advanced batching, continuous batching, and memory management

## Why SGLang + NVIDIA Technologies?

SGLang combined with NVIDIA's GPU computing platform creates a powerful solution for LLM serving that delivers unmatched performance and scalability:

### 🎯 **Hardware-Optimized Performance**
- **NVIDIA GPU Acceleration**: Leverages CUDA cores and Tensor Cores across NVIDIA's GPU lineup (A100, H100, L40S, A10, L4) for maximum throughput
- **FlashInfer Backend**: Optimized attention kernels specifically tuned for NVIDIA GPUs (sm75+), delivering 29-69% lower latency and 2-3x faster inference compared to generic implementations. FlashInfer is SGLang's default backend for production deployments, but requires CUDA compiler (nvcc) for JIT compilation
- **Triton Backend**: All notebooks use NVIDIA's Triton compiler backend for reliable, portable inference that doesn't require CUDA toolkit installation. While Triton may have slightly lower peak performance than FlashInfer, it provides excellent performance and is ideal for development and environments where FlashInfer installation is challenging
- **Memory Efficiency**: PagedAttention and RadixAttention algorithms maximize GPU memory utilization, enabling larger models and higher batch sizes on NVIDIA hardware

### ⚡ **Enterprise-Grade Scalability**
- **Multi-GPU Support**: Seamlessly scales across multiple NVIDIA GPUs for handling high-throughput production workloads
- **Continuous Batching**: Dynamic request batching optimized for NVIDIA's parallel processing architecture, maximizing GPU utilization
- **Low Latency**: CUDA-accelerated inference pipelines minimize latency for real-time applications

## 📚 Notebook Tutorials

Follow these notebooks in order to learn SGLang from basics to advanced usage:

### 1. [Sending Requests](01_send_request.py)
**Start here!** Learn the fundamentals of launching an SGLang server and sending requests.

- Launch an SGLang server
- Send basic generation requests
- Stream responses in real-time
- Understand the core API structure

### 2. [OpenAI APIs - Completions](02_openai_api_completions.py)
Use SGLang as a drop-in replacement for OpenAI's completions API.

- Chat completions API (`chat/completions`)
- Text completions API (`completions`)
- Parameter configuration (temperature, top_p, stop sequences, etc.)
- Structured outputs (JSON, Regex, EBNF)
- LoRA adapter support

### 3. [OpenAI APIs - Vision](03_openai_api_vision.py)
Work with vision-language models through OpenAI-compatible APIs.

- Vision model support
- Image input handling
- Multi-modal completions
- Vision-specific parameters

### 4. [OpenAI APIs - Embedding](04_openai_api_embeddings.py)
Generate embeddings using SGLang's embedding models.

- Embedding model support
- Text-to-embedding conversion
- Batch embedding generation
- Embedding API usage patterns

### 5. [Offline Engine API](05_offline_engine_api.py)
Use SGLang's offline engine for batch processing and offline inference.

- Offline engine setup
- Batch processing workflows
- Non-streaming inference
- Performance optimization

### 6. [SGLang Native APIs](06_native_api.py)
Explore SGLang's native Python APIs for advanced use cases.

- Native API structure
- Direct model access
- Custom inference patterns
- Advanced features

### 7. [Structured Outputs](07_structured_outputs.py)
Generate structured outputs with JSON schema, regex, or EBNF constraints.

- JSON schema constraints
- Regular expression patterns
- EBNF grammar constraints
- Grammar backend options (XGrammar, Outlines, Llguidance)
- Guaranteed format compliance

### 8. [Structured Outputs for Reasoning Models](08_structured_outputs_for_reasoning_models.py)
Use structured outputs with reasoning models that use special reasoning tokens.

- Reasoning model support (DeepSeek R1, QwQ)
- Free-form reasoning sections
- Grammar constraints on structured output
- Reasoning parser configuration

### 9. [LoRA Serving](09_lora.py)
Serve models with LoRA (Low-Rank Adaptation) adapters for fine-tuned variants.

- LoRA adapter loading
- Multiple adapter support
- Dynamic adapter switching
- Fine-tuned model serving

### 10. [Speculative Decoding](10_speculative_decoding.py)
Accelerate inference using EAGLE-based speculative decoding.

- EAGLE-2 and EAGLE-3 support
- Throughput improvements
- Speculative decoding setup
- Performance optimization

### 11. [Tool Parser](11_tool_parser.py)
Implement function calling and tool usage with OpenAI-compatible APIs.

- Function calling API
- Tool definition and parsing
- Multi-tool support
- Tool execution workflows

### 12. [Reasoning Parser](12_separate_reasoning.py)
Parse and separate reasoning content from normal output for reasoning models.

- DeepSeek R1/V3 support
- Qwen3 reasoning models
- Kimi model support
- Reasoning content extraction

## 🚀 Getting Started

### Prerequisites

- Marimo running (access at `http://localhost:8080`)
- SGLang installed (`pip install sglang`)
- GPU with CUDA support (recommended)

### Running the Notebooks

1. **Open Marimo** - Navigate to `http://localhost:8080` in your browser
2. **Open a notebook** - Click on any numbered notebook (e.g., `01_send_request.py`)
3. **Follow along** - Execute cells in order and read the documentation
4. **Experiment** - Modify parameters and see results in real-time

### Notebook Order

These notebooks are numbered to match the [SGLang documentation](https://docs.sglang.ai/basic_usage/) order:

```
01_send_request.py                              → Start here
02_openai_api_completions.py                    → OpenAI-compatible APIs
03_openai_api_vision.py                         → Vision models
04_openai_api_embeddings.py                     → Embedding models
05_offline_engine_api.py                        → Offline processing
06_native_api.py                                → Advanced native APIs
07_structured_outputs.py                        → Structured outputs
08_structured_outputs_for_reasoning_models.py  → Structured outputs for reasoning
09_lora.py                                      → LoRA adapter serving
10_speculative_decoding.py                      → Speculative decoding
11_tool_parser.py                               → Function calling / tools
12_separate_reasoning.py                        → Reasoning parser
```

## 📖 What You'll Learn

### Core Concepts
- **Server Launch** - How to start and configure SGLang servers
- **Request Handling** - Sending requests and processing responses
- **Streaming** - Real-time response streaming
- **Batching** - Efficient batch processing

### API Usage
- **OpenAI Compatibility** - Using SGLang as OpenAI replacement
- **Chat Completions** - Conversational AI workflows
- **Text Completions** - Traditional text generation
- **Vision APIs** - Multi-modal model support
- **Embeddings** - Vector generation for RAG and search

### Advanced Features
- **Structured Outputs** - JSON, Regex, EBNF constraints with grammar backends
- **Reasoning Models** - Support for DeepSeek R1/V3, Qwen3, Kimi reasoning models
- **LoRA Adapters** - Fine-tuned model adapters and dynamic switching
- **Speculative Decoding** - EAGLE-2/EAGLE-3 for accelerated inference
- **Function Calling** - Tool parser and OpenAI-compatible function calling
- **Reasoning Parsing** - Separate reasoning content from structured output
- **Offline Processing** - Batch inference workflows
- **Native APIs** - Direct model access

## 🔧 Configuration

### Server Launch

Each notebook demonstrates launching an SGLang server. Common options include:

```python
# Basic server launch (uses default backend - FlashInfer if available, falls back to Triton)
python -m sglang.launch_server \
    --model-path <model-name> \
    --port <port-number>

# With FlashInfer backend (recommended for production on sm75+ GPUs with CUDA toolkit)
python -m sglang.launch_server \
    --model-path <model-name> \
    --attention-backend flashinfer

# With Triton backend (used in all notebooks - reliable, no CUDA toolkit required)
python -m sglang.launch_server \
    --model-path <model-name> \
    --attention-backend triton

# With LoRA adapters
python -m sglang.launch_server \
    --model-path <model-name> \
    --enable-lora \
    --lora-paths adapter_a=/path/to/adapter
```

**Note**: All notebooks use `--attention-backend triton` for reliability and portability. For production deployments with CUDA toolkit installed, consider using `--attention-backend flashinfer` for optimal performance.

### Model Selection

SGLang supports many models. Examples in notebooks use:
- `Qwen/Qwen2.5-0.5B-Instruct` - Small, fast model for testing
- Other HuggingFace models work similarly

## 📚 Additional Resources

- **SGLang Documentation**: https://docs.sglang.ai
- **SGLang GitHub**: https://github.com/sgl-project/sglang
- **SGLang Paper**: [RadixAttention: SGLang](https://arxiv.org/abs/2402.14862)
- **Marimo Documentation**: https://docs.marimo.io

## 🛠️ Troubleshooting

### Server Won't Start
- Check GPU availability: `nvidia-smi`
- Verify CUDA installation: `nvcc --version`
- Ensure model path is correct
- Check port availability

### Import Errors
- Install SGLang: `pip install sglang --pre`
- Install FlashInfer: `pip install flashinfer-python`
- Verify Python version (3.10+)

### Performance Issues
- **Backend Selection**: 
  - Use FlashInfer backend (`--attention-backend flashinfer`) for production on sm75+ GPUs (T4, A10, A100, L4, L40S, H100) when CUDA toolkit is available - delivers 29-69% lower latency
  - Use Triton backend (`--attention-backend triton`) for development or when FlashInfer installation fails - provides excellent performance without requiring nvcc
- Adjust batch size based on GPU memory
- Check GPU utilization: `nvidia-smi`

## 📝 Notes

- These notebooks are converted from the official [SGLang documentation](https://docs.sglang.ai)
- Original Jupyter notebooks available at: https://github.com/sgl-project/sgl-project.github.io
- Notebooks are designed to be run sequentially for best learning experience
- All notebooks use Marimo's reactive execution model

## 🤝 Contributing

Found an issue or want to improve these notebooks? Contributions welcome!

1. Fork the repository
2. Make your changes
3. Submit a pull request

## 📄 License

These notebooks follow the same license as SGLang and Marimo.

---

**Happy Learning! 🎉**

Start with `01_send_request.py` and work through the tutorials in order.
