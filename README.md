# SensAI - Knowledge Distillation Server

A knowledge distillation project that focuses on extracting and sampling logits from teacher models for distillation purposes. The core functionality is implemented in a `TeacherLogitsServer` class that wraps HuggingFace Transformers models.

## Features

- **Logits extraction**: Extract logits from prompt tokens without generation using efficient batching
- **Probabilistic sampling**: Sample tokens according to probability distributions with temperature, top-p, and top-k controls
- **Sparse tensor output**: Returns sampled logits as sparse COO tensors for efficient storage
- **Token decoding**: Convert sampled tokens back to text
- **Multiple transport options**: Named pipes and shared memory for inter-process communication
- **Command line interface**: Easy server setup with configurable parameters

## Architecture

The project centers around the `TeacherLogitsServer` class which provides:

1. **Accept input_ids** tensors directly from clients with shape `[batch_size, seq_len]`
2. **Extract logits** for prompt tokens (excluding the last position for causal LM)
3. **Apply server-side sampling** using configured temperature, top-k, and top-p parameters
4. **Sample tokens** according to probability distributions with nucleus filtering
5. **Return results** as sparse COO tensors with shape `[num_samples, seq_len-1, vocab_size]`

The server is configured once with sampling parameters, making client requests simple and efficient.

## Installation

```bash
pip install torch>=2.7.1 transformers>=4.53.2 numpy>=2.3.1
```

## Usage

### Running the Server

The server can be run with different transport mechanisms:

#### Named Pipe Transport (Default)
```bash
# Basic usage with named pipes
python -m sensai.logits_server --transport named_pipe --num-clients 4

# Custom pipe directory
python -m sensai.logits_server --transport named_pipe --pipe-dir /tmp/my_pipes --num-clients 2
```

#### Shared Memory Transport
```bash
# Basic shared memory usage
python -m sensai.logits_server --transport shared_memory --shm-path /tmp/my_shm --max-elems 2000000

# High-performance setup for large tensors
python -m sensai.logits_server --transport shared_memory --shm-path /dev/shm/sensai --max-elems 5000000 --num-clients 8
```

#### Model and Device Configuration
```bash
# Custom model and device
python -m sensai.logits_server --model gpt2 --device cpu --transport named_pipe

# GPU with specific model
python -m sensai.logits_server --model google/gemma-3-1b-it --device cuda --transport shared_memory

# Auto device selection (default)
python -m sensai.logits_server --model microsoft/DialoGPT-medium --device auto
```

#### Knowledge Distillation Configuration
```bash
# Standard distillation setup (recommended)
python -m sensai.logits_server \
    --model gpt2 \
    --device cuda \
    --num-samples 256 \
    --temperature 1.0

# High-sampling distillation for better student training
python -m sensai.logits_server \
    --model google/gemma-3-1b-it \
    --device cuda \
    --num-samples 512 \
    --temperature 0.8

# Advanced configuration with transport settings
python -m sensai.logits_server \
    --model gpt2 \
    --device cuda \
    --transport shared_memory \
    --shm-path /dev/shm/teacher_logits \
    --num-clients 4 \
    --max-elems 1000000 \
    --num-samples 256 \
    --temperature 1.0 \
    --interval 0.001
```

#### Advanced Sampling (Optional)
```bash
# With nucleus sampling (experimental for distillation)
python -m sensai.logits_server \
    --model gpt2 \
    --num-samples 256 \
    --temperature 1.0 \
    --top-p 0.9

# With top-k sampling (experimental for distillation)
python -m sensai.logits_server \
    --model gpt2 \
    --num-samples 256 \
    --temperature 1.0 \
    --top-k 50
```

### Command Line Options

#### Core Configuration
- `--model MODEL`: Model name or path to load (default: `google/gemma-3-1b-it`)
- `--device DEVICE`: Device to load model on - `auto`, `cuda`, or `cpu` (default: `auto`)

#### Sampling Parameters (Server-Side)
- `--num-samples NUM_SAMPLES`: Number of samples to draw per position (default: `256`)
- `--temperature TEMPERATURE`: Temperature for sampling - higher = more random (default: `1.0`)

#### Advanced Sampling (Optional - Not Typical for Distillation)
- `--top-p TOP_P`: Nucleus sampling threshold (default: `1.0` - disabled)
- `--top-k TOP_K`: Top-k sampling, 0 = disabled (default: `0` - disabled)

> **Note**: Top-k and nucleus sampling are experimental for distillation. Most knowledge distillation scenarios work best with `temperature` control only, as these filters can remove important probability mass that students need to learn from.

#### Transport Configuration
- `--transport {named_pipe,shared_memory}`: Transport type to use (default: `named_pipe`)
- `--num-clients NUM_CLIENTS`: Number of client slots (default: `4`)
- `--pipe-dir PIPE_DIR`: Directory for named pipes (auto-generated if not specified)
- `--shm-path SHM_PATH`: Path for shared memory file (default: `/tmp/sensai_teacher_shm`)
- `--max-elems MAX_ELEMS`: Maximum elements per tensor for shared memory (default: `1000000`)
- `--interval INTERVAL`: Server polling interval in seconds (default: `0.001`)

### Request Format

The server accepts `input_ids` tensors directly (sampling parameters are configured server-side):

```python
import torch
import numpy as np

# Tokenize text to get input_ids
text = "The quick brown fox jumps over"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

# Send as numpy array with shape [batch_size, seq_len]
request_tensor = input_ids.numpy()  # Shape: [1, seq_len]

# For batch processing, stack multiple sequences
batch_texts = ["Text one", "Text two", "Text three"]
batch_input_ids = tokenizer(batch_texts, return_tensors="pt", padding=True)["input_ids"]
batch_request = batch_input_ids.numpy()  # Shape: [3, seq_len]
```

### Client Usage

```python
from sensai.client import SensAIClient
from sensai.transports import NamedPipeTransport
from transformers import AutoTokenizer
import torch
import numpy as np

# Initialize tokenizer (must match server model)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create transport (must match server configuration)
transport = NamedPipeTransport(pipe_dir="/tmp/sensai_teacher_xyz", num_clients=4)
client = SensAIClient(transport, slot_id=0)

# Prepare request with input_ids
text = "The quick brown fox"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
request = input_ids.numpy()  # Shape: [1, seq_len]

# Send request and get sparse tensor response
response = client.send_tensor(request)

# Response is flattened dense tensor - reshape to [num_samples, seq_len-1, vocab_size]
# (Note: seq_len-1 because we predict next tokens for causal LM)
vocab_size = 50257  # GPT-2 vocab size
num_samples = 256   # Server configuration
seq_len = input_ids.shape[1] - 1
response_tensor = response.reshape(num_samples, seq_len, vocab_size)

print(f"Received logits shape: {response_tensor.shape}")
print(f"Teacher sampled {num_samples} tokens for {seq_len} positions")
```

## Development

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_logits_server.py -v
python -m pytest tests/test_server_integration.py -v
python -m pytest tests/test_cli_transports.py -v

# Run specific test
python -m pytest tests/test_logits_server.py::TestTeacherLogitsServer::test_sample_to_sparse_coo_basic -v
```

### Running Tests Directly

```bash
# Run basic functionality tests
python tests/test_logits_server.py

# Run the main server example
python -m sensai.logits_server
```

## Transport Comparison

### Named Pipe Transport
- **Pros**: Simple setup, works on all Unix systems, automatic cleanup
- **Cons**: Slower than shared memory, file system dependent
- **Use case**: Development, moderate performance requirements

### Shared Memory Transport
- **Pros**: High performance, zero-copy transfers, memory-mapped
- **Cons**: Requires manual cleanup, platform-specific paths
- **Use case**: Production, high-throughput scenarios

## Dependencies

- Python >= 3.12
- PyTorch >= 2.7.1
- Transformers >= 4.53.2
- NumPy >= 2.3.1
- pytest >= 8.4.1 (for testing)

## License

[Add your license information here]