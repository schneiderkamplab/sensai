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

1. **Tokenize** input text with padding and attention masks
2. **Extract logits** for prompt tokens (excluding the last position for causal LM)
3. **Apply sampling filters** (temperature, top-k, top-p)
4. **Sample tokens** according to probability distributions
5. **Return results** as sparse COO tensors with shape `[num_samples, seq_len, vocab_size]`

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

#### Advanced Configuration
```bash
# Full configuration example
python -m sensai.logits_server \
    --model gpt2 \
    --device cuda \
    --transport shared_memory \
    --shm-path /dev/shm/teacher_logits \
    --num-clients 4 \
    --max-elems 1000000 \
    --interval 0.001
```

### Command Line Options

- `--model MODEL`: Model name or path to load (default: `google/gemma-3-1b-it`)
- `--device DEVICE`: Device to load model on - `auto`, `cuda`, or `cpu` (default: `auto`)
- `--transport {named_pipe,shared_memory}`: Transport type to use (default: `named_pipe`)
- `--num-clients NUM_CLIENTS`: Number of client slots (default: `4`)
- `--pipe-dir PIPE_DIR`: Directory for named pipes (auto-generated if not specified)
- `--shm-path SHM_PATH`: Path for shared memory file (default: `/tmp/sensai_teacher_shm`)
- `--max-elems MAX_ELEMS`: Maximum elements per tensor for shared memory (default: `1000000`)
- `--interval INTERVAL`: Server polling interval in seconds (default: `0.001`)

### Request Format

The server accepts structured requests via the transport layer:

```python
# Request tensor format: [request_type, num_samples, temperature, top_p, top_k, ...text_bytes]
request_tensor = np.array([
    0,          # request_type: 0=sample_to_sparse_coo, 1=get_prompt_logits
    256,        # num_samples
    0.8,        # temperature
    0.9,        # top_p
    50,         # top_k
    # ... followed by UTF-8 encoded text bytes
], dtype=np.float32)
```

### Client Usage

```python
from sensai.client import SensAIClient
from sensai.transports import NamedPipeTransport
import numpy as np

# Create transport (must match server configuration)
transport = NamedPipeTransport(pipe_dir="/tmp/sensai_teacher_xyz", num_clients=4)
client = SensAIClient(transport, slot_id=0)

# Prepare request
text = "The quick brown fox"
text_bytes = text.encode('utf-8')
request = np.array([0, 256, 0.8, 0.9, 50] + list(text_bytes) + [0]*100, dtype=np.float32)

# Send request and get response
response = client.send_tensor(request)
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