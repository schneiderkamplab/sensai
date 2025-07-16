# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SensAI is a knowledge distillation framework that provides online teacher logits/tokens generation in fully independent side processes. The core functionality extracts and samples logits from teacher models for distillation purposes, with efficient sparse tensor output and multiple transport mechanisms for inter-process communication.

## Key Architecture

The project follows a client-server architecture with three main components:

### 1. TeacherLogitsServer (`sensai/logits_server.py`)
The central component that wraps HuggingFace Transformers models and provides:
- **Input processing**: Accepts `input_ids` tensors directly (shape `[batch_size, seq_len]`)
- **Logits extraction**: Extracts logits for prompt tokens excluding the last position for causal LM
- **Server-side sampling**: All sampling parameters (num_samples, temperature, top-p, top-k) are configured at server startup
- **Sparse tensor output**: Returns sampled logits as sparse COO tensors with shape `[num_samples, seq_len-1, vocab_size]`

### 2. Transport Layer (`sensai/transports/`)
Provides inter-process communication with three implementations:
- **NamedPipeTransport**: Uses UNIX named pipes (FIFOs) for development/moderate performance
- **SharedMemoryTransport**: Uses memory-mapped files for high-performance production scenarios
- **UnixSocketTransport**: Uses Unix domain sockets for specific use cases
- **Transport ABC**: Abstract base class defining the interface (`is_ready`, `read_tensor`, `write_tensor`)

### 3. Client-Server Communication
- **SensAIServer** (`sensai/server.py`): Handles transport-agnostic request processing with polling loop
- **SensAIClient** (`sensai/client.py`): Provides simple `send_tensor()` interface for making requests
- **Collate Functions** (`sensai/utils.py`): PyTorch DataLoader integration with retry logic

## Development Commands

### Testing
```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run specific test files
uv run python -m pytest tests/test_logits_server.py -v
uv run python -m pytest tests/test_server_integration.py -v
uv run python -m pytest tests/test_cli_transports.py -v
uv run python -m pytest tests/test_collate_functions.py -v

# Run specific test
uv run python -m pytest tests/test_logits_server.py::TestTeacherLogitsServer::test_sample_to_sparse_coo_basic -v

# Run tests with output
uv run python -m pytest tests/test_logits_server.py -v -s

# Run individual test files directly
uv run python tests/test_input_ids_server.py
```

### Running the Server
```bash
# Basic server with named pipes (default)
uv run python -m sensai.logits_server --transport named_pipe --num-clients 4

# Standard distillation setup
uv run python -m sensai.logits_server --model gpt2 --num-samples 256 --temperature 1.0

# High-performance shared memory setup
uv run python -m sensai.logits_server --transport shared_memory --shm-path /dev/shm/sensai --max-elems 2000000

# Full configuration example
uv run python -m sensai.logits_server \
    --model gpt2 \
    --device cuda \
    --transport shared_memory \
    --num-samples 512 \
    --temperature 0.8 \
    --top-p 0.9

# Show all available options
uv run python -m sensai.logits_server --help
```

### Building and Distribution
```bash
# Install in development mode
uv pip install -e .

# Install with optional dependencies
uv pip install -e ".[test]"
uv pip install -e ".[dev]"
uv pip install -e ".[all]"

# Build package
uv run python -m build

# Upload to PyPI
uv run python -m twine upload dist/*
```

## Key Implementation Details

### Server-Side Configuration
- All sampling parameters are configured at server startup, not per-request
- Supports temperature scaling (primary for distillation), top-k, and top-p (experimental)
- For distillation, use `temperature` only; top-k/top-p can remove important probability mass

### Request/Response Flow
1. Client sends `input_ids` tensor as numpy array (shape `[batch_size, seq_len]`)
2. Server processes with pre-configured sampling parameters
3. Server returns either:
   - Single flattened dense tensor (from sparse COO) when `num_samples=None`
   - List of `[indices, values]` numpy arrays when sampling is enabled
4. Client reshapes to `[num_samples, seq_len-1, vocab_size]` for sampling mode

### Transport Selection
- **Named Pipes**: Simple setup, automatic cleanup, filesystem-dependent
- **Shared Memory**: High performance, zero-copy, requires manual cleanup
- **Unix Sockets**: Alternative transport for specific use cases
- All transports support multiple client slots and role-based access control

### Collate Function Integration
- `wrap_collate_function()` integrates with PyTorch DataLoader
- Includes retry logic with configurable attempts and delays
- Gracefully handles server failures by continuing without teacher logits
- Supports both numpy arrays and torch tensors

## Testing Strategy

The test suite covers:
- **Core functionality**: Logits extraction, sampling, sparse tensor creation
- **Transport compatibility**: Named pipes, shared memory, and Unix sockets
- **CLI functionality**: Command-line argument parsing and transport selection
- **Server integration**: Client-server communication and request processing
- **Collate functions**: DataLoader integration with retry logic and error handling
- **Edge cases**: Empty inputs, single tokens, various batch sizes
- **Error handling**: Server failures, invalid inputs, retry mechanisms

## Project Structure

```
sensai/
├── __init__.py              # Main package exports
├── logits_server.py         # Core TeacherLogitsServer class
├── server.py                # SensAIServer (transport-agnostic)
├── client.py                # SensAIClient (simple interface)
├── utils.py                 # PyTorch DataLoader integration (wrap_collate_function)
└── transports/              # Inter-process communication
    ├── __init__.py          # Transport exports
    ├── transport_abc.py     # Abstract base class
    ├── named_pipe.py        # UNIX named pipes implementation
    ├── shared_memory.py     # Memory-mapped files implementation
    └── unix_socket.py       # Unix domain sockets implementation

tests/                       # Comprehensive test suite
examples/                    # Usage examples
├── distillation/            # Complete distillation workflow
└── double/                  # Simple client-server communication
```

## Dependencies

- Python >= 3.12
- Core: numpy, torch, transformers, click, datasets
- Optional: pytest (testing), build/twine (distribution)

## Important Notes

- The server expects `input_ids` tensors, not text strings (simplified from earlier versions)
- For causal language models, output is `seq_len-1` because last position logits aren't used
- Sparse tensors use COO format with indices `[sample_dim, position_dim, vocab_dim]`
- Server-side sampling configuration eliminates per-request parameter overhead
- Transport layer handles numpy array serialization and dtype encoding automatically

## Development Environment

- This project uses uv to manage dependencies. All commands that rely on a python environment should therefore be executed with the prefix `uv run`.