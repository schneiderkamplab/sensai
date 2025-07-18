# Knowledge Distillation Example

This example demonstrates how to use SensAI for knowledge distillation with language models.

## Overview

The example sets up a teacher-student knowledge distillation scenario where:

- **Teacher Model**: `google/gemma-3-1b-it` (serving logits via SensAI server)
- **Student Model**: `google/gemma-3-1b-pt` (training with teacher guidance)
- **Dataset**: WikiText-2 (small subset for demonstration)

## Files

- `student.py`: Student training script that uses the SensAI collate function wrapper
- `run.bash`: Runner script that starts the teacher server and then runs student training
- `README.md`: This documentation file

## Key Features

### Teacher Server
- Runs `google/gemma-3-1b-it` as a teacher model
- Serves logits via named pipe transport
- Configured for single client connection

### Student Training
- Uses the `wrap_collate_function` from `sensai.utils` to automatically request teacher logits
- Implements knowledge distillation loss combining:
  - Task loss (standard cross-entropy)
  - KL divergence loss (student vs teacher logits)
- Includes retry logic for robust teacher-student communication

### Distillation Loss
The implementation uses a standard knowledge distillation loss:
```
total_loss = α * T² * KL(student_logits/T, teacher_logits/T) + (1-α) * CE(student_logits, labels)
```

Where:
- `α = 0.5` (weighting factor)
- `T = 3.0` (temperature)
- `KL` is KL divergence loss
- `CE` is cross-entropy loss

## Usage

### Quick Start
```bash
# From the project root directory
./examples/distillation/run.bash
```

### Manual Steps
1. Start the teacher server:
```bash
uv run python -m sensai.logits_server \
    --model google/gemma-2-1b-it \
    --device cpu \
    --transport named_pipe \
    --num-clients 1
```

2. Wait for the server to load (about 10 seconds)

3. Run the student training:
```bash
uv run python examples/distillation/student.py
```

## Configuration

### Student Training Parameters
- `batch_size = 2`: Small batch size for demo
- `num_samples = 20`: Small dataset for quick execution
- `max_length = 128`: Maximum sequence length
- `learning_rate = 5e-5`: Learning rate
- `temperature = 3.0`: Distillation temperature
- `alpha = 0.5`: Loss weighting factor

### Teacher Server Parameters
- `--model google/gemma-2-1b-it`: Teacher model
- `--device cpu`: Run on CPU (change to `cuda` if available)
- `--transport named_pipe`: Use named pipe transport
- `--num-clients 1`: Single client connection
- `--interval 0.1`: Server polling interval

## Expected Output

The student training will show:
1. Model loading progress
2. Dataset creation
3. SensAI client setup
4. Training progress with loss values
5. Breakdown of task loss and KL loss when teacher logits are available

## Notes

- The example uses the same model for both teacher and student for simplicity
- In practice, you would use a larger teacher model and smaller student model
- The dataset is kept small for demonstration purposes
- Error handling includes retry logic for robust teacher-student communication
- The example automatically handles tensor shape conversions and sparse logits processing

## Troubleshooting

If the student training fails:
1. Ensure the teacher server is running and loaded
2. Check that the named pipe transport is working
3. Verify that models can be loaded (sufficient memory/disk space)
4. Check for any CUDA/device compatibility issues

The `run.bash` script includes automatic cleanup of background processes and temporary files.
