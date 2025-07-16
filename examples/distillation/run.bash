#!/bin/bash

# Knowledge Distillation Example Runner
# This script starts the teacher server and then runs the student training

set -e  # Exit on any error

echo "=== SensAI Knowledge Distillation Example ==="
echo "Teacher: google/gemma-3-1b-it"
echo "Student: google/gemma-3-1b-pt"
echo ""

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Project root: $PROJECT_ROOT"
echo "Script directory: $SCRIPT_DIR"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ ! -z "$SERVER_PID" ]; then
        echo "Stopping teacher server (PID: $SERVER_PID)"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    
    # Clean up any remaining shared memory files
    rm -f /tmp/sensai_teacher_shm* 2>/dev/null || true
    
    echo "Cleanup completed"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

echo "Starting teacher server..."
echo "Command: uv run python -m sensai.logits_server --model google/gemma-3-1b-it --device cpu --transport shared_memory --num-clients 1"
echo ""

# Start the teacher server in the background
uv run python -m sensai.logits_server \
    --model google/gemma-3-1b-it \
    --device cpu \
    --transport shared_memory \
    --num-clients 1 \
    --max-elems 2_561_412_964 \
    --interval 0.1 &

SERVER_PID=$!
echo "Teacher server started with PID: $SERVER_PID"

# Wait for server to load
echo "Waiting 10 seconds for teacher server to load..."
sleep 10

# Check if server is still running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Teacher server failed to start or crashed"
    exit 1
fi

echo "Teacher server should be ready now"
echo ""

# Start student training
echo "Starting student training..."
echo "Command: uv run python examples/distillation/student.py"
echo ""

# Run the student training
uv run python examples/distillation/student.py

echo ""
echo "Student training completed!"
echo ""
echo "=== Distillation Example Finished ==="