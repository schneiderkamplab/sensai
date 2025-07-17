import click
import numpy as np
from sensai.client import SensAIClient
from sensai.transports.named_pipe import NamedPipeClient
from sensai.transports.shared_memory import SharedMemoryClient
import time

@click.command()
@click.option("--transport", type=click.Choice(["shm", "pipe"]), default="shm", help="Transport type")
@click.option("--path", default="shm.bin", help="Path to shared memory file")
@click.option("--slot-id", type=int, default=0, help="Client slot ID")
@click.option("--dtype", default="float32", help="Tensor dtype (e.g. float32, int32)")
def run_client(transport, path, slot_id, dtype):
    dtype = np.dtype(dtype)
    shape = (1024, 1024, 1024)
    tensor = np.ones(shape, dtype=dtype)
    max_elems = tensor.size
    print(f"[Client] Generated tensor of shape {tensor.shape}, dtype={dtype}, size={tensor.nbytes / 1e6:.2f} MB")
    if transport == "shm":
        cls = SharedMemoryClient
    elif transport == "pipe":
        cls = NamedPipeClient
    with cls(path, slot_id=slot_id) as t:
        client = SensAIClient(t)
        for i in range(10):
            print(f"[Client] Iteration {i+1}/10")
            start = time.time()
            result = client.send_tensor(tensor)
            print(f"[Client] Tensor sent, processed, and returned in {time.time() - start:.2f} seconds")
        print(f"[Client] Checking result...")
        if not np.allclose(result, tensor * 2):
            print("[Client] ERROR: Result mismatch.")
        else:
            print("[Client] SUCCESS: Result is correct.")

if __name__ == "__main__":
    run_client()
