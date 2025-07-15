import click
import numpy as np
from sensai.client import SensAIClient
from sensai.transports.shared_memory import SharedMemoryTransport

@click.command()
@click.option("--transport", type=click.Choice(["shm"]), default="shm", help="Transport type")
@click.option("--path", required=True, help="Path to shared memory file or pipe")
@click.option("--slot-id", type=int, default=0, help="Client slot ID")
@click.option("--max-elems", type=int, default=1024, help="Maximum number of elements per tensor")
@click.option("--dtype", default="float32", help="Tensor dtype (e.g. float32, int32)")
@click.option("--values", default="1.0,2.0,3.0", help="Comma-separated float values for tensor")
def run_client(transport, path, slot_id, max_elems, dtype, values):
    dtype = np.dtype(dtype)
    tensor = np.array([float(x) for x in values.split(",")], dtype=dtype)
    if tensor.size > max_elems:
        raise ValueError(f"Tensor size {tensor.size} exceeds max allowed {max_elems}")
    if transport == "shm":
        with SharedMemoryTransport(path, num_clients=slot_id+1, max_elems=max_elems, max_dtype=dtype) as t:
            client = SensAIClient(t, slot_id=slot_id)
            print(f"[Client] Sending tensor: {tensor}")
            result = client.send_tensor(tensor)
            print(f"[Client] Received result: {result}")

if __name__ == "__main__":
    run_client()
