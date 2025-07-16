import click
import numpy as np
from sensai.client import SensAIClient
from sensai.transports.named_pipe import NamedPipeTransport
from sensai.transports.shared_memory import SharedMemoryTransport

@click.command()
@click.option("--transport", type=click.Choice(["shm", "pipe"]), default="shm", help="Transport type")
@click.option("--path", default="shm.bin", help="Path to shared memory file or pipe")
@click.option("--slot-id", type=int, default=0, help="Client slot ID")
@click.option("--num-clients", type=int, default=1, help="Number of client slots")
@click.option("--max-elems", type=int, default=1024, help="Maximum number of elements per tensor")
@click.option("--dtype", default="float32", help="Tensor dtype (e.g. float32, int32)")
@click.option("--values", default="1.0,2.0,3.0", help="Comma-separated float values for tensor")
def run_client(transport, path, slot_id, num_clients, max_elems, dtype, values):
    dtype = np.dtype(dtype)
    tensor = np.array([float(x) for x in values.split(",")], dtype=dtype)
    if tensor.size > max_elems:
        raise ValueError(f"Tensor size {tensor.size} exceeds max allowed {max_elems}")
    if transport == "shm":
        ctx = SharedMemoryTransport(path, num_clients=num_clients, max_elems=max_elems, max_dtype=dtype)
    elif transport == "pipe":
        ctx = NamedPipeTransport(path, num_clients=num_clients)    
    with ctx as t:
        client = SensAIClient(t, slot_id=slot_id)
        print(f"[Client] Sending tensor: {tensor}")
        try:
            result = client.send_tensor(tensor)
        except Exception as e:
            print(f"[Client] Error sending tensor: {e}")
            return
        finally:
            print("[Client] Finished sending tensor.")
        print(f"[Client] Received result: {result}")

if __name__ == "__main__":
    run_client()
