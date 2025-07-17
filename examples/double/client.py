import click
import numpy as np
from sensai.client import SensAIClient
from sensai.transports.named_pipe import NamedPipeClient
from sensai.transports.shared_memory import SharedMemoryClient

@click.command()
@click.option("--transport", type=click.Choice(["shm", "pipe"]), default="shm", help="Transport type")
@click.option("--path", default="shm.bin", help="Path to shared memory file or pipe")
@click.option("--slot-id", type=int, default=0, help="Client slot ID")
@click.option("--values", default="1.0,2.0,3.0", help="Comma-separated float values for tensor")
def run_client(transport, path, slot_id, values):
    tensor = np.array([float(x) for x in values.split(",")], dtype=np.float32)
    if transport == "shm":
        cls = SharedMemoryClient
    elif transport == "pipe":
        cls = NamedPipeClient
    with cls(path, slot_id=slot_id) as t:
        client = SensAIClient(t)
        if t.supports_multi_tensor():
            tensor = [tensor, tensor.astype(np.int64)]
        print(f"[Client] Sending tensor: {tensor}")
        try:
            result = client.send_tensor(tensor)
        except Exception as e:
            print(f"[Client] Error sending tensor: {e}")
            raise e
        finally:
            print("[Client] Finished sending tensor.")
        print(f"[Client] Received result: {result}")

if __name__ == "__main__":
    run_client()
