import click
import numpy as np
from sensai.server import SensAIServer
from sensai.transports.shared_memory import SharedMemoryTransport

def process_fn(tensor):
    print(f"[Server] Received tensor: {tensor}")
    return tensor * 2

@click.command()
@click.option("--transport", type=click.Choice(["shm"]), default="shm", help="Transport type")
@click.option("--path", default="shm.bin", help="Path to shared memory file or pipe")
@click.option("--num-clients", type=int, default=1, help="Number of client slots")
@click.option("--max-elems", type=int, default=1024, help="Maximum number of elements per tensor")
@click.option("--dtype", default="float32", help="Tensor dtype (e.g. float32, int32)")
def run_server(transport, path, num_clients, max_elems, dtype):
    dtype = np.dtype(dtype)
    if transport == "shm":
        with SharedMemoryTransport(path, num_clients=num_clients, max_elems=max_elems, max_dtype=dtype) as t:
            server = SensAIServer(t)
            print("[Server] Starting server loop")
            try:
                server.run_loop(process_fn)
            except KeyboardInterrupt:
                print("[Server] Shutting down...")
            finally:
                t.unlink()

if __name__ == "__main__":
    run_server()

