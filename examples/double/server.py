import click
import numpy as np
from sensai.server import SensAIServer
from sensai.transports.named_pipe import NamedPipeTransport
from sensai.transports.shared_memory import SharedMemoryTransport
from sensai.transports.unix_socket import UnixSocketTransport

def process_fn(tensor):
    print(f"[Server] Received tensor: {tensor}")
    tensors = tensor if isinstance(tensor, list) else [tensor]
    tensors = [t * 2 for t in tensors]
    return tensors

@click.command()
@click.option("--transport", type=click.Choice(["shm", "pipe", "socket"]), default="shm", help="Transport type")
@click.option("--path", default="shm.bin", help="Path to shared memory file or pipe")
@click.option("--num-clients", type=int, default=1, help="Number of client slots")
@click.option("--max-nbytes", type=int, default=4*1024**3, help="Maximum number of bytes for each transport slot")
def run_server(transport, path, num_clients, max_nbytes):
    if transport == "shm":
        ctx = SharedMemoryTransport(path, num_clients=num_clients, is_server=True, max_nbytes=max_nbytes)
    elif transport == "pipe":
        ctx = NamedPipeTransport(path, num_clients=num_clients, is_server=True)
    with ctx as t:
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

