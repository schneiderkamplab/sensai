import os
import numpy as np
import struct
import select

from .transport_abc import Transport

__ALL__ = ["NamedPipeTransport"]

class NamedPipeTransport(Transport):
    def __init__(self, pipe_dir: str, num_clients: int, debug: bool = True):
        self.pipe_dir = pipe_dir
        self._num_clients = num_clients
        self.debug = debug
        os.makedirs(pipe_dir, exist_ok=True)
        self.pipes = {}
        self.open_fds = {}
        for slot_id in range(num_clients):
            c2s = os.path.join(pipe_dir, f"c2s_{slot_id}")  # client → server
            s2c = os.path.join(pipe_dir, f"s2c_{slot_id}")  # server → client
            for path in (c2s, s2c):
                if not os.path.exists(path):
                    os.mkfifo(path)
            self.pipes[slot_id] = {"c2s": c2s, "s2c": s2c}
            fd_r_c2s = os.open(c2s, os.O_RDONLY | os.O_NONBLOCK)
            fd_r_s2c = os.open(s2c, os.O_RDONLY | os.O_NONBLOCK)
            try:
                fd_w_s2c = os.open(s2c, os.O_WRONLY | os.O_NONBLOCK)
            except OSError:
                fd_w_s2c = None
            try:
                fd_w_c2s = os.open(c2s, os.O_WRONLY | os.O_NONBLOCK)
            except OSError:
                fd_w_c2s = None
            self.open_fds[slot_id] = {
                "fd_r_c2s": fd_r_c2s,
                "fd_r_s2c": fd_r_s2c,
                "fd_w_c2s": fd_w_c2s,
                "fd_w_s2c": fd_w_s2c,
            }
            self._debug(f"Pre-opened pipe FDs for slot {slot_id}")

    def _debug(self, msg):
        if self.debug:
            print(f"[NamedPipeTransport] {msg}")

    def _get_pipe_path(self, slot_id: int, role: str, direction: str) -> str:
        if role == "client":
            return self.pipes[slot_id]["s2c"] if direction == "read" else self.pipes[slot_id]["c2s"]
        elif role == "server":
            return self.pipes[slot_id]["c2s"] if direction == "read" else self.pipes[slot_id]["s2c"]
        else:
            raise ValueError(f"Invalid role: {role}")

    def is_ready(self, slot_id: int, role: str) -> bool:
        path = self._get_pipe_path(slot_id, role, direction="read")
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            rlist, _, _ = select.select([fd], [], [], 0)
            os.close(fd)
            return bool(rlist)
        except OSError:
            return False

    def read_tensor(self, slot_id: int, role: str) -> np.ndarray:
        path = self._get_pipe_path(slot_id, role, direction="read")
        self._debug(f"{role} will read tensor from slot {slot_id} at {path}")
        with open(path, "rb") as f:
            header = f.read(12)
            if len(header) < 12:
                raise IOError("Incomplete header received.")
            dtype_enum, ndim, size = struct.unpack("iii", header)

            shape_data = f.read(4 * ndim)
            if len(shape_data) < 4 * ndim:
                raise IOError("Incomplete shape data.")
            shape = struct.unpack("i" * ndim, shape_data)

            dtype = self._resolve_dtype(dtype_enum)
            data_bytes = f.read(size * dtype.itemsize)
            if len(data_bytes) < size * dtype.itemsize:
                raise IOError("Incomplete tensor data.")
            self._debug(f"{role} read tensor of shape {shape} with dtype {dtype} from slot {slot_id}")
            return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)

    def write_tensor(self, slot_id: int, tensor: np.ndarray, role: str) -> None:
        path = self._get_pipe_path(slot_id, role, direction="write")
        self._debug(f"{role} will write tensor of shape {tensor.shape} to slot {slot_id} at {path}")
        dtype_enum = self._encode_dtype(tensor.dtype)
        ndim = tensor.ndim
        size = tensor.size
        shape = tensor.shape
        self._debug(f"{role} writing tensor with dtype {tensor.dtype}, ndim {ndim}, size {size}, shape {shape}")
        with open(path, "wb") as f:
            header = struct.pack("iii", dtype_enum, ndim, size)
            shape_data = struct.pack("i" * ndim, *shape)
            f.write(header)
            f.write(shape_data)
            f.write(tensor.tobytes())
            self._debug(f"{role} wrote tensor of shape {tensor.shape} to slot {slot_id}")

    def close(self):
        for slot_id, fds in self.open_fds.items():
            for fd in fds.values():
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
        self.open_fds.clear()

    def unlink(self):
        for slot_id in range(self._num_clients):
            for role in ["client", "server"]:
                for direction in ["read", "write"]:
                    path = self._get_pipe_path(slot_id, role, direction)
                    try:
                        os.unlink(path)
                        self._debug(f"Unlinked {path}")
                    except FileNotFoundError:
                        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def num_clients(self) -> int:
        return self._num_clients
