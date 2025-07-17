import os
import numpy as np
import struct
import select

from .transport_abc import *

__ALL__ = ["NamedPipeClient", "NamedPipeServer"]

class NamedPipeTransport(Transport):

    def __init__(self, path: str, debug: bool, is_server: bool):
        super().__init__(path, debug=debug, is_server=is_server)
        self.pipes = {}
        self.open_fds = {}

    def _init_pipes(self, slot_id: int):
        c2s = os.path.join(self.path, f"c2s_{slot_id}")  # client → server
        s2c = os.path.join(self.path, f"s2c_{slot_id}")  # server → client
        for path in (c2s, s2c):
            if not os.path.exists(path):
                os.mkfifo(path)
        self.pipes[slot_id] = {"c2s": c2s, "s2c": s2c}
        fd_r_c2s = os.open(c2s, os.O_RDONLY | os.O_NONBLOCK)
        fd_r_s2c = os.open(s2c, os.O_RDONLY | os.O_NONBLOCK)
        try:
            fd_w_s2c = os.open(s2c, os.O_WRONLY | os.O_NONBLOCK)
        except OSError:
            self._debug(f"Warning: Unable to open write pipe {s2c}, non-blocking mode may not be supported.")
            fd_w_s2c = None
        try:
            fd_w_c2s = os.open(c2s, os.O_WRONLY | os.O_NONBLOCK)
        except OSError:
            self._debug(f"Warning: Unable to open write pipe {c2s}, non-blocking mode may not be supported.")
            fd_w_c2s = None
        self.open_fds[slot_id] = {
            "fd_r_c2s": fd_r_c2s,
            "fd_r_s2c": fd_r_s2c,
            "fd_w_c2s": fd_w_c2s,
            "fd_w_s2c": fd_w_s2c,
        }
        self._debug(f"Pre-opened pipe FDs for slot {slot_id}")

    def _get_pipe_path(self, slot_id: int, direction: str) -> str:
        self._validate(slot_id)
        if direction not in ("read", "write"):
            raise ValueError(f"Invalid direction: {direction}, must be 'read' or 'write'")
        if self.is_server:
            return self.pipes[slot_id]["c2s"] if direction == "read" else self.pipes[slot_id]["s2c"]
        return self.pipes[slot_id]["s2c"] if direction == "read" else self.pipes[slot_id]["c2s"]

    def is_ready_slot(self, slot_id: int) -> bool:
        path = self._get_pipe_path(slot_id, direction="read")
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            rlist, _, _ = select.select([fd], [], [], 0)
            os.close(fd)
            return bool(rlist)
        except OSError:
            return False

    def read_tensor_slot(self, slot_id: int) -> np.ndarray:
        path = self._get_pipe_path(slot_id, direction="read")
        self._debug(f"will read tensor from slot {slot_id} at {path}")
        with open(path, "rb") as f:
            header = f.read(12)
            if len(header) < 12:
                raise IOError("Incomplete header received.")
            dtype_enum, ndim, size = struct.unpack("iii", header)

            shape_data = f.read(4 * ndim)
            if len(shape_data) < 4 * ndim:
                raise IOError("Incomplete shape data.")
            shape = struct.unpack("i" * ndim, shape_data)

            dtype = self._decode_dtype(dtype_enum)
            data_bytes = f.read(size * dtype.itemsize)
            if len(data_bytes) < size * dtype.itemsize:
                raise IOError("Incomplete tensor data.")
            self._debug(f"read tensor of shape {shape} with dtype {dtype} from slot {slot_id}")
            return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)

    def write_tensor_slot(self, slot_id: int, tensor: np.ndarray | list[np.ndarray]) -> None:
        if isinstance(tensor, list):
            if len(tensor) != 1:
                raise ValueError("NamedPipeTransport only supports single tensor writes.")
            tensor = tensor[0]
        path = self._get_pipe_path(slot_id, direction="write")
        self._debug(f"will write tensor of shape {tensor.shape} to slot {slot_id} at {path}")
        dtype_enum = self._encode_dtype(tensor.dtype)
        ndim = tensor.ndim
        size = tensor.size
        shape = tensor.shape
        self._debug(f"writing tensor with dtype {tensor.dtype}, ndim {ndim}, size {size}, shape {shape}")
        with open(path, "wb") as f:
            header = struct.pack("iii", dtype_enum, ndim, size)
            shape_data = struct.pack("i" * ndim, *shape)
            f.write(header)
            f.write(shape_data)
            f.write(tensor.tobytes())
            self._debug(f"wrote tensor of shape {tensor.shape} to slot {slot_id}")

    def close(self):
        for fds in self.open_fds.values():
            for fd in fds.values():
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
        self.open_fds.clear()
        if self.is_server:
            self._unlink()

    def _unlink(self):
        for slot_id in range(self.num_slots):
            for direction in ["c2s", "s2c"]:
                path = self.pipes[slot_id][direction]
                try:
                    os.unlink(path)
                    self._debug(f"Unlinked {path}")
                except FileNotFoundError:
                    pass
        os.rmdir(self.path)

class NamedPipeClient(NamedPipeTransport, ClientMixin):

    def __init__(self, path: str, slot_id: int, debug: bool = True):
        super().__init__(path, debug=debug, is_server=False)
        self.slot_id = slot_id
        self._init_pipes(slot_id)

class NamedPipeServer(NamedPipeTransport):

    def __init__(self, path: str, num_slots: int, debug: bool = True):
        super().__init__(path, debug=debug, is_server=True)
        self.num_slots = num_slots
        os.makedirs(path, exist_ok=True)
        for slot_id in range(num_slots):
            self._init_pipes(slot_id)
