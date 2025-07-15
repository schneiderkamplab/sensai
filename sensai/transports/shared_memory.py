import mmap
import numpy as np
import os

from .transport_abc import Transport

__all__ = ["SharedMemoryTransport"]

class SharedMemoryTransport(Transport):

    _ROLE_MAP = {
        "client": 0,
        "server": 1
    }

    def __init__(self, shm_path: str, num_clients: int, max_elems: int, max_dtype: np.dtype, debug: bool = True):
        self.shm_path = shm_path
        self._num_clients = num_clients
        self.max_elems = max_elems
        self.max_dtype = np.dtype(max_dtype)
        self.debug = debug
        self._header_size = 256  # fixed header size for cache alignment
        self._slot_size = self._header_size + max_elems * self.max_dtype.itemsize
        self._fd = os.open(shm_path, os.O_CREAT | os.O_RDWR)
        os.ftruncate(self._fd, self._num_clients * self._slot_size)
        self._mmap = mmap.mmap(self._fd, self._num_clients * self._slot_size)
        self._headers = {}
        self._datas = {}
        self._init(0)

    def _init(self, from_slot):
        self._debug(f"Initializing shared memory transport with {self._num_clients} clients starting from slot {from_slot}")
        for slot_id in range(from_slot, self._num_clients):
            offset = slot_id * self._slot_size
            header_buf = memoryview(self._mmap)[offset : offset + self._header_size]
            data_buf = memoryview(self._mmap)[offset + self._header_size : offset + self._slot_size]
            self._debug(f"Setting up slot {slot_id} with header at {offset} and data at {offset + self._header_size}")
            self._debug(f"Header buffer size: {len(header_buf)}, Data buffer size: {len(data_buf)}")
            self._headers[slot_id] = np.frombuffer(header_buf, dtype=np.int64)
            self._headers[slot_id][0] = self._ROLE_MAP["client"]
            self._datas[slot_id] = data_buf

    def _debug(self, msg):
        if self.debug:
            print(f"[SharedMemoryTransport] {msg}")

    def write_tensor(self, slot_id: int, tensor: np.ndarray, role: str) -> None:
        self._validate(slot_id, role)
        header = self._headers[slot_id]
        self._debug(f"{role} will write tensor of shape {tensor.shape} to slot {slot_id} with header {header[:8]}")
        if tensor.size > self.max_elems:
            raise ValueError(f"Tensor too large: {tensor.size} > {self.max_elems}")
        self._datas[slot_id][:tensor.nbytes] = tensor.tobytes()
        header[1] = self._encode_dtype(tensor.dtype)
        header[2] = tensor.ndim
        header[3] = tensor.nbytes
        header[4:4 + tensor.ndim] = tensor.shape
        header[0] = 1-self._ROLE_MAP[role]
        self._debug(f"{role} wrote tensor of shape {tensor.shape} to slot {slot_id} with header {header[:8]}")

    def read_tensor(self, slot_id: int, role: str) -> np.ndarray | None:
        self._validate(slot_id, role)
        header = self._headers[slot_id]
        if header[0] != self._ROLE_MAP[role]:
            self._debug(f"{role} found no tensor to read in slot {slot_id}")
            return None
        self._debug(f"{role} will read tensor from slot {slot_id} with header {header[:8]}")
        dtype = self._resolve_dtype(header[1])
        ndim = header[2]
        shape = tuple(header[4:4 + ndim])
        expected_size = np.prod(shape, dtype=np.int64)
        assert expected_size * dtype.itemsize == header[3], f"Expected size {expected_size} does not match header size {header[3]} for dtype {dtype}"
        if expected_size > self.max_elems:
            raise ValueError(f"Tensor too large: {shape} (max_elems = {self.max_elems})")
        tensor_data = np.frombuffer(self._datas[slot_id][:expected_size * dtype.itemsize], dtype=dtype)
        tensor = tensor_data.reshape(shape).copy()
        self._debug(f"{role} read tensor of shape {tensor.shape} from slot {slot_id} with header {header[:8]}")
        return tensor

    def is_ready(self, slot_id: int, role: str) -> bool:
        self._validate(slot_id, role)
        ready = self._headers[slot_id][0] == self._ROLE_MAP[role]
        if ready:
            self._debug(f"{role} is ready to read from slot {slot_id}")
        return ready

    def _validate(self, slot_id: int, role: str):
        if slot_id < 0 or slot_id >= self._num_clients:
            raise ValueError(f"Invalid slot_id: {slot_id}, must be in range [0, {self._num_clients})")
        if role not in self._ROLE_MAP:
            raise ValueError(f"Invalid role: {role}")

    @property
    def num_clients(self) -> int:
        return self._num_clients

    def close(self):
        del self._headers, self._datas
        self._mmap.close()
        os.close(self._fd)

    def unlink(self):
        try:
            os.unlink(self.shm_path)
        except FileNotFoundError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
