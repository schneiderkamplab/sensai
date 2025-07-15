import mmap
import numpy as np
import os

class SharedMemoryTransport:

    _DTYPE_MAP = {
        0: np.float32,
        1: np.float16,
        2: np.int32,
        3: np.int64,
        4: np.uint8
    }

    _ROLE_MAP = {
        "client": 0,
        "server": 1
    }

    def __init__(self, shm_path: str, num_clients: int, max_elems: int, max_dtype: np.dtype):
        self.shm_path = shm_path
        self.num_clients = num_clients
        self.max_elems = max_elems
        self.max_dtype = np.dtype(max_dtype)
        self._header_size = 256  # fixed header size for cache alignment
        self._slot_size = self._header_size + max_elems * self.max_dtype.itemsize
        self._fd = os.open(shm_path, os.O_CREAT | os.O_RDWR)
        os.ftruncate(self._fd, self.num_clients * self._slot_size)
        self._mmap = mmap.mmap(self._fd, self.num_clients * self._slot_size)
        self._headers = {}
        self._datas = {}
        for slot_id in range(self.num_clients):
            offset = slot_id * self._slot_size
            header_buf = memoryview(self._mmap)[offset : offset + self._header_size]
            data_buf = memoryview(self._mmap)[offset + self._header_size : offset + self._slot_size]
            self._headers[slot_id] = np.frombuffer(header_buf, dtype=np.int64)
            self._headers[slot_id][0] = self._ROLE_MAP["client"]
            self._datas[slot_id] = data_buf

    def write_tensor(self, slot_id: int, tensor: np.ndarray, role: str) -> None:
        self._validate_role(role)
        header = self._headers[slot_id]
        if header[0] != 1-self._ROLE_MAP[role]:
            return None
        if tensor.size > self.max_elems:
            raise ValueError(f"Tensor too large: {tensor.size} > {self.max_elems}")
        self._datas[slot_id][:tensor.nbytes] = tensor.tobytes()
        header[1] = self._encode_dtype(tensor.dtype)
        header[2] = tensor.ndim
        header[3] = tensor.size
        header[4:4 + tensor.ndim] = tensor.shape
        header[0] = 1-self._ROLE_MAP[role]

    def read_tensor(self, slot_id: int, role: str) -> np.ndarray | None:
        self._validate_role(role)
        header = self._headers[slot_id]
        if header[0] != self._ROLE_MAP[role]:
            return None
        dtype_enum = header[1]
        ndim = header[2]
        shape = tuple(header[4:4 + ndim])
        expected_size = np.prod(shape)
        if expected_size > self.max_elems:
            raise ValueError(f"Tensor too large: {shape} (max_elems = {self.max_elems})")
        dtype = self._resolve_dtype(dtype_enum)
        tensor_data = np.frombuffer(self._datas[slot_id][:expected_size * dtype.itemsize], dtype=dtype)
        return tensor_data.reshape(shape).copy()

    def is_ready(self, slot_id: int, role: str) -> bool:
        self._validate_role(role)
        return self._headers[slot_id][0] == self._ROLE_MAP[role]

    def _encode_dtype(self, dtype: np.dtype) -> int:
        for key, value in self._DTYPE_MAP.items():
            if value == dtype.type:
                return key
        raise ValueError(f"Unsupported dtype: {dtype}")

    def _resolve_dtype(self, dtype_enum: int):
        if dtype_enum not in self._DTYPE_MAP:
            raise ValueError(f"Invalid dtype enum: {dtype_enum}")
        return self._DTYPE_MAP[dtype_enum]

    def _validate_role(self, role: str):
        if role not in self._ROLE_MAP:
            raise ValueError(f"Invalid role: {role}")

    def close(self):
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
