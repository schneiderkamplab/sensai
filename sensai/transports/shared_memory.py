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

    def _header(self, slot_id: int) -> np.ndarray:
        header_offset = slot_id * self._slot_size
        buf = memoryview(self._mmap)[header_offset : header_offset + self._header_size]
        return np.frombuffer(buf, dtype=np.int64)

    def _data(self, slot_id: int) -> memoryview:
        data_offset = slot_id * self._slot_size + self._header_size
        return memoryview(self._mmap)[data_offset : data_offset + self.max_elems * self.max_dtype.itemsize]

    def write_tensor(self, slot_id: int, tensor: np.ndarray, role: str) -> None:
        if tensor.size > self.max_elems:
            raise ValueError(f"Tensor too large: {tensor.size} > {self.max_elems}")
        if role not in ("client", "server"):
            raise ValueError(f"Invalid role: {role}")
        self._data(slot_id)[:tensor.nbytes] = tensor.tobytes()
        header = self._header(slot_id)
        header[1] = self._encode_dtype(tensor.dtype)
        header[2] = tensor.ndim
        header[3] = tensor.size
        header[4:4 + tensor.ndim] = tensor.shape
        header[0] = 1 if role == "client" else 0  # client sets READY, server sets EMPTY

    def read_tensor(self, slot_id: int, role: str) -> np.ndarray:
        if role not in ("client", "server"):
            raise ValueError(f"Invalid role: {role}")
        header = self._header(slot_id)
        expected_status = 0 if role == "client" else 1
        if header[0] != expected_status:
            return None
        dtype_enum = header[1]
        ndim = header[2]
        shape = tuple(header[4:4 + ndim])
        expected_size = np.prod(shape)
        if expected_size > self.max_elems:
            raise ValueError(f"Tensor shape too large: {shape} (max_elems = {self.max_elems})")
        dtype = self._resolve_dtype(dtype_enum)
        tensor_data = np.frombuffer(self._data(slot_id)[:expected_size * dtype.itemsize], dtype=dtype)
        return tensor_data.reshape(shape).copy()

    def is_ready(self, slot_id: int, role: str) -> bool:
        if role not in ("client", "server"):
            raise ValueError(f"Invalid role: {role}")
        header = self._header(slot_id)
        expected_status = 0 if role == "client" else 1
        return header[0] == expected_status

    def _encode_dtype(self, dtype: np.dtype) -> int:
        for key, value in self._DTYPE_MAP.items():
            if value == dtype.type:
                return key
        raise ValueError(f"Unsupported dtype: {dtype}")

    def _resolve_dtype(self, dtype_enum: int):
        return self._DTYPE_MAP.get(dtype_enum, np.float32)

    def close(self):
        self._mmap.close()
        os.close(self._fd)

    def unlink(self):
        try:
            os.unlink(self.shm_path)
        except FileNotFoundError:
            pass
