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

    _MAX_DIMS = 8
    _HEADER_DTYPE = np.dtype([
        ('status', 'i8'),
        ('dtype_code', 'i8'),
        ('ndim', 'i8'),
        ('nbytes', 'i64'),
        ('shape', f'({_MAX_DIMS},)i8')
    ])

    def __init__(self, shm_path: str, num_clients: int, max_elems: int, max_dtype: np.dtype, debug: bool = True):
        self.shm_path = shm_path
        self._num_clients = num_clients
        self.max_elems = max_elems
        self.max_dtype = np.dtype(max_dtype)
        self.debug = debug
        self._header_size = 256  # fixed header size for cache alignment
        assert self._HEADER_DTYPE.itemsize <= self._header_size, \
            f"Header dtype size {self._HEADER_DTYPE.itemsize} exceeds buffer size {self._header_size}"
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
            header = np.frombuffer(header_buf, dtype=self._HEADER_DTYPE, count=1)[0]
            header['status'] = self._ROLE_MAP["client"]
            self._headers[slot_id] = header
            self._datas[slot_id] = data_buf

    def _debug(self, msg):
        if self.debug:
            print(f"[SharedMemoryTransport] {msg}")

    def write_tensor(self, slot_id: int, tensor: np.ndarray, role: str) -> None:
        self._validate(slot_id, role)
        header = self._headers[slot_id]
        shape = tensor.shape

        if tensor.size > self.max_elems:
            raise ValueError(f"Tensor too large: {tensor.size} > {self.max_elems}")
        if tensor.ndim > self._MAX_DIMS:
            raise ValueError(f"Tensor rank {tensor.ndim} exceeds MAX_DIMS={self._MAX_DIMS}")

        header['dtype_code'] = self._encode_dtype(tensor.dtype)
        header['ndim'] = tensor.ndim
        header['nbytes'] = np.int64(tensor.nbytes)
        header['shape'][:tensor.ndim] = shape
        if tensor.ndim < self._MAX_DIMS:
            header['shape'][tensor.ndim:] = 0  # zero-fill unused dims
        header['status'] = 1 - self._ROLE_MAP[role]

        self._datas[slot_id][:tensor.nbytes] = tensor.tobytes()
        self._debug(f"{role} wrote tensor of shape {shape} to slot {slot_id} with header {header}")

    def read_tensor(self, slot_id: int, role: str) -> np.ndarray | None:
        self._validate(slot_id, role)
        header = self._headers[slot_id]

        if header['status'] != self._ROLE_MAP[role]:
            self._debug(f"{role} found no tensor to read in slot {slot_id}")
            return None

        dtype = self._resolve_dtype(header['dtype_code'])
        ndim = int(header['ndim'])
        nbytes = int(header['nbytes'])
        shape = tuple(int(d) for d in header['shape'][:ndim])

        expected_elems = np.prod(shape, dtype=np.int64)
        expected_nbytes = expected_elems * dtype.itemsize

        if expected_nbytes != nbytes:
            raise ValueError(f"Inconsistent nbytes in header: expected {expected_nbytes}, found {nbytes}")
        if expected_elems > self.max_elems:
            raise ValueError(f"Tensor too large: {shape} (max_elems = {self.max_elems})")

        data = self._datas[slot_id][:nbytes]
        tensor = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        self._debug(f"{role} read tensor of shape {shape} from slot {slot_id} with header {header}")
        return tensor

    def is_ready(self, slot_id: int, role: str) -> bool:
        self._validate(slot_id, role)
        ready = self._headers[slot_id]['status'] == self._ROLE_MAP[role]
        if ready:
            self._debug(f"{role} is ready to read from slot {slot_id}")
        return ready

    def _validate(self, slot_id: int, role: str):
        if not (0 <= slot_id < self._num_clients):
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
