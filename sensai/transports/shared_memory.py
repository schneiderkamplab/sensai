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

    _CACHE_ALIGNMENT = 256

    def __init__(
        self,
        shm_path: str,
        num_clients: int,
        max_nbytes: int,
        debug: bool = True,
        max_tensors: int = 4,
        max_dims: int = 8
    ):
        self.shm_path = shm_path
        self._num_clients = num_clients
        self.max_nbytes = max_nbytes
        self.debug = debug
        self._max_tensors = max_tensors
        self._max_dims = max_dims
        header_dtype = np.dtype([
            ('status', np.int8),
            ('num_tensors', np.int8),
            ('dtype_codes', (np.int8, self._max_tensors)),
            ('ndims', (np.int8, self._max_tensors)),
            ('nbytess', (np.int64, self._max_tensors)),
            ('shapes', (np.int64, (self._max_tensors, self._max_dims)))
        ])
        self._header_size = -(-header_dtype.itemsize // self._CACHE_ALIGNMENT) * self._CACHE_ALIGNMENT
        assert header_dtype.itemsize <= self._header_size, \
            f"Header dtype size {header_dtype.itemsize} exceeds buffer size {self._header_size}"

        self._slot_size = self._header_size + max_nbytes
        self._fd = os.open(shm_path, os.O_CREAT | os.O_RDWR)
        os.ftruncate(self._fd, self._num_clients * self._slot_size)
        self._mmap = mmap.mmap(self._fd, self._num_clients * self._slot_size)
        self._headers = {}
        self._datas = {}
        self._debug(f"Initializing shared memory transport with {self._num_clients} clients")
        for slot_id in range(self._num_clients):
            offset = slot_id * self._slot_size
            header_buf = memoryview(self._mmap)[offset : offset + self._header_size]
            data_buf = memoryview(self._mmap)[offset + self._header_size : offset + self._slot_size]
            self._debug(f"Setting up slot {slot_id} with header at {offset} and data at {offset + self._header_size}")
            self._debug(f"Header buffer size: {len(header_buf)}, Data buffer size: {len(data_buf)}")
            header = np.frombuffer(header_buf, dtype=header_dtype, count=1)[0]
            header['status'] = self._ROLE_MAP["client"]
            self._headers[slot_id] = header
            self._datas[slot_id] = data_buf

    def _debug(self, msg):
        if self.debug:
            print(f"[SharedMemoryTransport] {msg}")

    def write_tensor(self, slot_id: int, tensor: np.ndarray | list[np.ndarray], role: str) -> None:
        tensors = tensor if isinstance(tensor, list) else [tensor]
        self._validate(slot_id, role)
        header = self._headers[slot_id]

        if len(tensors) > self._max_tensors:
            raise ValueError(f"Too many tensors: {len(tensors)} > max_tensors={self._max_tensors}")

        offset = 0
        for i, t in enumerate(tensors):
            if t.ndim > self._max_dims:
                raise ValueError(f"Tensor {i} has too many dims: {t.ndim} > max_dims={self._max_dims}")

            nbytes = t.nbytes

            if offset + nbytes > self.max_nbytes:
                raise ValueError(f"Tensors exceed shared memory buffer capacity at tensor {i} (offset={offset}, size={nbytes})")

            # Write to data buffer directly
            self._datas[slot_id][offset : offset + nbytes] = t.tobytes()

            # Write header fields
            header['dtype_codes'][i] = self._encode_dtype(t.dtype)
            header['ndims'][i] = t.ndim
            header['nbytess'][i] = nbytes
            header['shapes'][i, :t.ndim] = t.shape
            if t.ndim < self._max_dims:
                header['shapes'][i, t.ndim:] = 0

            offset += nbytes

        header['num_tensors'] = len(tensors)
        header['status'] = 1 - self._ROLE_MAP[role]
        self._debug(f"{role} wrote {len(tensors)} tensor(s) to slot {slot_id} (total {offset} bytes)")

    def read_tensor(self, slot_id: int, role: str) -> np.ndarray | list[np.ndarray] | None:
        self._validate(slot_id, role)
        header = self._headers[slot_id]

        if header['status'] != self._ROLE_MAP[role]:
            self._debug(f"{role} found no tensor to read in slot {slot_id}")
            return None

        num_tensors = int(header['num_tensors'])
        if not (1 <= num_tensors <= self._max_tensors):
            raise ValueError(f"Invalid num_tensors: {num_tensors}")

        tensors = []
        offset = 0

        for i in range(num_tensors):
            dtype = self._resolve_dtype(header['dtype_codes'][i])
            ndim = int(header['ndims'][i])
            nbytes = int(header['nbytess'][i])
            shape = tuple(int(d) for d in header['shapes'][i, :ndim])

            if offset + nbytes > len(self._datas[slot_id]):
                raise ValueError(f"Tensor {i} exceeds buffer bounds: offset={offset}, nbytes={nbytes}")

            buf = self._datas[slot_id][offset : offset + nbytes]
            tensor = np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
            tensors.append(tensor)
            offset += nbytes

        self._debug(f"{role} read {num_tensors} tensor(s) from slot {slot_id}")
        return tensors[0] if num_tensors == 1 else tensors

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
