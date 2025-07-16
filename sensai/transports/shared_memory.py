import mmap
import numpy as np
import os

from .transport_abc import Transport

__all__ = ["SharedMemoryTransport"]

class SharedMemoryTransport(Transport):

    def __init__(
        self,
        path: str,
        num_clients: int,
        max_nbytes: int,
        is_server: bool = False,
        debug: bool = True,
        max_tensors: int = 4,
        max_dims: int = 8,
        cache_alignment: int = 256,
    ):
        super().__init__(path, num_slots=num_clients, is_server=is_server, debug=debug)
        self.max_nbytes = max_nbytes
        self._server = 1 if is_server else 0
        self.debug = debug
        self.max_tensors = max_tensors
        self.max_dims = max_dims
        header_dtype = np.dtype([
            ('status', np.int64),
            ('slot_id', np.int64),
            ('num_slots', np.int64),
            ('num_tensors', np.int64),
            ('max_nbytes', np.int64),
            ('max_tensors', np.int64),
            ('max_dims', np.int64),
            ('dtype_codes', (np.int64, self.max_tensors)),
            ('ndims', (np.int64, self.max_tensors)),
            ('nbytess', (np.int64, self.max_tensors)),
            ('shapes', (np.int64, (self.max_tensors, self.max_dims)))
        ])
        self._header_size = -(-header_dtype.itemsize // cache_alignment) * cache_alignment
        assert header_dtype.itemsize <= self._header_size, \
            f"Header dtype size {header_dtype.itemsize} exceeds buffer size {self._header_size}"
        self._slot_size = self._header_size + max_nbytes
        self._fd = os.open(path, os.O_CREAT | os.O_RDWR)
        os.ftruncate(self._fd, self.num_clients * self._slot_size)
        self._mmap = mmap.mmap(self._fd, self.num_clients * self._slot_size)
        self._headers = {}
        self._datas = {}
        self._debug(f"Initializing shared memory transport with {self.num_clients} clients")
        for slot_id in range(self.num_clients):
            offset = slot_id * self._slot_size
            header_buf = memoryview(self._mmap)[offset : offset + self._header_size]
            data_buf = memoryview(self._mmap)[offset + self._header_size : offset + self._slot_size]
            self._debug(f"Setting up slot {slot_id} with header at {offset} and data at {offset + self._header_size}")
            self._debug(f"Header buffer size: {len(header_buf)}, Data buffer size: {len(data_buf)}")
            header = np.frombuffer(header_buf, dtype=header_dtype, count=1)[0]
            header['status'] = 0
            header['slot_id'] = slot_id
            header['num_slots'] = self.num_clients
            self._headers[slot_id] = header
            self._datas[slot_id] = data_buf

    def write_tensor(self, slot_id: int, tensor: np.ndarray | list[np.ndarray]) -> None:
        tensors = tensor if isinstance(tensor, list) else [tensor]
        self._validate(slot_id)
        header = self._headers[slot_id]
        if len(tensors) > self.max_tensors:
            raise ValueError(f"Too many tensors: {len(tensors)} > max_tensors={self.max_tensors}")
        offset = 0
        for i, t in enumerate(tensors):
            if t.ndim > self.max_dims:
                raise ValueError(f"Tensor {i} has too many dims: {t.ndim} > max_dims={self.max_dims}")
            nbytes = t.nbytes
            if offset + nbytes > self.max_nbytes:
                raise ValueError(f"Tensors exceed shared memory buffer capacity at tensor {i} (offset={offset}, size={nbytes})")
            self._datas[slot_id][offset : offset + nbytes] = t.tobytes()
            header['dtype_codes'][i] = self._encode_dtype(t.dtype)
            header['ndims'][i] = t.ndim
            header['nbytess'][i] = nbytes
            header['shapes'][i, :t.ndim] = t.shape
            if t.ndim < self.max_dims:
                header['shapes'][i, t.ndim:] = 0
            offset += nbytes
        header['num_tensors'] = len(tensors)
        header['status'] = 1 - self._server
        self._debug(f"{self.role} wrote {len(tensors)} tensor(s) to slot {slot_id} (total {offset} bytes)")

    def read_tensor(self, slot_id: int) -> np.ndarray | list[np.ndarray] | None:
        self._validate(slot_id)
        header = self._headers[slot_id]
        if header['status'] != self._server:
            self._debug(f"{self.role} found no tensor to read in slot {slot_id}")
            return None
        num_tensors = int(header['num_tensors'])
        if not (1 <= num_tensors <= self.max_tensors):
            raise ValueError(f"Invalid num_tensors: {num_tensors}")
        tensors = []
        offset = 0
        for i in range(num_tensors):
            dtype = self._decode_dtype(header['dtype_codes'][i])
            ndim = int(header['ndims'][i])
            nbytes = int(header['nbytess'][i])
            shape = tuple(int(d) for d in header['shapes'][i, :ndim])
            if offset + nbytes > len(self._datas[slot_id]):
                raise ValueError(f"Tensor {i} exceeds buffer bounds: offset={offset}, nbytes={nbytes}")
            buf = self._datas[slot_id][offset : offset + nbytes]
            tensor = np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
            tensors.append(tensor)
            offset += nbytes
        self._debug(f"{self.role} read {num_tensors} tensor(s) from slot {slot_id}")
        return tensors[0] if num_tensors == 1 else tensors

    def is_ready(self, slot_id: int) -> bool:
        self._validate(slot_id)
        ready = self._headers[slot_id]['status'] == self._server
        if ready:
            self._debug(f"{self.role} is ready to read from slot {slot_id}")
        return ready

    def close(self):
        del self._headers, self._datas
        self._mmap.close()
        os.close(self._fd)

    def unlink(self):
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    def supports_multi_tensor(self) -> bool:
        return True
