import mmap
import numpy as np
import os

from .transport_abc import *

__all__ = ["SharedMemoryClient", "SharedMemoryServer"]

class SharedMemoryTransport(Transport):

    _alignment = mmap.PAGESIZE
    _meta_header_dtype = np.dtype([
        ('num_slots', np.int64),
        ('max_nbytes', np.int64),
        ('max_tensors', np.int64),
        ('max_dims', np.int64),
    ])
    @staticmethod
    def _align_size(size: int, alignment: int) -> int:
        return -(-size // alignment) * alignment
    _meta_header_size = _align_size(_meta_header_dtype.itemsize, _alignment)
    @staticmethod
    def _header_dtype(max_tensors: int, max_dims: int) -> np.dtype:
        return np.dtype([
            ('status', np.int64),
            ('recipient', np.int64),
            ('num_tensors', np.int64),
            ('dtype_codes', (np.int64, max_tensors)),
            ('ndims', (np.int64, max_tensors)),
            ('nbytess', (np.int64, max_tensors)),
            ('shapes', (np.int64, (max_tensors, max_dims)))
        ])

    def __init__(self, path: str, debug: bool, is_server: bool):
        super().__init__(path=path, debug=debug, is_server=is_server)
        self._headers = {}
        self._datas = {}

    def _init_slot(self, slot_id: int, header_dtype: np.dtype, offset: int = 0):
        header_buf = memoryview(self._mmap)[offset : offset + self._header_size]
        data_buf = memoryview(self._mmap)[offset + self._header_size : offset + self._slot_size]
        self._debug(f"Setting up slot {slot_id} with header at {offset} and data at {offset + self._header_size}")
        self._debug(f"Header buffer size: {len(header_buf)}, Data buffer size: {len(data_buf)}")
        header = np.frombuffer(header_buf, dtype=header_dtype, count=1)[0]
        header['status'] = 0
        self._headers[slot_id] = header
        self._datas[slot_id] = data_buf

    def write_tensor_slot(self, slot_id: int, tensor: np.ndarray | list[np.ndarray]) -> None:
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
        header['status'] = 0 if self.is_server else 1
        self._debug(f"wrote {len(tensors)} tensor(s) to slot {slot_id} (total {offset} bytes)")

    def read_tensor_slot(self, slot_id: int) -> np.ndarray | list[np.ndarray] | None:
        self._validate(slot_id)
        header = self._headers[slot_id]
        if header['status'] != (1 if self.is_server else 0):
            self._debug(f"found no tensor to read in slot {slot_id}")
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
        self._debug(f"read {num_tensors} tensor(s) from slot {slot_id}")
        return tensors[0] if num_tensors == 1 else tensors

    def is_ready_slot(self, slot_id: int) -> bool:
        self._validate(slot_id)
        ready = self._headers[slot_id]['status'] == (1 if self.is_server else 0)
        if ready:
            self._debug(f"is ready to read from slot {slot_id}")
        return ready

    def close(self):
        del self._headers, self._datas
        self._mmap.close()
        os.close(self._fd)
        if self.is_server:
            self.unlink()

    def _unlink(self):
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    def supports_multi_tensor(self) -> bool:
        return True

class SharedMemoryClient(SharedMemoryTransport, ClientMixin):

    def __init__(self, path: str, slot_id: int, debug: bool = True):
        super().__init__(path=path, debug=debug, is_server=False)
        self.slot_id = slot_id
        self._fd = os.open(path, os.O_RDWR)
        self._mmap = mmap.mmap(self._fd, self._meta_header_size)
        meta_header = np.frombuffer(memoryview(self._mmap), dtype=self._meta_header_dtype, count=1)[0]
        self.num_slots = meta_header['num_slots']
        self.max_nbytes = meta_header['max_nbytes']
        self.max_tensors = meta_header['max_tensors']
        self.max_dims = meta_header['max_dims']
        del meta_header
        self._mmap.close()
        os.close(self._fd)
        header_dtype = self._header_dtype(self.max_tensors, self.max_dims)
        self._header_size = self._align_size(header_dtype.itemsize, self._alignment)
        self._slot_size = self._header_size + self.max_nbytes
        self._fd = os.open(path, os.O_RDWR)

        file_size = os.fstat(self._fd).st_size
        required_size = self._meta_header_size + (slot_id + 1) * self._slot_size
        print(f"meta_header_size={self._meta_header_size}, slot_size={self._slot_size}")
        print(f"file_size={file_size}, required_size={required_size}, slot_id={slot_id}, slot_size={self._slot_size}, meta_header_size={self._meta_header_size}")

        self._mmap = mmap.mmap(self._fd, self._slot_size, offset=self._meta_header_size + slot_id * self._slot_size)
        self._headers = {}
        self._datas = {}
        self._debug(f"Initializing shared memory client for slot {self.slot_id}")
        self._init_slot(self.slot_id, header_dtype)

class SharedMemoryServer(SharedMemoryTransport):

    def __init__(self, path: str, num_slots: int, max_nbytes: int, max_tensors: int = 4, max_dims: int = 4, debug: bool = True):
        super().__init__(path, debug=debug, is_server=True)
        self.num_slots = num_slots
        self.max_nbytes = self._align_size(max_nbytes, self._alignment)
        self.max_tensors = max_tensors
        self.max_dims = max_dims
        header_dtype = self._header_dtype(self.max_tensors, self.max_dims)
        self._header_size = self._align_size(header_dtype.itemsize, self._alignment)
        self._slot_size = self._header_size + self.max_nbytes
        self._fd = os.open(path, os.O_CREAT | os.O_RDWR)
        os.ftruncate(self._fd, self._meta_header_size + self.num_slots * self._slot_size)
        self._mmap = mmap.mmap(self._fd, self._meta_header_size + self.num_slots * self._slot_size)
        meta_header = np.frombuffer(memoryview(self._mmap), dtype=self._meta_header_dtype, count=1)[0]
        meta_header['num_slots'] = self.num_slots
        meta_header['max_nbytes'] = self.max_nbytes
        meta_header['max_tensors'] = self.max_tensors
        meta_header['max_dims'] = self.max_dims
        self._debug(f"Initializing shared memory transport with {self.num_slots} slots")
        for slot_id in range(self.num_slots):
            self._init_slot(slot_id, header_dtype, offset=self._meta_header_size + slot_id * self._slot_size)
