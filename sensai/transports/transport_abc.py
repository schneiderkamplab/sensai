from abc import ABC, abstractmethod
import numpy as np

__all__ = ["ClientMixin", "Transport"]

class ClientMixin:

    def write_tensor(self, tensor: np.ndarray | list[np.ndarray]) -> None:
        self.write_tensor_slot(self.slot_id, tensor)

    def read_tensor(self) -> np.ndarray | list[np.ndarray] | None:
        return self.read_tensor_slot(self.slot_id)

    def is_ready(self) -> bool:
        return self.is_ready_slot(self.slot_id)

class Transport(ABC):

    _DTYPE_MAP = {k: np.dtype(v) for k, v in {
            0: np.float32,
            1: np.float16,
            2: np.int32,
            3: np.int64,
            4: np.uint8
        }.items()
    }

    def __init__(self, path: str, debug: bool = True, is_server: bool = False):
        self.path = path
        self.debug = debug
        self.is_server = is_server

    @abstractmethod
    def is_ready_slot(self, slot_id: int) -> bool:
        ...

    @abstractmethod
    def read_tensor_slot(self, slot_id: int) -> np.ndarray | list[np.ndarray] | None:
        ...

    @abstractmethod
    def write_tensor_slot(self, slot_id: int, tensor: np.ndarray | list[np.ndarray]) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def supports_multi_tensor(self) -> bool:
        return False

    def _debug(self, msg):
        if self.debug:
            print(f"[{self.__class__.__name__}] {msg}")

    def _decode_dtype(self, dtype_code: int):
        return self._DTYPE_MAP[dtype_code]

    def _encode_dtype(self, dtype: np.dtype) -> int:
        for k, v in self._DTYPE_MAP.items():
            if v == dtype.type:
                return k
        raise ValueError(f"Unsupported dtype: {dtype}")

    def _validate(self, slot_id: int):
        if self.is_server:
            if not (0 <= slot_id < self.num_slots):
                raise ValueError(f"Invalid slot_id: {slot_id}, must be in range [0, {self.num_slots})")
        else:
            if slot_id != self.slot_id:
                raise ValueError(f"Client can only use slot {self.slot_id}, got {slot_id}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
