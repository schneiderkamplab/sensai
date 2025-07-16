from abc import ABC, abstractmethod
import numpy as np

__all__ = ["Transport"]

class Transport(ABC):

    _DTYPE_MAP = {k: np.dtype(v) for k, v in {
            0: np.float32,
            1: np.float16,
            2: np.int32,
            3: np.int64,
            4: np.uint8
        }.items()
    }

    def __init__(self, path: str, num_slots: int, is_server: bool, debug: bool = True):
        self.path = path
        self.num_clients = num_slots
        self.is_server = is_server
        self.debug = debug

    @abstractmethod
    def is_ready(self, slot_id: int) -> bool:
        ...

    @abstractmethod
    def read_tensor(self, slot_id: int) -> np.ndarray | list[np.ndarray] | None:
        ...

    @abstractmethod
    def write_tensor(self, slot_id: int, tensor: np.ndarray | list[np.ndarray]) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def unlink(self) -> None:
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
        if not (0 <= slot_id < self.num_clients):
            raise ValueError(f"Invalid slot_id: {slot_id}, must be in range [0, {self.num_clients})")

    @property
    def role(self) -> str:
        return "server" if self.is_server else "client"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
