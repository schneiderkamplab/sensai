from abc import ABC, abstractmethod
import numpy as np

__all__ = ["Transport"]

class Transport(ABC):
    @abstractmethod
    def is_ready(self, slot_id: int, role: str) -> bool:
        ...

    @abstractmethod
    def read_tensor(self, slot_id: int, role: str) -> np.ndarray:
        ...

    @abstractmethod
    def write_tensor(self, slot_id: int, tensor: np.ndarray, role: str) -> None:
        ...

    @property
    @abstractmethod
    def num_clients(self) -> int:
        ...

    def _encode_dtype(self, dtype: np.dtype) -> int:
        for k, v in self._DTYPE_MAP().items():
            if v == dtype.type:
                return k
        raise ValueError(f"Unsupported dtype: {dtype}")

    def _resolve_dtype(self, dtype_enum: int):
        return np.dtype(self._DTYPE_MAP().get(dtype_enum, np.float32))

    @staticmethod
    def _DTYPE_MAP():
        return {
            0: np.float32,
            1: np.float16,
            2: np.int32,
            3: np.int64,
            4: np.uint8
        }
