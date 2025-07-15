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
