import numpy as np
import time

__all__ = ["SensAIClient"]

class SensAIClient:

    def __init__(self, transport):
        self.transport = transport

    def send_tensor(self, tensor: np.ndarray | list[np.ndarray], interval: float = 0.001) -> np.ndarray:
        self.transport.write_tensor(tensor)
        while not self.transport.is_ready():
            time.sleep(interval)
        result = self.transport.read_tensor()
        return result
