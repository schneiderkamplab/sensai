import numpy as np
import time

__all__ = ["SensAIClient"]

class SensAIClient:

    def __init__(self, transport, slot_id: int):
        self.transport = transport
        self.slot_id = slot_id

    def send_tensor(self, tensor: np.ndarray, interval: float = 0.001) -> np.ndarray:
        self.transport.write_tensor(self.slot_id, tensor, role="client")
        while not self.transport.is_ready(self.slot_id, role="client"):
            time.sleep(interval)
        result = self.transport.read_tensor(self.slot_id, role="client")
        return result
