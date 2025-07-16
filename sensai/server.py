import numpy as np
import time

__all__ = ["SensAIServer"]

class SensAIServer:

    def __init__(self, transport):
        self.transport = transport

    def process_slot(self, slot_id: int, fn) -> bool:
        if not self.transport.is_ready(slot_id):
            return False
        input_tensor = self.transport.read_tensor(slot_id)
        output_tensor = fn(input_tensor)
        if not isinstance(output_tensor, np.ndarray) and not (isinstance(output_tensor, list) and all(isinstance(t, np.ndarray) for t in output_tensor)):
            raise ValueError("Processing function must return a NumPy ndarray.")
        self.transport.write_tensor(slot_id, output_tensor)
        return True

    def run_loop(self, fn, interval: float = 0.001):
        while True:
            any_processed = False
            for slot_id in range(self.transport.num_clients):
                any_processed |= self.process_slot(slot_id, fn)
            if not any_processed:
                time.sleep(interval)
