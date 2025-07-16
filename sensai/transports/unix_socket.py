import os
import socket
import struct
import select
import numpy as np
from .transport_abc import Transport

__ALL__ = ["UnixSocketTransport"]

HEADER_FORMAT = "iii"  # dtype_enum, ndim, size
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
CHUNK_SIZE = 4 * 1024 * 1024  # 4MB

class UnixSocketTransport(Transport):
    def __init__(self, socket_dir: str, num_clients: int, server_mode: bool = True, debug: bool = True):
        self.socket_dir = socket_dir
        self._num_clients = num_clients
        self.server_mode = server_mode
        self.debug = debug
        self.c2s_paths = {}
        self.s2c_paths = {}
        self.connections = {}
        os.makedirs(socket_dir, exist_ok=True)
        for slot_id in range(num_clients):
            c2s_path = os.path.join(socket_dir, f"c2s_{slot_id}.sock")
            s2c_path = os.path.join(socket_dir, f"s2c_{slot_id}.sock")
            if self.server_mode:
                for path in (c2s_path, s2c_path):
                    if os.path.exists(path):
                        os.unlink(path)
            self.c2s_paths[slot_id] = c2s_path
            self.s2c_paths[slot_id] = s2c_path
        if self.server_mode:
            self.listeners = {}
            for slot_id in range(num_clients):
                c2s_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                c2s_sock.bind(self.c2s_paths[slot_id])
                c2s_sock.listen(1)
                s2c_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s2c_sock.bind(self.s2c_paths[slot_id])
                s2c_sock.listen(1)
                self.listeners[slot_id] = (c2s_sock, s2c_sock)
        else:
            for slot_id in range(num_clients):
                conn_c2s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                conn_s2c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                conn_c2s.connect(self.c2s_paths[slot_id])
                conn_s2c.connect(self.s2c_paths[slot_id])
                self.connections[slot_id] = (conn_c2s, conn_s2c)

    def _debug(self, msg):
        if self.debug:
            print(f"[MultiSocketTransport] {msg}")

    def is_ready(self, slot_id: int, role: str) -> bool:
        conn = self._get_connection(slot_id, role, "read")
        if conn is None:
            return False
        try:
            rlist, _, _ = select.select([conn], [], [], 0)
            if not rlist:
                return False
            peek = conn.recv(1, socket.MSG_PEEK)
            if not peek:
                self._debug(f"{role} peer disconnected on slot {slot_id}")
                self._close(slot_id)
                return False
            return True
        except Exception as e:
            self._debug(f"is_ready error on slot {slot_id}: {e}")
            self._close(slot_id)
            return False

    def _accept_if_needed(self, slot_id):
        if slot_id not in self.connections:
            c2s_sock, s2c_sock = self.listeners[slot_id]
            conn_c2s, _ = c2s_sock.accept()
            conn_s2c, _ = s2c_sock.accept()
            self.connections[slot_id] = (conn_c2s, conn_s2c)
            self._debug(f"Accepted connections for slot {slot_id}")

    def _get_connection(self, slot_id: int, role: str, direction: str):
        if self.server_mode:
            self._accept_if_needed(slot_id)
        if slot_id not in self.connections:
            return None
        conn_c2s, conn_s2c = self.connections[slot_id]
        if role == "server":
            return conn_c2s if direction == "read" else conn_s2c
        elif role == "client":
            return conn_s2c if direction == "read" else conn_c2s
        else:
            raise ValueError(f"Invalid role: {role}")

    def _send_all(self, sock, buffer: bytes):
        total_sent = 0
        while total_sent < len(buffer):
            sent = sock.send(buffer[total_sent : total_sent + CHUNK_SIZE])
            if sent == 0:
                raise IOError("Socket send failed")
            total_sent += sent

    def _recv_exact(self, sock, size):
        buf = bytearray()
        while len(buf) < size:
            chunk = sock.recv(min(CHUNK_SIZE, size - len(buf)))
            if not chunk:
                raise IOError("Socket connection broken")
            buf += chunk
        return bytes(buf)

    def write_tensor(self, slot_id: int, tensor: np.ndarray, role: str):
        s = self._get_connection(slot_id, role, "write")
        if s is None:
            raise IOError(f"No write socket for slot {slot_id}")
        dtype_enum = self._encode_dtype(tensor.dtype)
        ndim = tensor.ndim
        size = tensor.size
        shape = tensor.shape
        header = struct.pack("iii", dtype_enum, ndim, size)
        shape_data = struct.pack("i" * ndim, *shape)
        data = tensor.tobytes()
        self._send_all(s, header + shape_data + data)
        self._debug(f"{role} wrote tensor of shape {shape} to slot {slot_id}")

    def read_tensor(self, slot_id: int, role: str) -> np.ndarray:
        s = self._get_connection(slot_id, role, "read")
        if s is None:
            raise IOError(f"No read socket for slot {slot_id}")
        header = self._recv_exact(s, HEADER_SIZE)
        dtype_enum, ndim, size = struct.unpack("iii", header)
        shape_data = self._recv_exact(s, 4 * ndim)
        shape = struct.unpack("i" * ndim, shape_data)
        dtype = self._resolve_dtype(dtype_enum)
        data = self._recv_exact(s, size * dtype.itemsize)
        tensor = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        self._debug(f"{role} read tensor of shape {shape} from slot {slot_id}")
        return tensor

    def _close(self, slot_id):
        conns = self.connections.pop(slot_id, None)
        if conns:
            for c in conns:
                try:
                    c.close()
                except Exception:
                    pass
        self._debug(f"Closed slot {slot_id}")

    def close(self):
        if self.server_mode:
            for c2s, s2c in self.listeners.values():
                try:
                    c2s.close()
                    s2c.close()
                except Exception:
                    pass
        for slot_id in list(self.connections):
            self._close(slot_id)

    def unlink(self):
        for path in list(self.c2s_paths.values()) + list(self.s2c_paths.values()):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def num_clients(self) -> int:
        return self._num_clients
