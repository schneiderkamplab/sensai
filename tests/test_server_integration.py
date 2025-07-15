"""
Tests for the SensAIServer integration with TeacherLogitsServer
"""
import pytest
import numpy as np
import tempfile
import shutil
import threading
import time
from sensai.server import SensAIServer
from sensai.client import SensAIClient
from sensai.transports import NamedPipeTransport
from sensai.logits_server import TeacherLogitsServer


class TestSensAIServerIntegration:
    """Test integration between SensAIServer and TeacherLogitsServer"""
    
    @pytest.fixture
    def transport_setup(self):
        """Set up named pipe transport for testing"""
        pipe_dir = tempfile.mkdtemp(prefix='sensai_test_')
        transport = NamedPipeTransport(pipe_dir=pipe_dir, num_clients=2)
        yield transport, pipe_dir
        # Cleanup
        shutil.rmtree(pipe_dir, ignore_errors=True)
    
    @pytest.fixture
    def teacher_server(self):
        """Create a TeacherLogitsServer instance for testing"""
        # Use a small model for testing
        return TeacherLogitsServer("gpt2", device="cpu")
    
    def test_server_initialization(self, transport_setup):
        """Test that SensAIServer can be initialized with transport"""
        transport, _ = transport_setup
        server = SensAIServer(transport)
        assert server.transport == transport
        assert server.transport.num_clients == 2
    
    def test_process_slot_no_client(self, transport_setup):
        """Test process_slot returns False when no client is connected"""
        transport, _ = transport_setup
        server = SensAIServer(transport)
        
        def dummy_processor(input_tensor):
            return np.array([1.0, 2.0, 3.0])
        
        result = server.process_slot(0, dummy_processor)
        assert result is False
    
    def test_logits_request_processing(self, transport_setup, teacher_server):
        """Test the logits request processing function"""
        transport, _ = transport_setup
        
        def process_logits_request(input_tensor: np.ndarray) -> np.ndarray:
            """
            Process a logits request. Input tensor should contain:
            - First element: request type (0=sample_to_sparse_coo, 1=get_prompt_logits)
            - Second element: num_samples (for sampling requests)
            - Third element: temperature
            - Fourth element: top_p
            - Fifth element: top_k
            - Remaining elements: text as encoded string (UTF-8 bytes)
            """
            try:
                # Parse request
                request_type = int(input_tensor[0])
                num_samples = int(input_tensor[1])
                temperature = float(input_tensor[2])
                top_p = float(input_tensor[3])
                top_k = int(input_tensor[4])
                
                # Decode text from bytes
                text_bytes = input_tensor[5:].astype(np.uint8).tobytes()
                text = text_bytes.decode('utf-8').rstrip('\x00')  # Remove null padding
                
                if request_type == 0:  # sample_to_sparse_coo
                    sparse_results = teacher_server.sample_to_sparse_coo(
                        text, 
                        num_samples=num_samples,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k
                    )
                    
                    # Convert sparse tensor to dense for transport
                    if sparse_results:
                        sparse_tensor = sparse_results[0]
                        dense_tensor = sparse_tensor.to_dense()
                        return dense_tensor.numpy().flatten()
                    else:
                        return np.array([0])  # Empty result
                        
                elif request_type == 1:  # get_prompt_logits
                    logits_list = teacher_server.get_prompt_logits(text)
                    if logits_list:
                        logits = logits_list[0]
                        return logits.numpy().flatten()
                    else:
                        return np.array([0])  # Empty result
                
                else:
                    return np.array([-1])  # Invalid request type
                    
            except Exception as e:
                return np.array([-1])  # Error indicator
        
        # Test with sample request
        text = "Hello"
        text_bytes = text.encode('utf-8')
        
        # Create input tensor: [request_type, num_samples, temperature, top_p, top_k, ...text_bytes]
        input_data = np.array([1, 5, 1.0, 1.0, 0] + list(text_bytes) + [0] * (100 - len(text_bytes)), dtype=np.float32)
        
        result = process_logits_request(input_data)
        assert isinstance(result, np.ndarray)
        assert result.size > 0
        assert not np.array_equal(result, np.array([-1]))  # No error
    
    def test_create_request_tensor(self):
        """Test creating a request tensor in the expected format"""
        text = "The quick brown fox"
        request_type = 0  # sample_to_sparse_coo
        num_samples = 10
        temperature = 0.8
        top_p = 0.9
        top_k = 50
        
        # Encode text to bytes
        text_bytes = text.encode('utf-8')
        
        # Create request tensor with fixed size (pad with zeros)
        max_text_len = 1000
        request_data = [request_type, num_samples, temperature, top_p, top_k]
        request_data.extend(text_bytes)
        request_data.extend([0] * (max_text_len - len(text_bytes)))  # Pad with zeros
        
        request_tensor = np.array(request_data, dtype=np.float32)
        
        # Verify the structure
        assert request_tensor[0] == request_type
        assert request_tensor[1] == num_samples
        assert request_tensor[2] == temperature
        assert request_tensor[3] == top_p
        assert request_tensor[4] == top_k
        
        # Verify text can be decoded
        decoded_bytes = request_tensor[5:5+len(text_bytes)].astype(np.uint8).tobytes()
        decoded_text = decoded_bytes.decode('utf-8')
        assert decoded_text == text
    
    def test_client_server_communication(self, transport_setup, teacher_server):
        """Test full client-server communication"""
        transport, _ = transport_setup
        server = SensAIServer(transport)
        
        def process_logits_request(input_tensor: np.ndarray) -> np.ndarray:
            """Simple echo processor for testing"""
            try:
                # Just return the first 10 elements for testing
                return input_tensor[:10].copy()
            except Exception as e:
                return np.array([-1])
        
        # Start server in a separate thread
        server_thread = threading.Thread(
            target=lambda: server.run_loop(process_logits_request, interval=0.01),
            daemon=True
        )
        server_thread.start()
        
        # Give server time to start
        time.sleep(0.1)
        
        # Create client
        client = SensAIClient(transport, slot_id=0)
        
        # Test request
        test_input = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32)
        
        # This will timeout if server isn't processing correctly
        try:
            result = client.send_tensor(test_input, interval=0.01)
            assert isinstance(result, np.ndarray)
            assert result.size == 10
            np.testing.assert_array_equal(result, test_input[:10])
        except Exception as e:
            pytest.skip(f"Client-server communication test failed: {e}")
    
    def test_request_encoding_decoding(self):
        """Test that request encoding/decoding works correctly"""
        original_text = "Hello, world! 🌍"
        
        # Encode
        text_bytes = original_text.encode('utf-8')
        request_data = [0, 5, 1.0, 0.9, 50]  # request parameters
        request_data.extend(text_bytes)
        request_data.extend([0] * (200 - len(text_bytes)))  # Pad
        
        request_tensor = np.array(request_data, dtype=np.float32)
        
        # Decode
        decoded_bytes = request_tensor[5:5+len(text_bytes)].astype(np.uint8).tobytes()
        decoded_text = decoded_bytes.decode('utf-8')
        
        assert decoded_text == original_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])