import torch
import torch.nn.functional as F
import pytest
import numpy as np
from sensai.logits_server import TeacherLogitsServer, process_logits_request


class TestTeacherLogitsServer:
    """Test suite for TeacherLogitsServer functionality."""
    
    @pytest.fixture
    def server(self):
        """Create a TeacherLogitsServer instance for testing."""
        return TeacherLogitsServer("google/gemma-3-1b-it", device="cpu")
    
    def test_initialization(self, server):
        """Test that server initializes correctly."""
        assert server.tokenizer is not None
        assert server.model is not None
        assert server.device.type == "cpu"
        assert server.vocab_size > 0
        assert server.tokenizer.pad_token is not None
    
    def test_get_logits_from_input_ids_single_sequence(self, server):
        """Test get_logits_from_input_ids with a single sequence."""
        # Create input_ids directly
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # [batch_size=1, seq_len=5]
        
        logits = server.get_logits_from_input_ids(input_ids)
        
        # Check shape
        assert logits.dim() == 3
        assert logits.shape == (1, 4, server.vocab_size)  # [batch_size, seq_len-1, vocab_size]
        
        # Check that logits are finite
        assert torch.isfinite(logits).all()
    
    def test_get_logits_from_input_ids_batch(self, server):
        """Test get_logits_from_input_ids with batch input."""
        # Create batch input_ids
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # [batch_size=2, seq_len=4]
        
        logits = server.get_logits_from_input_ids(input_ids)
        
        # Check shape
        assert logits.dim() == 3
        assert logits.shape == (2, 3, server.vocab_size)  # [batch_size, seq_len-1, vocab_size]
        
        # Check that logits are finite
        assert torch.isfinite(logits).all()
    
    def test_process_input_ids_full_logits_mode(self, server):
        """Test process_input_ids with full logits mode (no sampling)."""
        # Test with num_samples=None for full logits
        full_logits_server = TeacherLogitsServer(
            "google/gemma-3-1b-it", 
            device="cpu", 
            num_samples=None
        )
        
        # Create sample input_ids
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # [batch_size=1, seq_len=5]
        
        result = full_logits_server.process_input_ids(input_ids)
        
        # Should return full logits tensor directly
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 4, full_logits_server.vocab_size)  # [batch_size, seq_len-1, vocab_size]
        assert torch.isfinite(result).all()
        
        # Test with num_samples=0 (should also return full logits)
        zero_samples_server = TeacherLogitsServer(
            "google/gemma-3-1b-it", 
            device="cpu", 
            num_samples=0
        )
        result_zero = zero_samples_server.process_input_ids(input_ids)
        assert isinstance(result_zero, torch.Tensor)
        assert result_zero.shape == (1, 4, zero_samples_server.vocab_size)
        
        # Test with num_samples=-1 (should also return full logits)
        neg_samples_server = TeacherLogitsServer(
            "google/gemma-3-1b-it", 
            device="cpu", 
            num_samples=-1
        )
        result_neg = neg_samples_server.process_input_ids(input_ids)
        assert isinstance(result_neg, torch.Tensor)
        assert result_neg.shape == (1, 4, neg_samples_server.vocab_size)
        
        # Test batch processing in full logits mode
        batch_input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # [batch_size=2, seq_len=3]
        batch_result = full_logits_server.process_input_ids(batch_input_ids)
        assert isinstance(batch_result, torch.Tensor)
        assert batch_result.shape == (2, 2, full_logits_server.vocab_size)  # [batch_size, seq_len-1, vocab_size]
    
    def test_input_ids_processing_integration(self, server):
        """Test the integration of input_ids processing functionality (from test_input_ids_server.py)."""
        # Test tokenization to get input_ids
        text = "The quick brown fox jumps over"
        inputs = server.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # Test get_logits_from_input_ids
        logits = server.get_logits_from_input_ids(input_ids)
        expected_shape = (input_ids.shape[0], input_ids.shape[1]-1, server.vocab_size)
        assert logits.shape == expected_shape
        
        # Test process_input_ids
        result = server.process_input_ids(input_ids)
        assert isinstance(result, tuple)
        indices, values = result
        expected_indices_shape = (input_ids.shape[0], input_ids.shape[1]-1, server.num_samples)
        assert indices.shape == expected_indices_shape
        assert values.shape == expected_indices_shape
        
        # Test batch processing
        batch_input_ids = torch.cat([input_ids, input_ids], dim=0)  # Create batch of 2
        batch_logits = server.get_logits_from_input_ids(batch_input_ids)
        expected_batch_shape = (2, input_ids.shape[1]-1, server.vocab_size)
        assert batch_logits.shape == expected_batch_shape
        
        # Test the request processing function format
        input_tensor = input_ids.numpy().flatten()  # Flatten for transport
        recovered_input_ids = torch.from_numpy(input_tensor).long().view(1, -1)
        assert torch.equal(input_ids, recovered_input_ids)
    
    def test_sample_from_logits_basic(self, server):
        """Test basic sampling from logits."""
        # Create sample logits directly
        logits = torch.randn(3, server.vocab_size)  # [seq_len=3, vocab_size]
        
        num_samples = 5
        sampled_tokens, sampled_logit_values = server.sample_from_logits(
            logits, num_samples=num_samples
        )
        
        # Check shapes
        assert sampled_tokens.shape == (num_samples, logits.shape[0])
        assert sampled_logit_values.shape == (num_samples, logits.shape[0])
        
        # Check that sampled tokens are valid vocabulary indices
        assert (sampled_tokens >= 0).all()
        assert (sampled_tokens < server.vocab_size).all()
        
        # Check that logit values are finite
        assert torch.isfinite(sampled_logit_values).all()
    
    def test_sample_from_logits_temperature(self, server):
        """Test sampling with different temperatures."""
        # Create sample logits directly
        logits = torch.randn(2, server.vocab_size)  # [seq_len=2, vocab_size]
        
        # Sample with low temperature (more deterministic)
        tokens_low, _ = server.sample_from_logits(logits, num_samples=100, temperature=0.1)
        
        # Sample with high temperature (more random)
        tokens_high, _ = server.sample_from_logits(logits, num_samples=100, temperature=2.0)
        
        # Just verify that both produce valid tokens
        assert (tokens_low >= 0).all() and (tokens_low < server.vocab_size).all()
        assert (tokens_high >= 0).all() and (tokens_high < server.vocab_size).all()
    
    def test_sample_from_logits_top_k(self, server):
        """Test sampling with top-k filtering."""
        # Create sample logits directly
        logits = torch.randn(2, server.vocab_size)  # [seq_len=2, vocab_size]
        
        # Sample with top-k
        tokens, _ = server.sample_from_logits(logits, num_samples=100, top_k=10)
        
        # Check that tokens are valid
        assert (tokens >= 0).all()
        assert (tokens < server.vocab_size).all()
    
    def test_sample_from_logits_top_p(self, server):
        """Test sampling with top-p (nucleus) filtering."""
        # Create sample logits directly
        logits = torch.randn(2, server.vocab_size)  # [seq_len=2, vocab_size]
        
        # Sample with top-p
        tokens, _ = server.sample_from_logits(logits, num_samples=100, top_p=0.9)
        
        # Check that tokens are valid
        assert (tokens >= 0).all()
        assert (tokens < server.vocab_size).all()
    
    def test_process_input_ids_basic(self, server):
        """Test basic input_ids processing with sampling."""
        # Create sample input_ids
        input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])  # [batch_size=2, seq_len=5]
        
        result = server.process_input_ids(input_ids)
        
        # Should return tuple for sampling mode
        assert isinstance(result, tuple)
        indices, values = result
        
        # Check shapes
        expected_shape = (2, 4, server.num_samples)  # [batch_size, seq_len-1, num_samples]
        assert indices.shape == expected_shape
        assert values.shape == expected_shape
        
        # Check that indices are valid vocabulary indices
        assert (indices >= 0).all()
        assert (indices < server.vocab_size).all()
        
        # Check that values are finite
        assert torch.isfinite(values).all()
    
    def test_process_input_ids_single_sequence(self, server):
        """Test input_ids processing with single sequence."""
        # Create single sequence input_ids
        input_ids = torch.tensor([1, 2, 3, 4])  # [seq_len=4]
        
        result = server.process_input_ids(input_ids)
        
        # Should return tuple for sampling mode
        assert isinstance(result, tuple)
        indices, values = result
        
        # Check shapes (should add batch dimension)
        expected_shape = (1, 3, server.num_samples)  # [batch_size=1, seq_len-1, num_samples]
        assert indices.shape == expected_shape
        assert values.shape == expected_shape
        
        # Check that indices are valid vocabulary indices
        assert (indices >= 0).all()
        assert (indices < server.vocab_size).all()
        
        # Check that values are finite
        assert torch.isfinite(values).all()
    
    def test_logits_consistency_with_get_logits_from_input_ids(self, server):
        """Test that sampled logits appear in expected positions by comparing with get_logits_from_input_ids."""
        # Create input_ids directly
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # [batch_size=1, seq_len=5]
        
        # Get logits directly
        original_logits = server.get_logits_from_input_ids(input_ids)  # [batch_size, seq_len-1, vocab_size]
        
        # Get sampled indices and values
        result = server.process_input_ids(input_ids)
        assert isinstance(result, tuple)
        indices, values = result
        
        # Check a few samples to verify consistency
        batch_size, seq_len, num_samples = indices.shape
        for batch_idx in range(batch_size):
            for pos_idx in range(seq_len):
                for sample_idx in range(min(5, num_samples)):  # Check first 5 samples
                    token_idx = indices[batch_idx, pos_idx, sample_idx].item()
                    sampled_logit = values[batch_idx, pos_idx, sample_idx].item()
                    
                    # Get the original logit value at this position and token
                    original_logit = original_logits[batch_idx, pos_idx, token_idx].item()
                    
                    # They should match (within floating point precision)
                    assert abs(sampled_logit - original_logit) < 1e-4, \
                        f"Logit mismatch at pos {pos_idx}, token {token_idx}: {sampled_logit} vs {original_logit}"
    
    def test_edge_cases(self, server):
        """Test edge cases and error conditions."""
        # Test with very short sequence using input_ids
        input_ids = torch.tensor([[1, 2]])  # Very short sequence
        result = server.process_input_ids(input_ids)
        assert isinstance(result, tuple)
        indices, values = result
        assert indices.shape == (1, 1, server.num_samples)  # [batch_size=1, seq_len-1=1, num_samples]
        assert values.shape == (1, 1, server.num_samples)
        
        # Test with minimum viable input
        input_ids = torch.tensor([[1]])  # Single token
        result = server.process_input_ids(input_ids)
        assert isinstance(result, tuple)
        indices, values = result
        assert indices.shape == (1, 1, server.num_samples)  # Single token still produces output
        assert values.shape == (1, 1, server.num_samples)
    
    def test_deterministic_sampling(self, server):
        """Test that sampling with very low temperature gives more deterministic results."""
        # Create sample input_ids
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # [batch_size=1, seq_len=5]
        
        # Create a server with very low temperature
        low_temp_server = TeacherLogitsServer(
            "google/gemma-3-1b-it", 
            device="cpu", 
            temperature=0.01,
            num_samples=5
        )
        
        # Sample twice with very low temperature
        result1 = low_temp_server.process_input_ids(input_ids)
        result2 = low_temp_server.process_input_ids(input_ids)
        
        assert isinstance(result1, tuple) and isinstance(result2, tuple)
        indices1, values1 = result1
        indices2, values2 = result2
        
        # Results should have the same shape
        assert indices1.shape == indices2.shape
        assert values1.shape == values2.shape
        
        # With very low temperature, there should be some overlap in sampled tokens
        # This is probabilistic but should generally hold
        batch_size, seq_len, num_samples = indices1.shape
        
        # Count positions where at least one sample matches
        matches = 0
        for batch_idx in range(batch_size):
            for pos_idx in range(seq_len):
                tokens1 = set(indices1[batch_idx, pos_idx].tolist())
                tokens2 = set(indices2[batch_idx, pos_idx].tolist())
                if tokens1 & tokens2:  # If there's any overlap
                    matches += 1
        
        # At least some positions should have overlapping tokens with low temperature
        assert matches > 0


class TestProcessLogitsRequest:
    """Test suite for the top-level process_logits_request function."""
    
    @pytest.fixture
    def server(self):
        """Create a TeacherLogitsServer instance for testing."""
        return TeacherLogitsServer("google/gemma-3-1b-it", device="cpu")
    
    def test_process_logits_request_basic(self, server):
        """Test basic functionality of process_logits_request with sampling."""
        # Create sample input_ids
        input_ids = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int32)
        
        # Call the function
        result = process_logits_request(server, input_ids)
        
        # Verify result structure for sampling mode
        assert isinstance(result, list)
        assert len(result) == 2  # [indices, values]
        
        indices, values = result
        assert isinstance(indices, np.ndarray)
        assert isinstance(values, np.ndarray)
        
        # Check shapes
        expected_shape = (2, 4, server.num_samples)  # [batch_size, seq_len-1, num_samples]
        assert indices.shape == expected_shape
        assert values.shape == expected_shape
        
        # Check that indices are valid vocabulary indices
        assert np.all(indices >= 0)
        assert np.all(indices < server.vocab_size)
        
        # Check that values are finite
        assert np.all(np.isfinite(values))
    
    def test_process_logits_request_single_sequence(self, server):
        """Test process_logits_request with single sequence."""
        # Create single sequence input_ids
        input_ids = np.array([1, 2, 3, 4], dtype=np.int32)
        
        # Call the function
        result = process_logits_request(server, input_ids)
        
        # Verify result structure for sampling mode
        assert isinstance(result, list)
        assert len(result) == 2  # [indices, values]
        
        indices, values = result
        assert isinstance(indices, np.ndarray)
        assert isinstance(values, np.ndarray)
        
        # Check shapes (should add batch dimension)
        expected_shape = (1, 3, server.num_samples)  # [batch_size=1, seq_len-1, num_samples]
        assert indices.shape == expected_shape
        assert values.shape == expected_shape
        
        # Check that indices are valid vocabulary indices
        assert np.all(indices >= 0)
        assert np.all(indices < server.vocab_size)
        
        # Check that values are finite
        assert np.all(np.isfinite(values))
    
    def test_process_logits_request_consistency(self, server):
        """Test that process_logits_request gives consistent structure as direct method calls."""
        # Create sample input_ids
        input_ids = np.array([[1, 2, 3, 4]], dtype=np.int32)
        
        # Call the top-level function
        result_function = process_logits_request(server, input_ids)
        
        # Call the method directly
        input_ids_torch = torch.from_numpy(input_ids).long()
        result_direct = server.process_input_ids(input_ids_torch)
        
        # Both should return tuples for sampling mode
        assert isinstance(result_function, list) and len(result_function) == 2
        assert isinstance(result_direct, tuple) and len(result_direct) == 2
        
        # Compare results structure (not exact values due to sampling)
        indices_function, values_function = result_function
        indices_direct, values_direct = result_direct
        
        # Check shapes are the same
        assert indices_function.shape == indices_direct.numpy().shape
        assert values_function.shape == values_direct.numpy().shape
        
        # Check that both produce valid indices
        assert np.all(indices_function >= 0)
        assert np.all(indices_function < server.vocab_size)
        assert torch.all(indices_direct >= 0)
        assert torch.all(indices_direct < server.vocab_size)
        
        # Check that both produce finite values
        assert np.all(np.isfinite(values_function))
        assert torch.all(torch.isfinite(values_direct))
    
    def test_process_logits_request_full_logits_mode(self, server):
        """Test process_logits_request with full logits mode (no sampling)."""
        # Create server with num_samples=None for full logits
        full_logits_server = TeacherLogitsServer(
            "google/gemma-3-1b-it", 
            device="cpu", 
            num_samples=None
        )
        
        # Create sample input_ids
        input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
        
        # Call the function
        result = process_logits_request(full_logits_server, input_ids)
        
        # Verify result structure for full logits mode
        assert isinstance(result, list)
        assert len(result) == 1  # [logits]
        
        logits = result[0]
        assert isinstance(logits, np.ndarray)
        
        # Check shape
        expected_shape = (1, 4, full_logits_server.vocab_size)  # [batch_size, seq_len-1, vocab_size]
        assert logits.shape == expected_shape
        
        # Check that logits are finite
        assert np.all(np.isfinite(logits))
    
    def test_process_logits_request_error_handling(self, server):
        """Test error handling in process_logits_request."""
        # Test with invalid input (empty array)
        empty_input = np.array([])
        
        result = process_logits_request(server, empty_input)
        
        # Should return error indicators
        assert isinstance(result, list)
        assert len(result) == 2
        
        indices, values = result
        assert isinstance(indices, np.ndarray)
        assert isinstance(values, np.ndarray)
        
        # Check for error indicators
        assert indices.size == 1 and indices[0] == -1
        assert values.size == 1 and values[0] == -1


if __name__ == "__main__":
    # Run some basic tests if executed directly
    print("Running basic tests...")
    
    # Test sampling mode
    server = TeacherLogitsServer("google/gemma-3-1b-it", device="cpu")
    
    # Test basic functionality
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    print(f"Testing with input_ids: {input_ids}")
    
    # Test logits extraction
    logits = server.get_logits_from_input_ids(input_ids)
    print(f"Logits shape: {logits.shape}")
    
    # Test sampling
    sampled_tokens, sampled_logit_values = server.sample_from_logits(
        logits[0], num_samples=3, temperature=0.8
    )
    print(f"Sampled tokens shape: {sampled_tokens.shape}")
    print(f"Sampled logit values shape: {sampled_logit_values.shape}")
    
    # Test process_input_ids
    result = server.process_input_ids(input_ids)
    assert isinstance(result, tuple)
    indices, values = result
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Values shape: {values.shape}")
    
    # Test top-level process_logits_request function
    print("\nTesting top-level process_logits_request function...")
    input_ids_np = input_ids.numpy()
    result = process_logits_request(server, input_ids_np)
    indices_func, values_func = result
    print(f"Function result - Indices shape: {indices_func.shape}")
    print(f"Function result - Values shape: {values_func.shape}")
    
    # Test full logits mode
    print("\nTesting full logits mode...")
    full_logits_server = TeacherLogitsServer("google/gemma-3-1b-it", device="cpu", num_samples=None)
    full_result = full_logits_server.process_input_ids(input_ids)
    assert isinstance(full_result, torch.Tensor)
    print(f"Full logits shape: {full_result.shape}")
    
    # Test full logits mode with process_logits_request
    full_result_func = process_logits_request(full_logits_server, input_ids_np)
    assert len(full_result_func) == 1
    print(f"Full logits function result shape: {full_result_func[0].shape}")
    
    print("\nAll basic tests passed!")