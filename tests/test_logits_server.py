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
    
    def test_get_prompt_logits_single_text(self, server):
        """Test get_prompt_logits with a single text input."""
        text = "The quick brown fox"
        logits_list = server.get_prompt_logits(text)
        
        assert len(logits_list) == 1
        logits = logits_list[0]
        
        # Check shape
        assert logits.dim() == 2
        assert logits.shape[1] == server.vocab_size
        
        # Check that we have reasonable number of tokens
        assert logits.shape[0] > 0
        assert logits.shape[0] < 20  # Should be less than 20 tokens for this text
        
        # Check that logits are finite
        assert torch.isfinite(logits).all()
    
    def test_get_prompt_logits_multiple_texts(self, server):
        """Test get_prompt_logits with multiple text inputs."""
        texts = ["Hello world", "The quick brown fox", "A"]
        logits_list = server.get_prompt_logits(texts)
        
        assert len(logits_list) == 3
        
        # Check each logits tensor
        for i, logits in enumerate(logits_list):
            assert logits.dim() == 2
            assert logits.shape[1] == server.vocab_size
            assert logits.shape[0] > 0
            assert torch.isfinite(logits).all()
    
    def test_get_prompt_logits_empty_input(self, server):
        """Test get_prompt_logits with empty input."""
        result = server.get_prompt_logits([])
        assert result == []
    
    def test_sample_from_logits_basic(self, server):
        """Test basic sampling from logits."""
        text = "The quick brown"
        logits_list = server.get_prompt_logits(text)
        logits = logits_list[0]
        
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
        text = "Hello"
        logits_list = server.get_prompt_logits(text)
        logits = logits_list[0]
        
        # Sample with low temperature (more deterministic)
        tokens_low, _ = server.sample_from_logits(logits, num_samples=100, temperature=0.1)
        
        # Sample with high temperature (more random)
        tokens_high, _ = server.sample_from_logits(logits, num_samples=100, temperature=2.0)
        
        # Just verify that both produce valid tokens
        assert (tokens_low >= 0).all() and (tokens_low < server.vocab_size).all()
        assert (tokens_high >= 0).all() and (tokens_high < server.vocab_size).all()
    
    def test_sample_from_logits_top_k(self, server):
        """Test sampling with top-k filtering."""
        text = "Hello"
        logits_list = server.get_prompt_logits(text)
        logits = logits_list[0]
        
        # Sample with top-k
        tokens, _ = server.sample_from_logits(logits, num_samples=100, top_k=10)
        
        # Check that tokens are valid
        assert (tokens >= 0).all()
        assert (tokens < server.vocab_size).all()
    
    def test_sample_from_logits_top_p(self, server):
        """Test sampling with top-p (nucleus) filtering."""
        text = "Hello"
        logits_list = server.get_prompt_logits(text)
        logits = logits_list[0]
        
        # Sample with top-p
        tokens, _ = server.sample_from_logits(logits, num_samples=100, top_p=0.9)
        
        # Check that tokens are valid
        assert (tokens >= 0).all()
        assert (tokens < server.vocab_size).all()
    
    def test_process_input_ids_basic(self, server):
        """Test basic input_ids processing."""
        # Create sample input_ids
        input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])  # [batch_size=2, seq_len=5]
        
        indices, values = server.process_input_ids(input_ids)
        
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
        
        indices, values = server.process_input_ids(input_ids)
        
        # Check shapes (should add batch dimension)
        expected_shape = (1, 3, server.num_samples)  # [batch_size=1, seq_len-1, num_samples]
        assert indices.shape == expected_shape
        assert values.shape == expected_shape
        
        # Check that indices are valid vocabulary indices
        assert (indices >= 0).all()
        assert (indices < server.vocab_size).all()
        
        # Check that values are finite
        assert torch.isfinite(values).all()
    
    def test_logits_consistency_with_parallel_model(self, server):
        """Test that sampled logits appear in expected positions by running model in parallel."""
        text = "The quick brown fox"
        
        # Get logits directly
        logits_list = server.get_prompt_logits(text)
        original_logits = logits_list[0]
        
        # Create input_ids for the same text
        input_ids = server.tokenizer(text, return_tensors="pt").input_ids
        
        # Get sampled indices and values
        indices, values = server.process_input_ids(input_ids)
        
        # Check a few samples to verify consistency
        batch_size, seq_len, num_samples = indices.shape
        for batch_idx in range(batch_size):
            for pos_idx in range(seq_len):
                for sample_idx in range(min(5, num_samples)):  # Check first 5 samples
                    token_idx = indices[batch_idx, pos_idx, sample_idx].item()
                    sampled_logit = values[batch_idx, pos_idx, sample_idx].item()
                    
                    # Get the original logit value at this position and token
                    original_logit = original_logits[pos_idx, token_idx].item()
                    
                    # They should match (within floating point precision)
                    assert abs(sampled_logit - original_logit) < 1e-4, \
                        f"Logit mismatch at pos {pos_idx}, token {token_idx}: {sampled_logit} vs {original_logit}"
    
    
    def test_edge_cases(self, server):
        """Test edge cases and error conditions."""
        # Test with single token
        single_token_text = "A"
        result = server.get_prompt_logits(single_token_text)
        assert len(result) == 1
        assert result[0].shape[0] >= 0  # Should handle single token case
        
        # Test with very short text using input_ids
        input_ids = torch.tensor([[1, 2]])  # Very short sequence
        indices, values = server.process_input_ids(input_ids)
        assert indices.shape == (1, 1, server.num_samples)  # [batch_size=1, seq_len-1=1, num_samples]
        assert values.shape == (1, 1, server.num_samples)
        
        # Test with minimum viable input
        input_ids = torch.tensor([[1]])  # Single token
        indices, values = server.process_input_ids(input_ids)
        assert indices.shape == (1, 1, server.num_samples)  # Single token still produces output
        assert values.shape == (1, 1, server.num_samples)
    
    def test_deterministic_sampling(self, server):
        """Test that sampling with very low temperature gives more deterministic results."""
        text = "Hello world"
        input_ids = server.tokenizer(text, return_tensors="pt").input_ids
        
        # Create a server with very low temperature
        low_temp_server = TeacherLogitsServer(
            "google/gemma-3-1b-it", 
            device="cpu", 
            temperature=0.01,
            num_samples=5
        )
        
        # Sample twice with very low temperature
        indices1, values1 = low_temp_server.process_input_ids(input_ids)
        indices2, values2 = low_temp_server.process_input_ids(input_ids)
        
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
        """Test basic functionality of process_logits_request."""
        # Create sample input_ids
        input_ids = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int32)
        
        # Call the function
        result = process_logits_request(server, input_ids)
        
        # Verify result structure
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
        
        # Verify result structure
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
        indices_direct, values_direct = server.process_input_ids(input_ids_torch)
        
        # Compare results structure (not exact values due to sampling)
        indices_function, values_function = result_function
        
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
    
    server = TeacherLogitsServer("google/gemma-3-1b-it", device="cpu")
    
    # Test basic functionality
    text = "The quick brown fox"
    print(f"Testing with text: '{text}'")
    
    # Test logits extraction
    logits_list = server.get_prompt_logits(text)
    print(f"Logits shape: {logits_list[0].shape}")
    
    # Test sampling
    sampled_tokens, sampled_logit_values = server.sample_from_logits(
        logits_list[0], num_samples=3, temperature=0.8
    )
    print(f"Sampled tokens shape: {sampled_tokens.shape}")
    print(f"Sampled logit values shape: {sampled_logit_values.shape}")
    
    # Test process_input_ids
    input_ids = server.tokenizer(text, return_tensors="pt").input_ids
    indices, values = server.process_input_ids(input_ids)
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
    
    # Verify consistency (structure only, not exact values due to sampling)
    print("Verifying structural consistency between direct method and function...")
    print(f"Direct method shapes: indices={indices.shape}, values={values.shape}")
    print(f"Function shapes: indices={indices_func.shape}, values={values_func.shape}")
    assert indices_func.shape == indices.shape
    assert values_func.shape == values.shape
    print("✓ Shapes match!")
    
    # Verify both produce valid outputs
    assert np.all(indices_func >= 0) and np.all(indices_func < server.vocab_size)
    assert np.all(np.isfinite(values_func))
    print("✓ Both produce valid outputs!")
    
    # Test logits consistency
    print("\nTesting logits consistency...")
    batch_size, seq_len, num_samples = indices.shape
    
    # Check a few samples
    for batch_idx in range(batch_size):
        for pos_idx in range(min(3, seq_len)):
            for sample_idx in range(min(2, num_samples)):
                token_idx = indices[batch_idx, pos_idx, sample_idx].item()
                sampled_logit = values[batch_idx, pos_idx, sample_idx].item()
                original_logit = logits_list[0][pos_idx, token_idx].item()
                
                print(f"Batch {batch_idx}, Position {pos_idx}, Sample {sample_idx}, Token {token_idx}: "
                      f"sampled={sampled_logit:.4f}, original={original_logit:.4f}, "
                      f"diff={abs(sampled_logit - original_logit):.6f}")
    
    print("\nAll basic tests passed!")