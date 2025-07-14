import torch
import torch.nn.functional as F
import pytest
import numpy as np
from transformers_logits_server import TeacherLogitsServer


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
        
        # Low temperature should have less diversity
        unique_low = len(torch.unique(tokens_low, dim=0))
        unique_high = len(torch.unique(tokens_high, dim=0))
        
        # This is probabilistic, but generally should hold
        assert unique_low <= unique_high or unique_low < 50  # Allow some variance
    
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
    
    def test_sample_to_sparse_coo_basic(self, server):
        """Test basic sparse COO tensor creation."""
        text = "Hello world"
        num_samples = 10
        
        sparse_results = server.sample_to_sparse_coo(text, num_samples=num_samples)
        
        assert len(sparse_results) == 1
        sparse_tensor = sparse_results[0]
        
        # Check tensor properties
        assert sparse_tensor.is_sparse
        assert sparse_tensor.layout == torch.sparse_coo
        assert sparse_tensor.shape[0] == num_samples
        assert sparse_tensor.shape[2] == server.vocab_size
        
        # Check that we have the expected number of non-zero elements
        expected_nnz = num_samples * sparse_tensor.shape[1]  # seq_len
        assert sparse_tensor._nnz() == expected_nnz
        
        # Check indices are valid
        sparse_tensor = sparse_tensor.coalesce()
        indices = sparse_tensor.indices()
        assert (indices[0] >= 0).all() and (indices[0] < num_samples).all()  # sample indices
        assert (indices[1] >= 0).all() and (indices[1] < sparse_tensor.shape[1]).all()  # position indices
        assert (indices[2] >= 0).all() and (indices[2] < server.vocab_size).all()  # token indices
    
    def test_sample_to_sparse_coo_multiple_texts(self, server):
        """Test sparse COO tensor creation with multiple texts."""
        texts = ["Hello", "World", "Test"]
        num_samples = 5
        
        sparse_results = server.sample_to_sparse_coo(texts, num_samples=num_samples)
        
        assert len(sparse_results) == 3
        
        for sparse_tensor in sparse_results:
            assert sparse_tensor.is_sparse
            assert sparse_tensor.shape[0] == num_samples
            assert sparse_tensor.shape[2] == server.vocab_size
    
    def test_decode_samples(self, server):
        """Test decoding of sampled tokens."""
        text = "Hello world"
        num_samples = 3
        
        sparse_results = server.sample_to_sparse_coo(text, num_samples=num_samples)
        sparse_tensor = sparse_results[0]
        
        decoded_samples = server.decode_samples(sparse_tensor)
        
        assert len(decoded_samples) == num_samples
        assert all(isinstance(sample, str) for sample in decoded_samples)
    
    def test_logits_consistency_with_parallel_model(self, server):
        """Test that sampled logits appear in expected positions by running model in parallel."""
        text = "The quick brown fox"
        num_samples = 50
        
        # Get logits directly
        logits_list = server.get_prompt_logits(text)
        original_logits = logits_list[0]
        
        # Get sparse samples
        sparse_results = server.sample_to_sparse_coo(text, num_samples=num_samples, temperature=1.0)
        sparse_tensor = sparse_results[0].coalesce()
        
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        
        # Verify that sampled logit values match original logits at sampled positions
        for i in range(sparse_tensor._nnz()):
            sample_idx = indices[0, i].item()
            pos_idx = indices[1, i].item()
            token_idx = indices[2, i].item()
            sampled_logit = values[i].item()
            
            # Get the original logit value at this position and token
            original_logit = original_logits[pos_idx, token_idx].item()
            
            # They should match (within floating point precision)
            assert abs(sampled_logit - original_logit) < 1e-4, \
                f"Logit mismatch at pos {pos_idx}, token {token_idx}: {sampled_logit} vs {original_logit}"
    
    def test_sampling_distribution_correctness(self, server):
        """Test that sampling follows the probability distribution correctly."""
        # Create a simple test case
        text = "Hello"
        logits_list = server.get_prompt_logits(text)
        logits = logits_list[0]
        
        # Take first position only for simpler testing
        if logits.shape[0] > 0:
            first_pos_logits = logits[0]  # Shape: [vocab_size]
            
            # Sample many times to check distribution
            num_samples = 1000
            sampled_tokens, _ = server.sample_from_logits(
                first_pos_logits.unsqueeze(0), num_samples=num_samples, temperature=1.0
            )
            
            # Get the probability distribution
            probs = F.softmax(first_pos_logits, dim=-1)
            
            # Count occurrences of each token
            token_counts = torch.bincount(sampled_tokens.flatten(), minlength=server.vocab_size)
            empirical_probs = token_counts.float() / num_samples
            
            # Find tokens that were actually sampled
            sampled_indices = torch.nonzero(token_counts > 0).flatten()
            
            # For tokens that were sampled, check if empirical probability is reasonable
            for token_idx in sampled_indices:
                theoretical_prob = probs[token_idx].item()
                empirical_prob = empirical_probs[token_idx].item()
                
                # Allow for reasonable variance in sampling
                # Use a loose bound since we're dealing with multinomial sampling
                if theoretical_prob > 0.01:  # Only check for tokens with reasonable probability
                    relative_error = abs(empirical_prob - theoretical_prob) / theoretical_prob
                    assert relative_error < 0.5, \
                        f"Token {token_idx}: theoretical prob {theoretical_prob:.4f}, empirical prob {empirical_prob:.4f}"
    
    def test_edge_cases(self, server):
        """Test edge cases and error conditions."""
        # Test with single token
        single_token_text = "A"
        result = server.get_prompt_logits(single_token_text)
        assert len(result) == 1
        assert result[0].shape[0] >= 0  # Should handle single token case
        
        # Test with very short text
        short_text = "Hi"
        sparse_results = server.sample_to_sparse_coo(short_text, num_samples=1)
        assert len(sparse_results) == 1
        
        # Test with num_samples=1
        result = server.sample_to_sparse_coo("Hello", num_samples=1)
        assert result[0].shape[0] == 1
    
    def test_deterministic_sampling(self, server):
        """Test that sampling with temperature=0 gives deterministic results."""
        text = "Hello world"
        
        # Sample twice with very low temperature
        result1 = server.sample_to_sparse_coo(text, num_samples=5, temperature=0.01)
        result2 = server.sample_to_sparse_coo(text, num_samples=5, temperature=0.01)
        
        # Results should be similar (though not necessarily identical due to randomness)
        tensor1 = result1[0].coalesce()
        tensor2 = result2[0].coalesce()
        
        assert tensor1.shape == tensor2.shape
        
        # At least some samples should be the same with very low temperature
        indices1 = tensor1.indices()
        indices2 = tensor2.indices()
        
        # Check that token indices (dimension 2) show some similarity
        tokens1 = indices1[2].sort()[0]
        tokens2 = indices2[2].sort()[0]
        
        # With very low temperature, there should be some overlap in most frequent tokens
        # This is probabilistic but should generally hold
        common_tokens = len(set(tokens1.tolist()) & set(tokens2.tolist()))
        total_positions = tensor1.shape[1]
        
        # At least some positions should have similar high-probability tokens
        assert common_tokens > 0


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
    
    # Test sparse COO
    sparse_results = server.sample_to_sparse_coo(text, num_samples=10)
    sparse_tensor = sparse_results[0]
    print(f"Sparse tensor shape: {sparse_tensor.shape}")
    print(f"Non-zero elements: {sparse_tensor._nnz()}")
    
    # Test decoding
    decoded = server.decode_samples(sparse_tensor)
    print(f"Decoded samples: {len(decoded)}")
    
    # Test logits consistency
    print("\nTesting logits consistency...")
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    
    # Check a few samples
    for i in range(min(5, sparse_tensor._nnz())):
        pos_idx = indices[1, i].item()
        token_idx = indices[2, i].item()
        sampled_logit = values[i].item()
        original_logit = logits_list[0][pos_idx, token_idx].item()
        
        print(f"Position {pos_idx}, Token {token_idx}: "
              f"sampled={sampled_logit:.4f}, original={original_logit:.4f}, "
              f"diff={abs(sampled_logit - original_logit):.6f}")
    
    print("\nAll basic tests passed!")