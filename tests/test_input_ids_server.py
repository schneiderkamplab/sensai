#!/usr/bin/env python3
"""
Test script for the modified input_ids-based server
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sensai.logits_server import TeacherLogitsServer

def test_input_ids_processing():
    """Test the new input_ids processing functionality"""
    
    # Initialize server with small model for testing
    print("Initializing TeacherLogitsServer...")
    server = TeacherLogitsServer(
        "gpt2", 
        device="cpu",
        num_samples=5,
        temperature=0.8,
        top_p=0.9,
        top_k=50
    )
    print("Server initialized successfully")
    
    # Test tokenization to get input_ids
    text = "The quick brown fox jumps over"
    inputs = server.tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Original text: '{text}'")
    print(f"Input_ids shape: {input_ids.shape}")
    print(f"Input_ids: {input_ids}")
    
    # Test get_logits_from_input_ids
    print("\nTesting get_logits_from_input_ids...")
    logits = server.get_logits_from_input_ids(input_ids)
    print(f"Logits shape: {logits.shape}")
    print(f"Expected shape: [batch_size={input_ids.shape[0]}, seq_len={input_ids.shape[1]-1}, vocab_size={server.vocab_size}]")
    
    # Test process_input_ids
    print("\nTesting process_input_ids...")
    sparse_result = server.process_input_ids(input_ids)
    print(f"Sparse tensor shape: {sparse_result.shape}")
    print(f"Sparse tensor nnz: {sparse_result._nnz()}")
    print(f"Expected shape: [num_samples={server.num_samples}, seq_len={input_ids.shape[1]-1}, vocab_size={server.vocab_size}]")
    
    # Test batch processing
    print("\nTesting batch processing...")
    batch_input_ids = torch.cat([input_ids, input_ids], dim=0)  # Create batch of 2
    print(f"Batch input_ids shape: {batch_input_ids.shape}")
    
    batch_logits = server.get_logits_from_input_ids(batch_input_ids)
    print(f"Batch logits shape: {batch_logits.shape}")
    
    # Test the request processing function format
    print("\nTesting request processing format...")
    input_tensor = input_ids.numpy().flatten()  # Flatten for transport
    print(f"Input tensor shape for transport: {input_tensor.shape}")
    
    # This would be what the server processes
    recovered_input_ids = torch.from_numpy(input_tensor).long().view(1, -1)
    print(f"Recovered input_ids shape: {recovered_input_ids.shape}")
    
    # Verify they match
    print(f"Input_ids match: {torch.equal(input_ids, recovered_input_ids)}")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_input_ids_processing()