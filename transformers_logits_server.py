import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Union, Optional, Tuple
import numpy as np


class TeacherLogitsServer:
    """
    A wrapper around HuggingFace Transformers that extracts logits from prompt tokens and samples according to probabilities.
    Returns sparse COO tensors for efficient storage of sampled logits.
    """
    
    def __init__(self, model_name: str, device: str = "auto", **model_kwargs):
        """
        Initialize the HuggingFace model and tokenizer.
        
        Args:
            model_name: Name or path of the model to load
            device: Device to load model on ("auto", "cuda", "cpu")
            **model_kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            **model_kwargs
        )
        
        if self.device.type != "cuda" or "device_map" not in model_kwargs:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        self.vocab_size = self.model.config.vocab_size
        
    def get_prompt_logits(self, texts: Union[str, List[str]]) -> List[torch.Tensor]:
        """
        Extract logits for prompt tokens without generation using efficient batching.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            List of tensors, each with shape [seq_len, vocab_size] containing logits
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Handle empty texts
        if not texts:
            return []
        
        logits_list = []
        
        with torch.no_grad():
            # Tokenize all texts in batch with padding
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                return_attention_mask=True
            ).to(self.device)
            
            # Get model outputs for the entire batch
            outputs = self.model(**inputs)
            batch_logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
            
            # Process each text in the batch
            for i in range(len(texts)):
                # Get attention mask for this sequence to find actual length
                attention_mask = inputs.attention_mask[i]
                actual_length = attention_mask.sum().item()
                
                if actual_length == 0:
                    # Empty sequence
                    logits_list.append(torch.empty(0, self.vocab_size))
                    continue
                
                # Extract logits for this sequence
                sequence_logits = batch_logits[i]  # Shape: [seq_len, vocab_size]
                
                # Only keep logits for non-padded positions
                sequence_logits = sequence_logits[:actual_length]
                
                # For causal LM, we want logits for positions 0 to seq_len-1
                # where logits[i] predicts token at position i+1
                if sequence_logits.shape[0] > 1:
                    # Use logits for positions 0 to seq_len-1 (predicting next tokens)
                    prompt_logits = sequence_logits[:-1]  # Remove last position
                else:
                    # Single token case
                    prompt_logits = sequence_logits
                
                logits_list.append(prompt_logits.cpu())
        
        return logits_list
    
    def sample_from_logits(self, 
                          logits: torch.Tensor, 
                          num_samples: int = 1,
                          temperature: float = 1.0,
                          top_p: float = 1.0,
                          top_k: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample tokens from logits according to their probabilities.
        
        Args:
            logits: Tensor of shape [seq_len, vocab_size] or [vocab_size]
            num_samples: Number of samples to draw per position
            temperature: Temperature for sampling (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (0 = disabled)
            
        Returns:
            Tuple of (sampled_token_ids, sampled_logit_values)
            - sampled_token_ids: Tensor of shape [num_samples, seq_len] or [num_samples]
            - sampled_logit_values: Tensor of shape [num_samples, seq_len] or [num_samples]
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)  # Add sequence dimension
        
        seq_len, vocab_size = logits.shape
        original_logits = logits.clone()  # Keep original logits for value extraction
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, vocab_size)
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample
        samples = torch.multinomial(probs.view(-1, vocab_size), num_samples, replacement=True)
        sampled_token_ids = samples.view(num_samples, seq_len).squeeze() if seq_len == 1 else samples.view(num_samples, seq_len)
        
        # Extract corresponding logit values
        if sampled_token_ids.dim() == 1:
            sampled_token_ids_expanded = sampled_token_ids.unsqueeze(0)
        else:
            sampled_token_ids_expanded = sampled_token_ids
            
        # Get logit values for sampled tokens
        sampled_logit_values = torch.gather(
            original_logits.unsqueeze(0).expand(sampled_token_ids_expanded.shape[0], -1, -1),
            dim=-1,
            index=sampled_token_ids_expanded.unsqueeze(-1)
        ).squeeze(-1)
        
        if seq_len == 1:
            sampled_logit_values = sampled_logit_values.squeeze()
        
        return sampled_token_ids, sampled_logit_values
    
    def sample_to_sparse_coo(self, 
                            texts: Union[str, List[str]],
                            num_samples: int = 1,
                            temperature: float = 1.0,
                            top_p: float = 1.0,
                            top_k: int = 0) -> List[torch.Tensor]:
        """
        Main function: Extract logits and return sampled results as sparse COO tensors.
        
        Args:
            texts: Input text(s) to process
            num_samples: Number of samples per position
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            
        Returns:
            List of sparse COO tensors, one per input text.
            Each tensor has shape [num_samples, seq_len, vocab_size] with logit values at sampled positions.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Get logits for all texts
        logits_list = self.get_prompt_logits(texts)
        
        sparse_results = []
        for logits in logits_list:
            if logits.numel() == 0:
                # Empty sequence - return empty sparse tensor
                sparse_results.append(torch.sparse_coo_tensor(
                    torch.empty((3, 0), dtype=torch.long),
                    torch.empty(0),
                    (num_samples, 0, self.vocab_size)
                ))
                continue
            
            seq_len = logits.shape[0]
            
            # Sample tokens and get their logit values
            sampled_tokens, sampled_logit_values = self.sample_from_logits(
                logits, num_samples, temperature, top_p, top_k
            )
            
            if sampled_tokens.dim() == 1:
                sampled_tokens = sampled_tokens.unsqueeze(0)
                sampled_logit_values = sampled_logit_values.unsqueeze(0)
            
            # Convert to sparse COO tensor
            # Create indices for sparse tensor
            sample_indices = torch.arange(num_samples).unsqueeze(1).expand(-1, seq_len)
            position_indices = torch.arange(seq_len).unsqueeze(0).expand(num_samples, -1)
            
            # Flatten indices
            sample_flat = sample_indices.flatten()
            position_flat = position_indices.flatten()
            token_flat = sampled_tokens.flatten()
            logit_values_flat = sampled_logit_values.flatten()
            
            # Stack indices: [sample_dim, position_dim, vocab_dim]
            indices = torch.stack([sample_flat, position_flat, token_flat])
            values = logit_values_flat.float()
            
            # Create sparse COO tensor
            sparse_tensor = torch.sparse_coo_tensor(
                indices, values, (num_samples, seq_len, self.vocab_size)
            )
            
            sparse_results.append(sparse_tensor)
        
        return sparse_results
    
    def decode_samples(self, sparse_tensor: torch.Tensor) -> List[str]:
        """
        Decode sampled tokens back to text.
        
        Args:
            sparse_tensor: Sparse COO tensor from sample_to_sparse_coo
            
        Returns:
            List of decoded text samples
        """
        sparse_tensor = sparse_tensor.coalesce()
        indices = sparse_tensor.indices()
        num_samples = sparse_tensor.shape[0]
        
        # Group indices by sample
        decoded_samples = []
        for sample_idx in range(num_samples):
            sample_mask = indices[0] == sample_idx
            sample_positions = indices[1][sample_mask]
            sample_tokens = indices[2][sample_mask]
            
            # Sort by position
            _, sort_indices = torch.sort(sample_positions)
            sorted_tokens = sample_tokens[sort_indices]
            
            # Decode tokens
            decoded_text = self.tokenizer.decode(sorted_tokens.tolist(), skip_special_tokens=True)
            decoded_samples.append(decoded_text)
        
        return decoded_samples


# Example usage
if __name__ == "__main__":
    # Initialize server
    server = TeacherLogitsServer("microsoft/DialoGPT-medium")
    
    # Sample text
    text = "The quick brown fox"
    
    # Get logits first to show the sampling process
    logits_list = server.get_prompt_logits(text)
    logits = logits_list[0]
    print(f"Original logits shape: {logits.shape}")
    print(f"Sample logits for first position: {logits[0][:10]}...")  # Show first 10 logits
    
    # Sample tokens and show the process
    sampled_tokens, sampled_logit_values = server.sample_from_logits(logits, num_samples=3, temperature=0.8, top_p=0.9)
    print(f"\nSampled token indices: {sampled_tokens}")
    print(f"\nSampled logits values:")
    for i in range(sampled_tokens.shape[0]):
        print(f"Sample {i+1}: {sampled_logit_values[i].tolist()}")
    
    # Get sparse COO tensor with samples (test with 256 samples for distillation)
    sparse_results = server.sample_to_sparse_coo(
        text, 
        num_samples=256,
        temperature=0.8,
        top_p=0.9
    )
    
    # Print results
    sparse_tensor = sparse_results[0]
    print(f"\nSparse tensor shape: {sparse_tensor.shape}")
    print(f"Number of non-zero elements: {sparse_tensor._nnz()}")
    print(f"Sparsity: {1 - sparse_tensor._nnz() / sparse_tensor.numel():.4f}")
    
    # Decode samples
    decoded = server.decode_samples(sparse_tensor)
    print(f"\nDecoded samples:")
    for i, sample in enumerate(decoded):
        print(f"Sample {i+1}: {sample}")