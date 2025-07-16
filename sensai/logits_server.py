import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Tuple
import numpy as np


class TeacherLogitsServer:
    """
    A wrapper around HuggingFace Transformers that extracts logits from prompt tokens and samples according to probabilities.
    Returns separate indices and values tensors for efficient storage of sampled logits.
    """
    
    def __init__(self, 
                 model_name: str, 
                 device: str = "auto", 
                 num_samples: Optional[int] = 256,
                 temperature: float = 1.0,
                 top_p: float = 1.0,
                 top_k: int = 0,
                 **model_kwargs):
        """
        Initialize the HuggingFace model and tokenizer.
        
        Args:
            model_name: Name or path of the model to load
            device: Device to load model on ("auto", "cuda", "cpu")
            num_samples: Number of samples to draw per position (None/0/-1 = return full logits)
            temperature: Temperature for sampling (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (0 = disabled)
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
        
        # Store sampling parameters
        # Convert num_samples=None/0/-1 to None for full logits mode
        if num_samples is None or num_samples == 0 or num_samples == -1:
            self.num_samples = None
        else:
            self.num_samples = num_samples
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
    
    def get_logits_from_input_ids(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract logits from input_ids tensor directly.
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_len] containing token IDs
            attention_mask: Optional tensor of shape [batch_size, seq_len] containing attention mask.
                          If None, assumes all tokens are valid (no padding)
            
        Returns:
            Tensor of shape [batch_size, seq_len-1, vocab_size] containing logits for prompt tokens
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension
        
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided (assume all tokens are valid, no padding)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            # Move to device
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Get model outputs
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            batch_logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
            
            # For causal LM, we want logits for positions 0 to seq_len-1
            # where logits[i] predicts token at position i+1
            if seq_len > 1:
                # Use logits for positions 0 to seq_len-1 (predicting next tokens)
                prompt_logits = batch_logits[:, :-1, :]  # Remove last position
            else:
                # Single token case
                prompt_logits = batch_logits
            
            return prompt_logits.cpu()
    
    def process_input_ids(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process input_ids and return logits tensor.
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_len] containing token IDs
            attention_mask: Optional tensor of shape [batch_size, seq_len] containing attention mask.
                          If None, assumes all tokens are valid (no padding)
            
        Returns:
            If num_samples is None: Tensor of shape [batch_size, seq_len-1, vocab_size] containing full logits
            If num_samples is set: Tuple of (indices, values) where:
            - indices: Tensor of shape [batch_size, seq_len-1, num_samples] containing token indices
            - values: Tensor of shape [batch_size, seq_len-1, num_samples] containing logit values
        """
        # Get logits from input_ids
        logits = self.get_logits_from_input_ids(input_ids, attention_mask)  # [batch_size, seq_len-1, vocab_size]
        
        # If num_samples is None, return full logits
        if self.num_samples is None:
            return logits
        
        # Otherwise, proceed with sampling
        batch_size, seq_len, vocab_size = logits.shape
        
        # Initialize output tensors
        all_indices = torch.zeros(batch_size, seq_len, self.num_samples, dtype=torch.long)
        all_values = torch.zeros(batch_size, seq_len, self.num_samples, dtype=torch.float32)
        
        for batch_idx in range(batch_size):
            sequence_logits = logits[batch_idx]  # [seq_len, vocab_size]
            
            if sequence_logits.numel() == 0:
                # Empty sequence - indices and values already initialized to zeros
                continue
            else:
                # Sample tokens and get their logit values
                sampled_tokens, sampled_logit_values = self.sample_from_logits(
                    sequence_logits, self.num_samples, self.temperature, self.top_p, self.top_k
                )
                
                # Reshape to [seq_len, num_samples] if needed
                if sampled_tokens.dim() == 1:
                    # Single position case: [num_samples] -> [1, num_samples]
                    sampled_tokens = sampled_tokens.unsqueeze(0)
                    sampled_logit_values = sampled_logit_values.unsqueeze(0)
                else:
                    # Multiple positions case: [num_samples, seq_len] -> [seq_len, num_samples]
                    sampled_tokens = sampled_tokens.transpose(0, 1)
                    sampled_logit_values = sampled_logit_values.transpose(0, 1)
                
                # Store in output tensors
                all_indices[batch_idx, :seq_len, :] = sampled_tokens
                all_values[batch_idx, :seq_len, :] = sampled_logit_values
        
        return all_indices, all_values
    
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
        sampled_token_ids = samples.view(num_samples, seq_len)
        
        # Get logit values for sampled tokens
        sampled_logit_values = torch.gather(
            original_logits.unsqueeze(0).expand(num_samples, -1, -1),
            dim=-1,
            index=sampled_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Handle single sequence case
        if seq_len == 1:
            sampled_token_ids = sampled_token_ids.squeeze(-1)  # Remove seq_len dimension
            sampled_logit_values = sampled_logit_values.squeeze(-1)  # Remove seq_len dimension
        
        return sampled_token_ids, sampled_logit_values


def process_logits_request(teacher_server: TeacherLogitsServer, input_data) -> List[np.ndarray]:
    """
    Process a logits request. Input can be either a single tensor (input_ids) or list of tensors (input_ids, attention_mask).
    
    Args:
        teacher_server: TeacherLogitsServer instance to process the request
        input_data: Either:
                   - Single NumPy array containing input_ids (token IDs) for backward compatibility
                   - List of NumPy arrays [input_ids, attention_mask] for dual tensor input
        
    Returns:
        If num_samples is None: List containing [logits] as NumPy array
        If num_samples is set: List containing [indices, values] as NumPy arrays
    """
    try:
        # Handle both single tensor and list of tensors
        if isinstance(input_data, np.ndarray):
            # Single tensor case - assume it's input_ids
            input_ids = torch.from_numpy(input_data).long()
            attention_mask = None
        elif isinstance(input_data, list) and len(input_data) >= 1:
            # List of tensors case
            input_ids = torch.from_numpy(input_data[0]).long()
            attention_mask = torch.from_numpy(input_data[1]).long() if len(input_data) > 1 else None
        else:
            raise ValueError("Input must be either a numpy array or list of numpy arrays")
        
        # Handle different input shapes for input_ids
        if input_ids.dim() == 1:
            # Single sequence: [seq_len] -> [1, seq_len]
            input_ids = input_ids.unsqueeze(0)
        elif input_ids.dim() > 2:
            # Flatten to 2D: [batch_size, seq_len]
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        
        # Handle different input shapes for attention_mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 1:
                # Single sequence: [seq_len] -> [1, seq_len]
                attention_mask = attention_mask.unsqueeze(0)
            elif attention_mask.dim() > 2:
                # Flatten to 2D: [batch_size, seq_len]
                attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        
        batch_size, seq_len = input_ids.shape
        print(f"Processing input_ids: batch_size={batch_size}, seq_len={seq_len}")
        if attention_mask is not None:
            print(f"Processing attention_mask: shape={attention_mask.shape}")
        
        # Process input_ids with optional attention_mask
        result = teacher_server.process_input_ids(input_ids, attention_mask)
        
        if teacher_server.num_samples is None:
            # Return full logits
            logits_np = result.numpy()
            return [logits_np]
        else:
            # Return indices and values
            indices, values = result
            indices_np = indices.numpy()
            values_np = values.numpy()
            return [indices_np, values_np]
            
    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        return [np.array([-1]), np.array([-1])]  # Error indicators


# Example usage
if __name__ == "__main__":
    import argparse
    import os
    from .server import SensAIServer
    from .utils import create_transport
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Teacher Logits Server")
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it", 
                        help="Model name or path to load")
    parser.add_argument("--device", type=str, default="auto", 
                        help="Device to load model on (auto, cuda, cpu)")
    parser.add_argument("--transport", type=str, default="named_pipe", 
                        choices=["named_pipe", "shared_memory"],
                        help="Transport type to use")
    parser.add_argument("--num-clients", type=int, default=4, 
                        help="Number of client slots")
    parser.add_argument("--pipe-dir", type=str, 
                        help="Directory for named pipes (auto-generated if not specified)")
    parser.add_argument("--shm-path", type=str, default="/tmp/sensai_teacher_shm",
                        help="Path for shared memory file")
    parser.add_argument("--max-nbytes", type=int, default=1000000000,
                        help="Maximum bytes for shared memory buffer")
    parser.add_argument("--max-tensors", type=int, default=4,
                        help="Maximum number of tensors per message")
    parser.add_argument("--max-dims", type=int, default=8,
                        help="Maximum number of dimensions per tensor")
    parser.add_argument("--interval", type=float, default=0.001,
                        help="Server polling interval in seconds")
    
    # Sampling parameters
    parser.add_argument("--num-samples", type=int, default=256,
                        help="Number of samples to draw per position")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling (higher = more random)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Nucleus sampling threshold")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-k sampling (0 = disabled)")
    
    args = parser.parse_args()
    
    # Initialize teacher logits server with sampling parameters
    teacher_server = TeacherLogitsServer(
        args.model, 
        device=args.device,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    print(f"TeacherLogitsServer initialized with model: {args.model}")
    print(f"Sampling config: num_samples={args.num_samples}, temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    
    # Create transport based on arguments
    transport_kwargs = {
        "num_clients": args.num_clients,
        "pipe_dir": args.pipe_dir,
        "shm_path": args.shm_path,
        "max_nbytes": args.max_nbytes,
        "max_tensors": args.max_tensors,
        "max_dims": args.max_dims
    }
    
    transport, cleanup_path = create_transport(args.transport, **transport_kwargs)
    print(f"Transport created: {args.transport} with {args.num_clients} client slots")
    
    # Create request processor function that uses the teacher_server
    def request_processor(input_tensor: np.ndarray) -> List[np.ndarray]:
        return process_logits_request(teacher_server, input_tensor)
    
    # Create and run server
    sensai_server = SensAIServer(transport)
    print(f"SensAIServer started with {transport.num_clients} client slots")
    print("Server is running... Press Ctrl+C to stop")
    
    try:
        sensai_server.run_loop(request_processor, interval=args.interval)
    except KeyboardInterrupt:
        print("\nServer stopped")
    finally:
        # Clean up resources
        if args.transport == "named_pipe":
            import shutil
            shutil.rmtree(cleanup_path, ignore_errors=True)
        elif args.transport == "shared_memory":
            try:
                if hasattr(transport, 'close'):
                    transport.close()
            except:
                pass
            # Clean up shared memory file
            try:
                os.unlink(cleanup_path)
            except:
                pass