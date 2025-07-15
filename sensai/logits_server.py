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
    import argparse
    import os
    import tempfile
    from .server import SensAIServer
    from .transports import NamedPipeTransport, SharedMemoryTransport
    
    def create_transport(transport_type: str, **kwargs):
        """Create transport based on type and arguments"""
        if transport_type == "named_pipe":
            pipe_dir = kwargs.get("pipe_dir", tempfile.mkdtemp(prefix="sensai_teacher_"))
            num_clients = kwargs.get("num_clients", 4)
            return NamedPipeTransport(pipe_dir=pipe_dir, num_clients=num_clients), pipe_dir
        
        elif transport_type == "shared_memory":
            shm_path = kwargs.get("shm_path", "/tmp/sensai_teacher_shm")
            num_clients = kwargs.get("num_clients", 4)
            max_elems = kwargs.get("max_elems", 1000000)
            max_dtype = kwargs.get("max_dtype", np.float32)
            return SharedMemoryTransport(
                shm_path=shm_path, 
                num_clients=num_clients, 
                max_elems=max_elems, 
                max_dtype=max_dtype
            ), shm_path
        
        else:
            raise ValueError(f"Unknown transport type: {transport_type}")
    
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
    parser.add_argument("--max-elems", type=int, default=1000000,
                        help="Maximum elements per tensor for shared memory")
    parser.add_argument("--interval", type=float, default=0.001,
                        help="Server polling interval in seconds")
    
    args = parser.parse_args()
    
    # Initialize teacher logits server
    teacher_server = TeacherLogitsServer(args.model, device=args.device)
    print(f"TeacherLogitsServer initialized with model: {args.model}")
    
    # Create transport based on arguments
    transport_kwargs = {
        "num_clients": args.num_clients,
        "pipe_dir": args.pipe_dir,
        "shm_path": args.shm_path,
        "max_elems": args.max_elems,
        "max_dtype": np.float32
    }
    
    transport, cleanup_path = create_transport(args.transport, **transport_kwargs)
    print(f"Transport created: {args.transport} with {args.num_clients} client slots")
    
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
            
            print(f"Processing request: type={request_type}, text='{text}', samples={num_samples}")
            
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
                    # Return tensor info and indices/values
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
            print(f"Error processing request: {e}")
            return np.array([-1])  # Error indicator
    
    # Create and run server
    sensai_server = SensAIServer(transport)
    print(f"SensAIServer started with {transport.num_clients} client slots")
    print("Server is running... Press Ctrl+C to stop")
    
    try:
        sensai_server.run_loop(process_logits_request, interval=args.interval)
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