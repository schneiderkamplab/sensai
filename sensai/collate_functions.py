import warnings
import time
import numpy as np
from typing import Optional
from .client import SensAIClient

def wrap_collate_function(
    collate_fn, 
    sensai_client: SensAIClient, 
    teacher_logits_attribute: str = 'teacher_logits',
    num_retries: int = 3,
    retry_delay: float = 0.1
):
    """
    Wraps a collate function to add teacher logits from SensAI server with retry logic.
    
    Args:
        collate_fn (callable): The original collate function to be wrapped.
        sensai_client (SensAIClient): The SensAI client for requesting teacher logits.
        teacher_logits_attribute (str): Name of the attribute to store teacher logits in batch.
        num_retries (int): Number of retry attempts if request fails (default: 3).
        retry_delay (float): Delay in seconds between retry attempts (default: 0.1).
        
    Returns:
        callable: A wrapped collate function that requests teacher logits and handles errors.
    """
    def wrapped_collate(batch):
        # Call the original collate function
        batch = collate_fn(batch)
        
        # Validate batch format
        if not isinstance(batch, dict):
            raise ValueError("Collate function must return a dictionary.")
        
        if 'input_ids' not in batch:
            raise ValueError("Batch must contain 'input_ids' key for teacher logits request.")
        
        # Skip if teacher logits already present
        if teacher_logits_attribute in batch:
            return batch
        
        # Request teacher logits with retry logic
        input_ids = batch['input_ids']
        
        # Convert to numpy if needed
        if hasattr(input_ids, 'numpy'):
            input_tensor = input_ids.numpy()
        elif isinstance(input_ids, np.ndarray):
            input_tensor = input_ids
        else:
            raise ValueError(f"input_ids must be numpy array or torch tensor, got {type(input_ids)}")
        
        teacher_logits = _request_teacher_logits_with_retry(
            sensai_client, 
            input_tensor, 
            num_retries, 
            retry_delay
        )
        
        if teacher_logits is not None:
            batch[teacher_logits_attribute] = teacher_logits
        else:
            warnings.warn(
                f"Failed to obtain teacher logits after {num_retries} retries. "
                f"Continuing without teacher logits.",
                RuntimeWarning
            )
        
        return batch

    return wrapped_collate


def _request_teacher_logits_with_retry(
    client: SensAIClient,
    input_tensor: np.ndarray,
    num_retries: int,
    retry_delay: float
) -> Optional[np.ndarray]:
    """
    Request teacher logits with retry logic.
    
    Args:
        client (SensAIClient): The SensAI client.
        input_tensor (np.ndarray): Input tensor to send to server.
        num_retries (int): Number of retry attempts.
        retry_delay (float): Delay between retries in seconds.
        
    Returns:
        Optional[np.ndarray]: Teacher logits if successful, None if all retries failed.
    """
    for attempt in range(num_retries + 1):  # +1 for initial attempt
        try:
            result = client.send_tensor(input_tensor)
            
            # Validate result
            if result is None:
                raise ValueError("Server returned None")
            
            if not isinstance(result, np.ndarray):
                raise ValueError(f"Server returned invalid type: {type(result)}")
            
            if result.size == 0:
                raise ValueError("Server returned empty tensor")
            
            # Check for error indicator
            if result.size == 1 and result[0] == -1:
                raise ValueError("Server returned error indicator")
            
            # Success
            return result
            
        except Exception as e:
            if attempt < num_retries:
                warnings.warn(
                    f"Teacher logits request failed (attempt {attempt + 1}/{num_retries + 1}): {e}. "
                    f"Retrying in {retry_delay}s...",
                    RuntimeWarning
                )
                time.sleep(retry_delay)
            else:
                warnings.warn(
                    f"Teacher logits request failed after {num_retries + 1} attempts. "
                    f"Last error: {e}",
                    RuntimeWarning
                )
    
    return None