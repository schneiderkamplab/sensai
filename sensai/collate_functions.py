####################
# WORK IN PROGRESS #
####################

from .client import SensAIClient

def wrap_collate_function(collate_fn, sensai_client: SensAIClient, teacher_logits_attribute: str = 'teacher_logits'):
    """
    Wraps a collate function to ensure it is compatible with the SensAI framework.
    
    Args:
        collate_fn (callable): The original collate function to be wrapped.
        
    Returns:
        callable: A wrapped collate function that can be used with SensAI.
    """
    def wrapped_collate(batch):
        # Call the original collate function
        batch = collate_fn(batch)
        # Batch may consist of token ids, attention masks, and labels

        assert isinstance(batch, dict), "Collate function must be a dictionary."
        if teacher_logits_attribute not in batch:
            # If teacher logits are present, ensure they are in the correct format

            # FIXME: This doesn't work yet, as the server expects some kwargs before the data
            
            result = sensai_client.send_tensor(
                batch['input_ids'] 
            )
            # TODO: error handling if result is None or not as expected
            batch['teacher_logits'] = result
        return batch

    return wrapped_collate