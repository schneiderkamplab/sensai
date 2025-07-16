"""
Tests for the collate function wrapper
"""
import pytest
import numpy as np
import warnings
from unittest.mock import Mock, patch
from sensai.utils import wrap_collate_function, _request_teacher_logits_with_retry
from sensai.client import SensAIClient


class TestWrapCollateFunction:
    """Test the wrap_collate_function functionality"""
    
    def test_basic_functionality(self):
        """Test basic collate function wrapping"""
        # Create mock client
        mock_client = Mock(spec=SensAIClient)
        mock_client.send_tensor.return_value = np.array([1, 2, 3, 4])
        
        # Create original collate function
        def original_collate(batch):
            return {
                'input_ids': np.array([[1, 2, 3], [4, 5, 6]]),
                'labels': np.array([0, 1])
            }
        
        # Wrap the collate function
        wrapped_collate = wrap_collate_function(original_collate, mock_client)
        
        # Test with dummy batch
        batch = [{}, {}]  # Dummy batch data
        result = wrapped_collate(batch)
        
        # Verify structure
        assert isinstance(result, dict)
        assert 'input_ids' in result
        assert 'labels' in result
        assert 'teacher_logits' in result
        
        # Verify teacher logits were added
        np.testing.assert_array_equal(result['teacher_logits'], np.array([1, 2, 3, 4]))
        
        # Verify client was called with correct input
        mock_client.send_tensor.assert_called_once()
        call_args = mock_client.send_tensor.call_args[0][0]
        np.testing.assert_array_equal(call_args, np.array([[1, 2, 3], [4, 5, 6]]))
    
    def test_skip_if_teacher_logits_present(self):
        """Test that function skips request if teacher logits already present"""
        mock_client = Mock(spec=SensAIClient)
        
        def original_collate(batch):
            return {
                'input_ids': np.array([[1, 2, 3]]),
                'teacher_logits': np.array([5, 6, 7])
            }
        
        wrapped_collate = wrap_collate_function(original_collate, mock_client)
        
        batch = [{}]
        result = wrapped_collate(batch)
        
        # Verify teacher logits unchanged
        np.testing.assert_array_equal(result['teacher_logits'], np.array([5, 6, 7]))
        
        # Verify client was not called
        mock_client.send_tensor.assert_not_called()
    
    def test_custom_teacher_logits_attribute(self):
        """Test using custom attribute name for teacher logits"""
        mock_client = Mock(spec=SensAIClient)
        mock_client.send_tensor.return_value = np.array([1, 2, 3])
        
        def original_collate(batch):
            return {'input_ids': np.array([[1, 2, 3]])}
        
        wrapped_collate = wrap_collate_function(
            original_collate, 
            mock_client, 
            teacher_logits_attribute='custom_logits'
        )
        
        batch = [{}]
        result = wrapped_collate(batch)
        
        assert 'custom_logits' in result
        assert 'teacher_logits' not in result
        np.testing.assert_array_equal(result['custom_logits'], np.array([1, 2, 3]))
    
    def test_torch_tensor_input(self):
        """Test handling of torch tensor inputs"""
        mock_client = Mock(spec=SensAIClient)
        mock_client.send_tensor.return_value = np.array([1, 2, 3])
        
        # Mock torch tensor
        mock_tensor = Mock()
        mock_tensor.numpy.return_value = np.array([[1, 2, 3]])
        
        def original_collate(batch):
            return {'input_ids': mock_tensor}
        
        wrapped_collate = wrap_collate_function(original_collate, mock_client)
        
        batch = [{}]
        result = wrapped_collate(batch)
        
        # Verify tensor was converted to numpy
        mock_tensor.numpy.assert_called_once()
        mock_client.send_tensor.assert_called_once()
    
    def test_invalid_batch_format(self):
        """Test error handling for invalid batch format"""
        mock_client = Mock(spec=SensAIClient)
        
        def original_collate(batch):
            return ["not", "a", "dict"]  # Invalid format
        
        wrapped_collate = wrap_collate_function(original_collate, mock_client)
        
        batch = [{}]
        with pytest.raises(ValueError, match="Collate function must return a dictionary"):
            wrapped_collate(batch)
    
    def test_missing_input_ids(self):
        """Test error handling when input_ids is missing"""
        mock_client = Mock(spec=SensAIClient)
        
        def original_collate(batch):
            return {'labels': np.array([0, 1])}  # Missing input_ids
        
        wrapped_collate = wrap_collate_function(original_collate, mock_client)
        
        batch = [{}]
        with pytest.raises(ValueError, match="Batch must contain 'input_ids' key"):
            wrapped_collate(batch)
    
    def test_invalid_input_ids_type(self):
        """Test error handling for invalid input_ids type"""
        mock_client = Mock(spec=SensAIClient)
        
        def original_collate(batch):
            return {'input_ids': "invalid_type"}
        
        wrapped_collate = wrap_collate_function(original_collate, mock_client)
        
        batch = [{}]
        with pytest.raises(ValueError, match="input_ids must be numpy array or torch tensor"):
            wrapped_collate(batch)


class TestRequestTeacherLogitsWithRetry:
    """Test the retry logic for teacher logits requests"""
    
    def test_successful_request(self):
        """Test successful request on first try"""
        mock_client = Mock(spec=SensAIClient)
        mock_client.send_tensor.return_value = np.array([1, 2, 3])
        
        input_tensor = np.array([[1, 2, 3]])
        result = _request_teacher_logits_with_retry(mock_client, input_tensor, 3, 0.1)
        
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))
        mock_client.send_tensor.assert_called_once_with(input_tensor)
    
    def test_retry_on_failure(self):
        """Test retry logic when request fails"""
        mock_client = Mock(spec=SensAIClient)
        mock_client.send_tensor.side_effect = [
            Exception("Network error"),
            Exception("Server error"),
            np.array([1, 2, 3])  # Success on third try
        ]
        
        input_tensor = np.array([[1, 2, 3]])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _request_teacher_logits_with_retry(mock_client, input_tensor, 3, 0.01)
        
        # Verify success
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))
        
        # Verify retries occurred
        assert mock_client.send_tensor.call_count == 3
        
        # Verify warnings were issued
        assert len(w) == 2  # Two retry warnings
        assert "Network error" in str(w[0].message)
        assert "Server error" in str(w[1].message)
    
    def test_all_retries_fail(self):
        """Test when all retries fail"""
        mock_client = Mock(spec=SensAIClient)
        mock_client.send_tensor.side_effect = Exception("Persistent error")
        
        input_tensor = np.array([[1, 2, 3]])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _request_teacher_logits_with_retry(mock_client, input_tensor, 2, 0.01)
        
        # Verify failure
        assert result is None
        
        # Verify all attempts made
        assert mock_client.send_tensor.call_count == 3  # Initial + 2 retries
        
        # Verify warnings
        assert len(w) == 3  # 2 retry warnings + 1 final failure warning
        assert "failed after 3 attempts" in str(w[2].message)
    
    def test_server_error_indicator(self):
        """Test handling of server error indicator"""
        mock_client = Mock(spec=SensAIClient)
        mock_client.send_tensor.return_value = np.array([-1])  # Error indicator
        
        input_tensor = np.array([[1, 2, 3]])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _request_teacher_logits_with_retry(mock_client, input_tensor, 2, 0.01)
        
        # Verify failure
        assert result is None
        
        # Verify error was detected
        assert "Server returned error indicator" in str(w[2].message)
    
    def test_none_result(self):
        """Test handling when server returns None"""
        mock_client = Mock(spec=SensAIClient)
        mock_client.send_tensor.return_value = None
        
        input_tensor = np.array([[1, 2, 3]])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _request_teacher_logits_with_retry(mock_client, input_tensor, 1, 0.01)
        
        assert result is None
        assert "Server returned None" in str(w[1].message)
    
    def test_empty_tensor(self):
        """Test handling when server returns empty tensor"""
        mock_client = Mock(spec=SensAIClient)
        mock_client.send_tensor.return_value = np.array([])
        
        input_tensor = np.array([[1, 2, 3]])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _request_teacher_logits_with_retry(mock_client, input_tensor, 1, 0.01)
        
        assert result is None
        assert "Server returned empty tensor" in str(w[1].message)
    
    def test_invalid_result_type(self):
        """Test handling when server returns invalid type"""
        mock_client = Mock(spec=SensAIClient)
        mock_client.send_tensor.return_value = "invalid_type"
        
        input_tensor = np.array([[1, 2, 3]])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _request_teacher_logits_with_retry(mock_client, input_tensor, 1, 0.01)
        
        assert result is None
        assert "Server returned invalid type" in str(w[1].message)


class TestIntegrationWithRetry:
    """Integration tests for collate function with retry logic"""
    
    def test_collate_function_with_retry_success(self):
        """Test collate function with successful retry"""
        mock_client = Mock(spec=SensAIClient)
        mock_client.send_tensor.side_effect = [
            Exception("Temporary failure"),
            np.array([1, 2, 3, 4])  # Success on retry
        ]
        
        def original_collate(batch):
            return {'input_ids': np.array([[1, 2, 3]])}
        
        wrapped_collate = wrap_collate_function(
            original_collate, 
            mock_client, 
            num_retries=3, 
            retry_delay=0.01
        )
        
        batch = [{}]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = wrapped_collate(batch)
        
        # Verify success
        assert 'teacher_logits' in result
        np.testing.assert_array_equal(result['teacher_logits'], np.array([1, 2, 3, 4]))
        
        # Verify retry occurred
        assert mock_client.send_tensor.call_count == 2
        assert len(w) == 1  # One retry warning
    
    def test_collate_function_with_retry_failure(self):
        """Test collate function when all retries fail"""
        mock_client = Mock(spec=SensAIClient)
        mock_client.send_tensor.side_effect = Exception("Persistent failure")
        
        def original_collate(batch):
            return {'input_ids': np.array([[1, 2, 3]])}
        
        wrapped_collate = wrap_collate_function(
            original_collate, 
            mock_client, 
            num_retries=2, 
            retry_delay=0.01
        )
        
        batch = [{}]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = wrapped_collate(batch)
        
        # Verify batch still processed but without teacher logits
        assert 'input_ids' in result
        assert 'teacher_logits' not in result
        
        # Verify all retries attempted
        assert mock_client.send_tensor.call_count == 3  # Initial + 2 retries
        
        # Verify warnings
        assert len(w) == 4  # 2 retry warnings + 1 final failure + 1 collate warning
        assert "Failed to obtain teacher logits" in str(w[3].message)
        assert "Continuing without teacher logits" in str(w[3].message)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])