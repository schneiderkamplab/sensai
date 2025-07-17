#!/usr/bin/env python3
"""
Student training script using SensAI teacher logits for knowledge distillation.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset

# Add the parent directory to the path to import sensai
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sensai.client import SensAIClient
from sensai.transports import SharedMemoryClient
from sensai.utils import wrap_collate_function


def create_simple_dataset(tokenizer, max_length=32, num_samples=100):
    """Create a simple text dataset for training"""
    # Use a small subset of a public dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    def tokenize_function(examples):
        # Tokenize the text
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze()
        }
    
    # Filter out empty texts and take a small subset
    filtered_dataset = dataset.filter(lambda x: len(x["text"].strip()) > 10)
    small_dataset = filtered_dataset.select(range(min(num_samples, len(filtered_dataset))))
    
    # Tokenize
    tokenized_dataset = small_dataset.map(tokenize_function, batched=False)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    return tokenized_dataset


def distillation_loss(student_logits, teacher_indices, teacher_values, labels, temperature=3.0, alpha=0.5):
    """
    Compute knowledge distillation loss using sparse teacher logits.
    
    Args:
        student_logits: Logits from student model [batch_size, seq_len, vocab_size]
        teacher_indices: Sampled token indices from teacher [batch_size, seq_len-1, num_samples]
        teacher_values: Sampled logit values from teacher [batch_size, seq_len-1, num_samples]
        labels: Ground truth token IDs [batch_size, seq_len]
        temperature: Temperature for softmax
        alpha: Weighting factor for distillation loss
    
    Returns:
        Combined loss (distillation + task loss)
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    
    # Task loss (standard cross-entropy)
    student_logits_flat = student_logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    task_loss = F.cross_entropy(student_logits_flat, labels_flat, ignore_index=-100)
    
    # For KL divergence, we need to align teacher and student logits
    # Teacher logits are for positions 0 to seq_len-1, student logits are for positions 0 to seq_len-1 too
    # We'll use the student logits for positions 0 to seq_len-1 to match teacher
    if seq_len > 1:
        student_logits_for_kl = student_logits[:, :-1, :]  # Remove last position to match teacher
    else:
        student_logits_for_kl = student_logits
    
    # Reshape for efficient computation
    student_logits_kl_flat = student_logits_for_kl.reshape(-1, vocab_size)  # [batch_size * (seq_len-1), vocab_size]
    teacher_indices_flat = teacher_indices.reshape(-1, teacher_indices.shape[-1])  # [batch_size * (seq_len-1), num_samples]
    teacher_values_flat = teacher_values.reshape(-1, teacher_values.shape[-1])  # [batch_size * (seq_len-1), num_samples]
    
    # Compute KL divergence efficiently using sparse teacher logits
    # We'll use the teacher samples to compute a sparse KL divergence
    student_log_probs = F.log_softmax(student_logits_kl_flat / temperature, dim=-1)
    
    # For each position, gather student log probabilities at teacher-sampled positions
    # teacher_indices_flat: [batch_size * (seq_len-1), num_samples]
    # student_log_probs: [batch_size * (seq_len-1), vocab_size]
    
    # Gather student log probabilities at teacher-sampled indices
    student_log_probs_sampled = torch.gather(
        student_log_probs, 
        dim=1, 
        index=teacher_indices_flat
    )  # [batch_size * (seq_len-1), num_samples]
    
    # Convert teacher logit values to probabilities
    teacher_probs_sampled = F.softmax(teacher_values_flat / temperature, dim=-1)
    
    # Compute KL divergence using the sampled positions
    # KL(P || Q) = sum(P * log(P/Q))
    # For sparse case: approximate by using only the sampled positions
    kl_loss = torch.sum(teacher_probs_sampled * (torch.log(teacher_probs_sampled + 1e-8) - student_log_probs_sampled))
    kl_loss = kl_loss / (batch_size * (seq_len - 1))  # Normalize by batch size and sequence length
    
    # Combine losses
    total_loss = alpha * (temperature ** 2) * kl_loss + (1 - alpha) * task_loss
    
    return total_loss, task_loss, kl_loss


def default_collate_fn(batch):
    """Default collate function for the dataset"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def main():
    print("Starting student training with teacher logits...")
    
    # Configuration
    model_name = "google/gemma-3-1b-pt"  # Student model
    max_length = 32
    batch_size = 2
    num_epochs = 1
    num_samples = 20  # Small dataset for demo
    learning_rate = 5e-5
    temperature = 3.0
    alpha = 0.5
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading student model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    student_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        attn_implementation='eager'
    )
    if device.type == "cpu":
        student_model = student_model.to(device)
    
    # Create dataset
    print("Creating dataset...")
    dataset = create_simple_dataset(tokenizer, max_length, num_samples)
    
    # Set up SensAI client
    print("Setting up SensAI client...")
    shm_path = "/tmp/sensai_teacher_shm"
    transport = SharedMemoryClient(
        path=shm_path, 
        slot_id=0,
    )
    client = SensAIClient(transport)
    
    # Wrap the collate function
    wrapped_collate = wrap_collate_function(
        default_collate_fn,
        client,
        teacher_logits_attribute="teacher_logits",
        num_retries=3,
        retry_delay=0.1
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=wrapped_collate
    )
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("Starting training...")
    student_model.train()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        total_loss = 0.0
        total_task_loss = 0.0
        total_kl_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Check if we have teacher logits (new format: indices and values)
            if "teacher_logits_indices" in batch and "teacher_logits_values" in batch:
                teacher_indices = torch.from_numpy(batch["teacher_logits_indices"]).to(device)
                teacher_values = torch.from_numpy(batch["teacher_logits_values"]).to(device)
                print(f"Step {step + 1}: Using teacher logits (indices: {teacher_indices.shape}, values: {teacher_values.shape})")
            else:
                teacher_indices = None
                teacher_values = None
                print(f"Step {step + 1}: No teacher logits available")
            
            # Forward pass through student
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids  # Use input_ids as labels for language modeling
            )
            
            student_logits = student_outputs.logits
            
            # Compute loss
            if teacher_indices is not None and teacher_values is not None:
                # Knowledge distillation loss using sparse teacher logits
                loss, task_loss, kl_loss = distillation_loss(
                    student_logits, teacher_indices, teacher_values, input_ids, temperature, alpha
                )
                total_task_loss += task_loss.item()
                total_kl_loss += kl_loss.item()
            else:
                # Standard task loss only
                loss = student_outputs.loss
                task_loss = loss
                kl_loss = torch.tensor(0.0)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            print(f"  Step {step + 1}: Loss = {loss.item():.4f}")
            if teacher_indices is not None and teacher_values is not None:
                print(f"    Task Loss = {task_loss.item():.4f}, KL Loss = {kl_loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        avg_task_loss = total_task_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        
        print(f"Epoch {epoch + 1} completed:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Task Loss: {avg_task_loss:.4f}")
        print(f"  Average KL Loss: {avg_kl_loss:.4f}")
    
    print("\nTraining completed!")
    
    # Clean up
    try:
        transport.close()
    except:
        pass


if __name__ == "__main__":
    main()