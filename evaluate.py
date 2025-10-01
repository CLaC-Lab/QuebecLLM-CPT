#!/usr/bin/env python
"""
Evaluation script for the Quebec French model
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from typing import List
import argparse
from inference import QuebecFrenchGenerator
from prepare_data import QuebecFrenchDataPreparer


class QuebecFrenchEvaluator:
    """Evaluate the fine-tuned model on Quebec French text"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = self.load_model()
    
    def load_model(self):
        """Load model and tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        
        return model, tokenizer
    
    def calculate_perplexity(self, texts: List[str], batch_size: int = 8) -> float:
        """Calculate perplexity on a list of texts"""
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Calculating perplexity"):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**encodings, labels=encodings.input_ids)
                
                # Accumulate loss
                loss = outputs.loss.item()
                num_tokens = (encodings.attention_mask == 1).sum().item()
                
                total_loss += loss * num_tokens
                total_tokens += num_tokens
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def evaluate_file(self, file_path: str) -> dict:
        """Evaluate model on a text file"""
        # Load texts
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Evaluating on {len(texts)} texts from {file_path}")
        
        # Calculate metrics
        perplexity = self.calculate_perplexity(texts)
        
        # Calculate average text length
        avg_length = np.mean([len(self.tokenizer.encode(t)) for t in texts[:100]])
        
        results = {
            "file": file_path,
            "num_texts": len(texts),
            "perplexity": perplexity,
            "avg_token_length": avg_length
        }
        
        return results
    
    def compare_models(self, base_model_name: str, test_file: str):
        """Compare fine-tuned model with base model"""
        print("Evaluating fine-tuned model...")
        finetuned_results = self.evaluate_file(test_file)
        
        print(f"\nLoading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Temporarily replace model
        original_model = self.model
        original_tokenizer = self.tokenizer
        self.model = base_model
        self.tokenizer = base_tokenizer
        
        print("Evaluating base model...")
        base_results = self.evaluate_file(test_file)
        
        # Restore original model
        self.model = original_model
        self.tokenizer = original_tokenizer
        
        # Compare results
        print("\n=== Evaluation Results ===")
        print(f"Base Model Perplexity: {base_results['perplexity']:.2f}")
        print(f"Fine-tuned Model Perplexity: {finetuned_results['perplexity']:.2f}")
        print(f"Improvement: {(base_results['perplexity'] - finetuned_results['perplexity']) / base_results['perplexity'] * 100:.1f}%")
        
        return {
            "base_model": base_results,
            "finetuned_model": finetuned_results
        }


# ============= Main execution scripts =============

def prepare_data_main():
    """Main function for data preparation"""
    parser = argparse.ArgumentParser(description="Prepare Quebec French data for training")
    parser.add_argument("--input", type=str, required=True, help="Input text file")
    parser.add_argument("--output", type=str, default="./data", help="Output directory")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--min_length", type=int, default=10, help="Minimum text length")
    parser.add_argument("--max_length", type=int, default=10000, help="Maximum text length")
    
    args = parser.parse_args()
    
    preparer = QuebecFrenchDataPreparer(
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    train_file, val_file = preparer.process_file(
        args.input,
        args.output,
        args.val_split
    )
    
    print(f"\nData preparation complete!")
    print(f"Train file: {train_file}")
    print(f"Validation file: {val_file}")


def inference_main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description="Generate Quebec French text")
    parser.add_argument("--model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--prompt", type=str, help="Input prompt")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--batch_file", type=str, help="File with prompts for batch generation")
    parser.add_argument("--output", type=str, default="generated.json", help="Output file for batch generation")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Model uses LoRA")
    
    args = parser.parse_args()
    
    generator = QuebecFrenchGenerator(
        model_path=args.model,
        use_lora=args.use_lora
    )
    
    if args.interactive:
        generator.interactive_generation()
    elif args.batch_file:
        with open(args.batch_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        generator.batch_generate(prompts, args.output)
    elif args.prompt:
        generated = generator.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature
        )
        print("\n--- Generated Text ---")
        for text in generated:
            print(text)
    else:
        print("Please provide --prompt, --interactive, or --batch_file")


def evaluate_main():
    """Main function for evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate Quebec French model")
    parser.add_argument("--model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--test_file", type=str, required=True, help="Test file path")
    parser.add_argument("--base_model", type=str, help="Base model for comparison")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    evaluator = QuebecFrenchEvaluator(model_path=args.model)
    
    if args.base_model:
        results = evaluator.compare_models(args.base_model, args.test_file)
    else:
        results = evaluator.evaluate_file(args.test_file)
        print(f"\nPerplexity: {results['perplexity']:.2f}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")