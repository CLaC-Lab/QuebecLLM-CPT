#!/usr/bin/env python
# ============= inference.py =============
"""
Inference script for the fine-tuned Quebec French model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from typing import List, Optional


class QuebecFrenchGenerator:
    """Generate Quebec French text using the fine-tuned model"""
    
    def __init__(self, model_path: str, use_lora: bool = True, device: str = "cuda"):
        self.model_path = model_path
        self.use_lora = use_lora
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.model, self.tokenizer = self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        print(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.use_lora:
            # Load base model
            from peft import AutoPeftModelForCausalLM
            model = AutoPeftModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # Load full model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        model.eval()
        return model, tokenizer
    
    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        num_return_sequences: int = 1,
        repetition_penalty: float = 1.2,
        early_stopping: bool = False,
        do_sample: bool = True
    ) -> List[str]:
        """Generate text from prompt"""
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                early_stopping=early_stopping 
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove the prompt from the generated text
            text = text[len(prompt):].strip()
            generated_texts.append(text)
        
        return generated_texts
    
    def interactive_generation(self):
        """Interactive text generation loop"""
        print("\n=== Quebec French Text Generator ===")
        print("Enter 'quit' to exit\n")
        
        while True:
            prompt = input("\nPrompt: ").strip()
            if prompt.lower() == 'quit':
                break
            
            if not prompt:
                print("Please enter a prompt")
                continue
            
            print("\nGenerating...")
            generated = self.generate(prompt, max_length=150)
            
            print("\n--- Generated Text ---")
            for i, text in enumerate(generated, 1):
                if len(generated) > 1:
                    print(f"\n[{i}]")
                print(text)
    
    def batch_generate(self, prompts: List[str], output_file: str):
        """Generate text for multiple prompts and save to file"""
        results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"Processing prompt {i}/{len(prompts)}: {prompt[:50]}...")
            generated = self.generate(prompt, max_length=200)
            results.append({
                "prompt": prompt,
                "generated": generated[0]
            })
        
        # Save results
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Continual Pretraining for LLaMA")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="cpt-model", required=True)
    parser.add_argument("--use_lora", type=str, default=True)
    parser.add_argument("--device", type=str, default="cuda")

    # Inference arguments
    parser.add_argument("--prompt", type=str, default="Le capitale du Québec est")
    parser.add_argument("--max_length", type=str, default=200)
    parser.add_argument("--temperature", type=str, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--early_stopping", type=bool, default=True)
    
    args = parser.parse_args()

    qfg = QuebecFrenchGenerator(args.model_path, args.use_lora, args.device)
    texts = qfg.generate(
        args.prompt, 
        max_length=args.max_length, 
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        early_stopping=args.early_stopping
    )
    print(args.prompt + " " + texts[0])


if __name__ == "__main__":
    main()