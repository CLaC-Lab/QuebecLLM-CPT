import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Model paths
base_model = "croissantllm/CroissantLLMBase"
finetuned_model_path = "/home/k_ammade/Projects/QuebecCPT/CPT_scratch/quebec_croissant/checkpoint-12032"

print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load baseline model
print("Loading baseline model...")
baseline_model = AutoModelForCausalLM.from_pretrained(
    base_model, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# Load fine-tuned model
print("Loading fine-tuned model...")
base_for_ft = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)
finetuned_model = PeftModel.from_pretrained(base_for_ft, finetuned_model_path)

baseline_model.eval()
finetuned_model.eval()

generation_args = {
    "max_length": 200,
    "do_sample": True,
    "top_p": 0.9,
    "top_k": 60,
    "temperature": 0.6,
    "repetition_penalty": 1.05
}

# Test prompts targeting Quebec French capabilities
tests = [
    ("Quebec French Completion", """Je vais au dépanneur pour acheter une liqueur pis des chips. Après ça, je vais"""),
    
    ("Translation to Quebec French", """Translate to Quebec French:
"I'm going shopping at the mall this weekend."
Quebec French:"""),
    
    ("Weather Expressions", """À Montréal, quand il fait froid, on dit qu'il fait"""),
    
    ("Anglicisms Usage", """Les expressions québécoises:
"Je vais checker mes courriels" (vérifier)
"Tu peux-tu""")
]

def generate_response(model, prompt):
    """Generate response from a model given a prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        tokens = model.generate(**inputs, **generation_args)
    response = tokenizer.decode(tokens[0], skip_special_tokens=True)
    # Return only the generated part (after the prompt)
    return response[len(prompt):].strip()

print("\n" + "="*80)
print("QUEBEC FRENCH CPT EVALUATION - BASELINE vs FINE-TUNED")
print("="*80)

for i, (test_name, prompt) in enumerate(tests, 1):
    print(f"\n{i}. {test_name}")
    print("-" * 60)
    print(f"PROMPT: {prompt}")
    print("-" * 60)
    
    # Generate responses from both models
    baseline_response = generate_response(baseline_model, prompt)
    finetuned_response = generate_response(finetuned_model, prompt)
    
    print("BASELINE (Original):")
    print(f"  {baseline_response}")
    print()
    print("FINE-TUNED (Your CPT):")
    print(f"  {finetuned_response}")
    print()
    
    # Simple comparison indicators
    baseline_words = set(baseline_response.lower().split())
    finetuned_words = set(finetuned_response.lower().split())
    
    print("="*80)

print("\nEVALUATION COMPLETE!")

