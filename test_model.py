#!/usr/bin/env python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import StoppingCriteria, StoppingCriteriaList
from dataclasses import dataclass
from typing import Optional, Dict, Any

class StopOnSequences(StoppingCriteria):
    def __init__(self, stop_seqs: list[torch.Tensor]):
        self.stop_seqs = stop_seqs
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0]
        for pat in self.stop_seqs:
            L = pat.shape[0]
            if L <= seq.shape[0] and torch.equal(seq[-L:], pat.to(seq.device)):
                return True
        return False

def make_im_end_stops(tokenizer, model_device):
    """Create stopping criteria for <|im_end|> tokens"""
    cands = ["<|im_end|>", "<|im_end|>\n", "\n<|im_end|>"]
    pats = []
    for c in cands:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if ids:
            pats.append(torch.tensor(ids, device=model_device))
    return StoppingCriteriaList([StopOnSequences(pats)]) if pats else None


@dataclass
class GenerationConfig:
    """Configuration for text generation with sensible defaults"""
    max_new_tokens: int = 200
    min_new_tokens: int = 20  # Reduced from 48 - don't force long responses
    temperature: float = 0.1
    top_p: float = 0.9  # Slightly more focused than 0.95
    top_k: int = 50  # Add top-k for additional control
    repetition_penalty: float = 1.05  # Actually penalize repetition (>1.0)
    no_repeat_ngram_size: int = 4  # Less restrictive than 3
    do_sample: bool = True
    use_cache: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model.generate()"""
        return {
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "do_sample": self.do_sample,
            "use_cache": self.use_cache,
        }


# Different configs for different response types
CONFIGS = {
    "default": GenerationConfig()
}


def generate_response(
    model, 
    tokenizer,
    user_message: str,
    system_msg: Optional[str] = None,
    config_name: str = "default",
    debug: bool = False
) -> str:
    """Generate a response using specified configuration"""
    
    if system_msg is None:
        system_msg = (
            "Tu es un assistant amical et naturel. Tu peux répondre informellement."
            "Sois concis mais informatif, utilise 2-4 phrases selon le contexte."
        )
    
    # Build the conversation
    msgs = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_message}
    ]
    
    # Apply chat template
    prompt_text = tokenizer.apply_chat_template(
        msgs, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize
    enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    # Get generation config
    config = CONFIGS.get(config_name, CONFIGS["default"])
    gen_kwargs = config.to_dict()
    
    # Add model-specific parameters
    gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
    gen_kwargs["pad_token_id"] = tokenizer.eos_token_id
    gen_kwargs["stopping_criteria"] = make_im_end_stops(tokenizer, model.device)
    
    # Generate
    with torch.no_grad():
        gen = model.generate(**enc, **gen_kwargs)
    
    # Debug output if requested
    if debug:
        raw = tokenizer.decode(gen[0], skip_special_tokens=False)
        print("\n[DEBUG - Raw Output Tail]")
        print(raw[-400:])
        print("[/DEBUG]")
    
    # Extract response (remove prompt and stop tokens)
    out_ids = gen[0, enc["input_ids"].shape[-1]:]
    text = tokenizer.decode(out_ids, skip_special_tokens=False)
    
    # Clean up - remove everything after <|im_end|>
    response = text.split("<|im_end|>")[0].strip()
    
    return response


def main():
    # Model paths
    base_model = "croissantllm/CroissantLLMChat-v0.1"
    finetuned_model_path = "/home/o_vanesb/QuebecLLM-CPT/models/checkpoint-2094-croissant-3-epochs" #"/home/k_ammade/CPT_scratch/quebec_croissant_chat_ALL_DATA_6EPOCHS/checkpoint-8376"
    
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
    
    # Test cases with appropriate config selection
    tests = [
        # Casual greeting - should trigger informal Quebec response
        ("Casual Greeting", 
         "TÂCHE: Question à choix multiple sur le français québécois.\n\nCONSIGNE: Choisir la bonne réponse parmi les options (0 à 9).\nRépondre UNIQUEMENT avec le chiffre correspondant.\n\nÉNONCÉ: Ayoye!\n\nOPTIONS:\n0) Ce mot exprime la joie lorsqu’on reçoit un cadeau inattendu.\n1) On utilise «Ayoye!» pour demander de l’aide dans une situation difficile.\n2) On l’emploie pour dire au revoir de façon amicale entre amis proches.\n3) C’est une salutation matinale typique entre voisins québécois.\n4) «Ayoye!» est utilisé pour féliciter quelqu’un après une bonne nouvelle.\n5) «Ayoye!» est souvent dit pour signaler qu’on a compris une blague.\n6) Mélange de \" Aïe! \" et \" Ouille! \". On l'utilise pour exprimer l'étonnement ou la douleur.\n7) On l’emploie pour indiquer qu’on a faim ou qu’on attend un repas.\n8) C’est un mot qu’on dit pour encourager quelqu’un à essayer encore une fois.\n9) C’est un cri de guerre traditionnel lors des matchs de hockey au Québec.\n\nRéponse (chiffre seulement):"),
        
        ("Casual Greeting", 
         "Pourrais-tu me décrire la différence entre un chat et un chien?"),
        
        # Story about winter - should trigger Quebec winter vocabulary
        ("Winter Story", 
         "Raconte-moi ta pire journée d'hiver. Il faisait vraiment froid et tout allait mal."),
        
        # Weekend plans - should use 'fin de semaine' and informal language
        ("Weekend Plans", 
         "Qu'est-ce que tu fais en fin de semaine? Moi je vais probablement relaxer."),
        
        # Car problems - should trigger 'char' instead of 'voiture'
        ("Car Trouble", 
         "Mon auto est brisée encore. C'est la troisième fois ce mois-ci!"),
        
        # Shopping - should use 'magasiner' instead of 'faire du shopping'
        ("Shopping Plans", 
         "Je dois aller acheter des vêtements pour l'hiver. Tu connais des bons endroits?"),
        
        # Food/meal discussion - should use Quebec meal terms naturally
        ("Meal Planning", 
         "Qu'est-ce qu'on mange à midi? J'ai vraiment faim!"),
        
        # Dating/relationships - should use 'blonde/chum' naturally
        ("Relationship Talk", 
         "Mon ami sort avec quelqu'un depuis 6 mois. C'est sérieux entre eux!"),
        
        # Weather complaint - should trigger Quebec expressions
        ("Weather Complaint", 
         "Il fait tellement mauvais dehors! Je suis tanné de ce temps-là."),
        
        # Party/social - should use Quebec party vocabulary
        ("Party Planning", 
         "On organise une fête samedi. Apporte quelque chose à boire!"),
        
        # Work frustration - should trigger informal Quebec expressions
        ("Work Rant", 
         "J'en ai assez de mon travail. Mon patron est vraiment difficile."),
        
        # Directions - should use Quebec location terms
        ("Asking Directions", 
         "Je cherche un endroit pour prendre un café. Y'a quelque chose proche d'ici?"),
        
        # Tech problems - should use Quebec anglicisms naturally
        ("Computer Issues", 
         "Mon ordinateur fonctionne plus. Je pense que j'ai un virus."),
        
        # Morning routine - should trigger 'à matin'
        ("Morning Talk",
         "Comment ça s'est passé ce matin? T'as bien dormi?"),
        
        # Informal story request
        ("Story Request",
         "Raconte-moi quelque chose de drôle qui t'est arrivé."),
        
        # Money/expense complaint - should trigger Quebec expressions
        ("Expense Complaint",
         "Tout coûte tellement cher maintenant! C'est rendu ridicule."),
        
        ("Poutine",
         "Où est-ce que je peux trouver la meilleure poutine en ville?"),
    ]
    
    print("\n" + "="*80)
    print("QUEBEC FRENCH CPT EVALUATION - IMPROVED GENERATION SETTINGS")
    print("="*80)
    
    for i, (test_name, user_message) in enumerate(tests, 1):
        print(f"\n{i}. {test_name}")
        print("-" * 60)
        print(f"USER: {user_message}")
        print("-" * 60)
        
        try:
            # Generate with both models
            baseline_response = generate_response(
                baseline_model, 
                tokenizer,
                user_message, 
                # config_name=config_name
            )
            
            finetuned_response = generate_response(
                finetuned_model,
                tokenizer,
                user_message, 
                # config_name=config_name
            )
            
            print("BASELINE:")
            print(f"  {baseline_response}")
            print()
            
            print("FINE-TUNED:")
            print(f"  {finetuned_response}")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    # Additional test with different temperatures
    print("\n" + "="*80)
    print("TEMPERATURE COMPARISON TEST")
    print("="*80)
    
    test_prompt = "Raconte-moi une courte histoire québécoise."
    
    for temp in [0.3, 0.5, 0.7, 0.9]:
        print(f"\nTemperature: {temp}")
        print("-" * 40)
        
        response = generate_response(
            finetuned_model,
            tokenizer,
            test_prompt,
        )
        print(f"  {response}")
    
    print("\n✅ EVALUATION COMPLETE!")


if __name__ == "__main__":
    main()