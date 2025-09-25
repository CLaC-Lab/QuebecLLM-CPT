from datasets import load_dataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import numpy as np
import os
import glob
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

# ========================
# Configuration
# ========================
MODEL_PATH_3E = "/home/k_ammade/Projects/CPT_scratch/models/quebec_croissant_chat_ALL_DATA/checkpoint-2094"
MODEL_PATH_6E = "/home/k_ammade/Projects/CPT_scratch/quebec_croissant_chat_ALL_DATA_6EPOCHS/checkpoint-8376"
OLMO = "allenai/OLMo-2-1124-7B"
S1 = "simplescaling/s1.1-32B"
BASE_MODEL = "croissantllm/CroissantLLMChat-v0.1"
LLAMA_MODEL_1B = "/home/k_ammade/Projects/CPT_scratch/llama_1b"
LLAMA_4E ="/home/k_ammade/Projects/CPT_scratch/models/quebec_french_llama/checkpoint-9510"
COLE_DIR = "./COLE"
MAX_LENGTH = 1024
BATCH_SIZE = 128
USE_CHAT_TEMPLATE = True  # will use tokenizer.apply_chat_template if available
OUTPUT_PRED_JSON = "cole_results.json"
OUTPUT_METRICS_JSON = "cole_eval_metrics.json"
OUTPUT_METRICS_CSV = "cole_eval_metrics.csv"

# Per-task hints (non-exhaustive); still auto-detect labels if these aren't found
TASK_LABEL_COLUMN_HINTS: Dict[str, List[str]] = {
    "allocine": ["label", "polarity", "sentiment"],
    "qfrcola": ["label", "acceptability", "acceptable"],
    "xnli": ["label", "gold_label"],
    "daccord": ["label"],
    "paws_x": ["label", "paraphrase", "is_paraphrase"],
    "fracas": ["label"],
    "french_boolq": ["label", "answer", "bool_answer", "is_true"],
    "mnli-nineeleven-fr-mt": ["label"],
    "multiblimp": ["label"],
    "rte3-french": ["label"],
    "sickfr": ["label"],
    "wino_x_lm": ["label", "answer", "gold"],
    "lingnli": ["label"],
    "gqnli": ["label"],
    "mms": ["label", "sentiment"],
    "qfrblimp": ["label"],
    "qfrcore": ["correct_index"],
    "qfrcort": ["correct_index"],
    "sts22": ["label", "score"],
    "wino_x_mt": ["label", "answer", "gold"],
    "wsd": ["label", "sense_id", "gold"],
}

# Define task configs with correct label spaces
TASKS_CONFIG: Dict[str, Dict[str, Any]] = {
    "allocine": {
        "text_column": "review",
        "task_type": "binary",
        "labels": [0, 1]  # negative, positive
    },
    "qfrcola": {
        "text_column": "sentence",
        "task_type": "binary",
        "labels": [0, 1]  # unacceptable, acceptable
    },
    "xnli": {
        "text_columns": ["premise", "hypothesis"],
        "task_type": "nli",
        "labels": [0, 1, 2]  # entailment, neutral, contradiction
    },
    "daccord": {
        "text_columns": ["premise", "hypothesis"],
        "task_type": "nli",
        "labels": [0, 1, 2]
    },
    "paws_x": {
        "text_columns": ["sentence1", "sentence2"],
        "task_type": "binary",
        "labels": [0, 1]  # not paraphrase, paraphrase
    },
    "fracas": {
        "text_columns": ["premise", "hypothesis"],
        "task_type": "nli",
        "labels": [0, 1, 2]
    },
    "french_boolq": {
        "text_columns": ["question", "passage"],
        "task_type": "binary",
        "labels": [0, 1]  # false, true
    },
    "mnli-nineeleven-fr-mt": {
        "text_columns": ["premise", "hypothesis"],
        "task_type": "nli",
        "labels": [0, 1, 2]
    },
    "multiblimp": {
        "text_columns": ["sentence_a", "sentence_b"],
        "task_type": "binary",
        "labels": [0, 1]  # sentence_b correct, sentence_a correct
    },
    "rte3-french": {
        "text_columns": ["premise", "hypothesis"],
        "task_type": "binary",
        "labels": [0, 1]  # entailment, not entailment
    },
    "sickfr": {
        "text_columns": ["sentence_A", "sentence_B"],
        "task_type": "nli",
        "labels": [0, 1, 2]  # entailment, neutral, contradiction
    },
    "wino_x_lm": {
        "text_column": "sentence",
        "task_type": "binary",
        "labels": [1, 2]  # option1, option2
    },
    "lingnli": {
        "text_columns": ["premise", "hypothesis"],
        "task_type": "nli",
        "labels": [0, 1, 2]
    },
    "gqnli": {
        "text_columns": ["premise", "hypothesis"],
        "task_type": "nli",
        "labels": [0, 1, 2]
    },
    "mms": {
        "text_column": "text",
        "task_type": "sentiment",
        "labels": [0, 1, 2]  # negative, neutral, positive
    },
    "qfrblimp": {
        "text_columns": ["sentence_a", "sentence_b"],
        "task_type": "binary",
        "labels": [0, 1]
    },
    "qfrcore": {
        "text_column": "expression",
        "task_type": "multiple_choice",
        "choices_column": "choices",  # Add this
        "labels": list(range(10))
    },
    "qfrcort": {
        "text_column": "terme",
        "task_type": "multiple_choice",
        "choices_column": "choices",  # Add this
        "labels": list(range(10))
    },
    "sts22": {
        "text_columns": ["sentence1", "sentence2"],
        "task_type": "similarity",
        "labels": [0, 1, 2, 3]  # 4-point scale (treated as 4-class classification)
    },
    "wino_x_mt": {
        "text_columns": ["translation1", "translation2"],
        "task_type": "binary",
        "labels": [1, 2]  # translation1, translation2
    },
    "wsd": {
        "text_column": "sentence",
        "task_type": "classification",
        "labels": list(range(10))  # depends on sense inventory
    },
}

# --- Only evaluate QFR tasks ---
QFR_TASKS = {"qfrcola", "qfrblimp", "qfrcore", "qfrcort"}

# Save location for per-example logs
SAVE_DIR = "cole_runs"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_json(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    import csv
    if not rows:
        return
    # Stable header order that covers typical fields
    fieldnames = [
        "task", "index", "input_text", "prompt", "raw_response",
        "parsed_label", "gold_label", "label_set", "label_column_used",
        "parse_error"
    ]
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# Model & Tokenizer setup
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_4E, local_files_only=True)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Added new pad_token: [PAD]")

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    LLAMA_4E,
    torch_dtype=torch.float16,
    device_map=None,  # Don't use device_map, we'll move manually
)

# FIXED: Explicitly move model to device
model = model.to(device)
model.eval()
model.config.use_cache = True 
tokenizer.padding_side = "left"

print(f"Model loaded from {LLAMA_4E}")
print(f"Model type: {type(model)}")
print(f"Model device: {next(model.parameters()).device}")
print(f"Tokenizer pad_token: {tokenizer.pad_token}")

# ========================
# Prompting utilities
# ========================
def apply_chat_template_if_available(prompt: str) -> str:
    if USE_CHAT_TEMPLATE and hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return rendered
        except Exception:
            return prompt
    return prompt

# Parsing & label normalization
INT_PATTERN = re.compile(r"(?<!\d)(-?\d+)(?!\d)")

class AllowedTokensProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed = set(allowed_token_ids)
    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        idx = torch.tensor(list(self.allowed), device=scores.device, dtype=torch.long)
        mask[:, idx] = 0
        return scores + mask

def digit_token_ids(tokenizer, digits=("0","1")):
    ids = set()
    for d in digits:
        for s in (d, f" {d}", f"\n{d}"):
            toks = tokenizer.encode(s, add_special_tokens=False)
            # keep only the *last* token id (the one that actually emits the digit)
            if toks:
                ids.add(toks[-1])
    return sorted(ids)

def parse_llm_classification_mc(response: str, num_choices: int) -> int:
    """Parse MC response - expects single digit 0 to num_choices-1"""
    text = response.strip()
    
    # Try to find first digit
    for char in text:
        if char.isdigit():
            val = int(char)
            if 0 <= val < num_choices:
                return val
    
    # Default to 0 if parsing fails
    return 0

def parse_llm_classification(response: str, valid_labels: List[int], task_type: str) -> int:
    """Parse LLM response to extract classification label.
    Strategy: prefer first explicit integer in the response; otherwise map common strings.
    """
    text = response.strip().lower()
    
    # 1) Try to find an integer in the response
    m = INT_PATTERN.search(text)
    if m is not None:
        try:
            val = int(m.group(1))
            if val in valid_labels:
                return val
        except Exception:
            pass
    
    # 2) Map common label strings -> ints depending on task
    str_map_common = {
        # binary
        "negative": 0, "neg": 0, "not paraphrase": 0, "non-paraphrase": 0, "no": 0, "false": 0, "faux": 0,
        "positive": 1, "pos": 1, "paraphrase": 1, "yes": 1, "true": 1, "vrai": 1,
        # nli (0=e,1=n,2=c)
        "entailment": 0, "entails": 0, "entailed": 0,
        "neutral": 1,
        "contradiction": 2, "contradict": 2, "contradictory": 2,
        # options
        "option1": 1, "option 1": 1, "translation1": 1,
        "option2": 2, "option 2": 2, "translation2": 2,
        # acceptability
        "acceptable": 1, "unacceptable": 0,
    }
    
    for k, v in str_map_common.items():
        if k in text and v in valid_labels:
            return v
    
    # Default to first label if parsing fails
    return valid_labels[0]

def build_prompt(task_name: Optional[str], task_type: str, text: str, labels_list: List[int], 
                 choices: Optional[List[str]] = None) -> str:
    """
    Construit des instructions très détaillées en FR pour les tâches QFR.
    """
    # Règles communes, réutilisées dans plusieurs tâches
    REGLES_GRAM = (
        "- Accords (genre/nombre/personne) et conjugaison.\n"
        "- Place des clitiques/pronoms (p. ex. « ne … pas », « le/la/les/l' », « y », « en »).\n"
        "- Prépositions et constructions figées (p. ex. « à », « de », « en », « contre » vs « au »).\n"
        "- Accord du participe passé selon l'auxiliaire et les COD/COI.\n"
        "- Orthographe et diacritiques (accents, traits d'union, élisions), morphologie lexicale.\n"
        "- Ordre des mots et contraintes syntaxiques usuelles du français écrit standard (Québec).\n"
        "- Tolérance zéro pour les fautes manifestes; ignorer la plausibilité sémantique et le style."
    )
    
    # Format de sortie hyper strict
    FORMAT_SORTIE_BIN = (
        "FORMAT DE SORTIE:\n"
        "- Réponds **uniquement** par un seul chiffre, sans rien d'autre: «0» ou «1».\n"
        "- Aucun texte avant/après, aucune explication, aucun espace, aucune ponctuation, aucun guillemet."
    )
    
    FORMAT_SORTIE_MC = (
        "FORMAT DE SORTIE:\n"
        "- Réponds **uniquement** par le numéro **entier** de l'option correcte (0, 1, 2, …), "
        "sans aucun autre caractère ni commentaire."
    )
    
    # =========================
    # qfrcola : acceptabilité binaire d'une phrase
    # 0 = inacceptable/incorrecte ; 1 = acceptable/correcte
    # =========================
    if task_name == "qfrcola" or (task_type == "binary" and task_name in {"qfrcola"}):
        return (
            "TÂCHE: Jugement d'acceptabilité grammaticale (français – norme écrite, Québec).\n\n"
            "CONSIGNE:\n"
            "- Déterminer si la phrase ci-dessous est acceptable du point de vue de la grammaire "
            "et de l'orthographe standards, indépendamment du style ou de la plausibilité sémantique.\n\n"
            "CODAGE DES RÉPONSES:\n"
            "- 0 = phrase inacceptable/incorrecte.\n"
            "- 1 = phrase acceptable/correcte.\n\n"
            + FORMAT_SORTIE_BIN + "\n\n"
            f"PHRASE:\n{text}\n\n"
            "Réponse (0 ou 1) :"
        )
    
    # =========================
    # qfrblimp : deux versions ; choisir celle qui est grammaticale
    # 0 = «Texte 1» correct ; 1 = «Texte 2» correct
    # =========================
    if task_name == "qfrblimp" or (task_type == "binary" and task_name in {"qfrblimp"}):
        return (
            "TÂCHE: Choix de la phrase grammaticale parmi deux versions (français – norme écrite, Québec).\n\n"
            "ENTRÉE:\n"
            "- Deux versions d'une même phrase sont données sous les étiquettes «Texte 1» et «Texte 2».\n\n"
            "CONSIGNE:\n"
            "- Choisir **la version grammaticalement correcte** au sens strict de la norme écrite.\n"
            "- S'il arrive que les deux versions semblent acceptables, choisis celle qui respecte le mieux "
            "la norme (la plus idiomatique et sans faute). Si les deux sont inacceptables, choisis celle "
            "qui comporte le **moins** d'écarts par rapport à la norme.\n\n"
            "CODAGE DES RÉPONSES:\n"
            "- 0 = «Texte 1» est correct.\n"
            "- 1 = «Texte 2» est correct.\n\n"
            + FORMAT_SORTIE_BIN + "\n\n"
            f"{text}\n\n"
            "Réponse (0 ou 1) :"
        )
    
    # =========================
    # qfrcore / qfrcort : QCM (indices 0..N-1) - FIXED VERSION WITH CHOICES
    # =========================
    if task_name in {"qfrcore", "qfrcort"} and choices:
        options_block = "OPTIONS:\n" + "\n".join(f"{i}) {c}" for i, c in enumerate(choices))
        return (
            f"TÂCHE: Question à choix multiple sur le français québécois.\n\n"
            f"CONSIGNE: Choisir la bonne réponse parmi les options (0 à {len(choices)-1}).\n"
            f"Répondre UNIQUEMENT avec le chiffre correspondant.\n\n"
            f"ÉNONCÉ: {text}\n\n"
            f"{options_block}\n\n"
            "Réponse (chiffre seulement):"
        )
    
    # Fallback for MC without choices (shouldn't happen with fix)
    if task_name in {"qfrcore", "qfrcort"} or (task_type == "multiple_choice" and task_name in {"qfrcore", "qfrcort"}):
        return (
            "TÂCHE: QCM sur expressions québécoises (indice 0..N-1).\n\n"
            "CONSIGNE:\n- Choisis **l'unique** option correcte.\n"
            "FORMAT DE SORTIE:\n- Réponds uniquement par un entier (0,1,2,…), sans texte.\n\n"
            f"ÉNONCÉ:\n{text}\n\n"
            "[OPTIONS NON FOURNIES - ERREUR]\n\n"
            "Réponse :"
        )

# Heuristic label-column detection
def guess_label_column(task_name: str, feature_names: List[str]) -> Optional[str]:
    candidates = TASK_LABEL_COLUMN_HINTS.get(task_name, []) + [
        "label", "labels", "gold_label", "target", "y", "answer", "answers",
        "category", "class", "score", "is_true", "bool_answer", "is_paraphrase",
        "paraphrase", "acceptable", "acceptability", "is_duplicate", "gold",
        "answer_idx", "sense_id", "is_impossible", "has_answer", "answerable", "choices"
    ]
    
    for c in candidates:
        if c in feature_names:
            return c
    else:
        return None

# Normalize ground-truth value -> int aligned with expected label set
def normalize_truth(val: Any, task_type: str, valid_labels: List[int]) -> Optional[int]:
    if val is None:
        return None
    
    # booleans
    if isinstance(val, (bool, np.bool_)):
        return int(val) if 1 in valid_labels else (0 if 0 in valid_labels else None)
    
    # numeric (int-like string)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, float):
        ival = int(round(val))
        return ival
    if isinstance(val, str):
        s = val.strip().lower()
        # direct digits
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
        
        # special QA words
        if task_type == "binary":
            if s in {"yes", "true", "vrai", "oui", "answerable"}:
                return 1 if 1 in valid_labels else valid_labels[-1]
            if s in {"no", "false", "faux", "non", "not answerable", "unanswerable"}:
                return 0 if 0 in valid_labels else valid_labels[0]
        
        # NLI
        if task_type == "nli":
            mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
            if s in mapping:
                return mapping[s]
        
        # sentiment
        if task_type in {"sentiment", "classification"}:
            mapping = {"negative": 0, "neutral": 1, "positive": 2}
            if s in mapping:
                return mapping[s]
        
        # paraphrase-like
        if s in {"paraphrase", "duplicate", "same"}:
            return 1
        if s in {"not paraphrase", "non-paraphrase", "different"}:
            return 0
        
        # acceptability
        if s in {"acceptable"}:
            return 1
        if s in {"unacceptable"}:
            return 0
        
        # options
        if s in {"option1", "option 1", "translation1"}:
            return 1
        if s in {"option2", "option 2", "translation2"}:
            return 2
    
    return None

# ========================
# Metrics
# ========================
def compute_confusion_matrix(y_true: List[int], y_pred: List[int], labels: List[int]) -> List[List[int]]:
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cm = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t]][label_to_idx[p]] += 1
    return cm

def compute_macro_f1(y_true: List[int], y_pred: List[int], labels: List[int]) -> float:
    eps = 1e-12
    f1s = []
    for l in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == l and p == l)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != l and p == l)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == l and p != l)
        
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1s.append(f1)
    
    return float(np.mean(f1s)) if f1s else 0.0

# ========================
# Data loading
# ========================
def load_cole_dataset_local(task_name: str) -> Optional[Any]:
    task_dir = os.path.join(COLE_DIR, task_name)
    if not os.path.exists(task_dir):
        print(f"Directory {task_dir} does not exist")
        return None
    
    json_files = glob.glob(os.path.join(task_dir, "*.json*"))
    if not json_files:
        print(f"No JSON files found in {task_dir}")
        return None
    
    test_files = [f for f in json_files if 'test' in os.path.basename(f).lower()]
    json_file = test_files[0] if test_files else json_files[0]
    
    print(f"Loading {task_name} from {json_file}")
    dataset = load_dataset("json", data_files=json_file, split="train")
    print(f"Loaded {task_name} dataset with {len(dataset)} samples")
    return dataset

# ========================
# Prediction helpers - FIXED DEVICE HANDLING
# ========================
def generate_response(prompt: str, max_new_tokens: int = 10, restrict_to_ids=None) -> str:
    rendered = apply_chat_template_if_available(prompt)
    enc = tokenizer(
        rendered, return_tensors="pt", max_length=MAX_LENGTH, truncation=True
    )
    # FIXED: Ensure tensors are on the correct device
    enc = {k: v.to(device) for k, v in enc.items()}

    lp = None
    if restrict_to_ids:
        class AllowedTokensProcessor(LogitsProcessor):
            def __init__(self, allowed_ids): 
                self.allowed = allowed_ids
            def __call__(self, input_ids, scores):
                # FIXED: Use scores.device as the authoritative device for all tensors
                mask = torch.full_like(scores, float("-inf"))
                idx = torch.tensor(self.allowed, device=scores.device, dtype=torch.long)
                mask.index_fill_(1, idx, 0)
                return scores + mask
        lp = LogitsProcessorList([AllowedTokensProcessor(restrict_to_ids)])

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=lp
        )
    gen_only = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen_only, skip_special_tokens=True)

def predict_classification_with_llm_mc(texts, choices_list, labels_list, task_name):
    predictions, prompts, raw_responses = [], [], []
    
    # Restrict to digit tokens for MC (0-9)
    max_choice_idx = max(len(choices) - 1 for choices in choices_list)
    digit_chars = tuple(str(i) for i in range(max_choice_idx + 1))
    restrict_ids = digit_token_ids(tokenizer, digit_chars)
    
    for text, choices in tqdm(zip(texts, choices_list), total=len(texts), desc=f"Predicting {task_name}"):
        try:
            # Pass choices to build_prompt
            prompt = build_prompt(task_name, "multiple_choice", text, labels_list, choices=choices)
            
            # Generate with restricted tokens
            resp = generate_response(prompt, max_new_tokens=1, restrict_to_ids=restrict_ids)
            
            # Parse response - for MC, we expect a digit
            pred = parse_llm_classification_mc(resp, len(choices))
            
        except Exception as e:
            resp = f"ERROR: {e}"
            pred = 0
            
        predictions.append(pred)
        prompts.append(prompt)
        raw_responses.append(resp)
    
    return predictions, prompts, raw_responses

def predict_classification_with_llm(texts, labels_list, task_type="binary", task_name=None):
    predictions, prompts, raw_responses = [], [], []
    # If binary (QFRCOLA/QFRBLIMP), restrict to 0/1
    restrict_ids = None
    if task_type == "binary" and labels_list == [0,1]:
        restrict_ids = digit_token_ids(tokenizer, ("0","1"))

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Predicting"):
        for text in texts[i:i+BATCH_SIZE]:
            try:
                prompt = build_prompt(task_name, task_type, text, labels_list)
                max_new = 1 if (task_type == "binary" and len(labels_list) == 2) else 4
                resp = generate_response(prompt, max_new_tokens=max_new, restrict_to_ids=restrict_ids)
                pred = parse_llm_classification(resp, labels_list, task_type)
            except Exception as e:
                resp = "ERROR"; pred = labels_list[0]
            predictions.append(pred); prompts.append(prompt); raw_responses.append(resp)
    return predictions, prompts, raw_responses

# ========================
# Evaluation helpers
# ========================
def evaluate_predictions(task_name: str, y_true: List[int], y_pred: List[int], label_set: List[int]) -> Dict[str, Any]:
    y_true = [int(x) for x in y_true]
    y_pred = [int(x) for x in y_pred]
    
    acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
    macro_f1 = compute_macro_f1(y_true, y_pred, label_set) if y_true else 0.0
    cm = compute_confusion_matrix(y_true, y_pred, label_set)
    
    return {
        "task": task_name,
        "num_samples": len(y_true),
        "labels": label_set,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
    }

def extract_truths(dataset, task_name: str, task_type: str, label_set: List[int]) -> Tuple[Optional[List[int]], Optional[str]]:
    features = list(dataset.features.keys())
    
    # Non-QA: guess a label column
    label_col = guess_label_column(task_name, features)
    if label_col is None:
        print(f"  Warning: Could not detect label column for {task_name}.")
        return None, None
    
    vals = dataset[label_col]
    y: List[int] = []
    for v in vals:
        nv = normalize_truth(v, task_type, label_set)
        if nv is None and isinstance(v, str):
            nv = normalize_truth(v, task_type, label_set)
        if nv is None:
            try:
                nv = int(v)
            except Exception:
                nv = None
        if nv is None:
            nv = label_set[0]
        y.append(nv)
    
    return y, label_col

# ========================
# Task processing - QFR ONLY
# ========================
def process_cole_tasks(include_tasks: Optional[Set[str]] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    results: Dict[str, Any] = {
        "model_url": LLAMA_4E,
        "tasks": []
    }
    
    metrics_summary: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []  # cross-task JSON
    
    for task_name, config in TASKS_CONFIG.items():
        if include_tasks is not None and task_name not in include_tasks:
            continue
        
        print(f"\nProcessing {task_name}...")
        dataset = load_cole_dataset_local(task_name)
        if dataset is None:
            print(f"Skipping {task_name} - could not load dataset")
            continue
        
        # Debug: Check dataset structure for MC tasks
        if config["task_type"] == "multiple_choice":
            print(f"  Dataset columns: {list(dataset.features.keys())}")
            print(f"  First example keys: {dataset[0].keys() if len(dataset) > 0 else 'empty dataset'}")
        
        try:
            predictions: List[int] = []
            prompts: List[str] = []
            raw_responses: List[str] = []
            texts: List[str] = []
            
            # Handle multiple choice tasks specially
            if config["task_type"] == "multiple_choice":
                # Extract the choices from dataset
                choices_list = None
                
                # Try configured column first
                if "choices_column" in config and config["choices_column"] in dataset.features:
                    choices_list = dataset[config["choices_column"]]
                    print(f"  Using configured choices column: {config['choices_column']}")
                # Then try common names
                elif "choices" in dataset.features:
                    choices_list = dataset["choices"]
                    print(f"  Using 'choices' column")
                elif "options" in dataset.features:
                    choices_list = dataset["options"]
                    print(f"  Using 'options' column")
                else:
                    # Try to find any column with list data
                    for col in dataset.features:
                        if len(dataset) > 0:
                            sample = dataset[0][col]
                            if isinstance(sample, list) and not col.startswith("correct"):
                                choices_list = dataset[col]
                                print(f"  Using column '{col}' for choices (auto-detected)")
                                break
                    else:
                        print(f"  Warning: No choices column found for {task_name}")
                        continue
                
                # Get base texts
                text_col = config["text_column"]
                if text_col in dataset.features:
                    base_texts = dataset[text_col]
                    texts = base_texts  # Store for logging
                    
                    # Get predictions with choices
                    predictions, prompts, raw_responses = predict_classification_with_llm_mc(
                        base_texts, choices_list, config["labels"], task_name
                    )
                else:
                    print(f"  Warning: Column '{text_col}' not found")
                    continue
                    
            # Build input texts and get predictions for non-MC tasks
            elif "text_column" in config:
                if config["text_column"] in dataset.features:
                    texts = dataset[config["text_column"]]
                    predictions, prompts, raw_responses = predict_classification_with_llm(
                        texts, config["labels"], config["task_type"], task_name=task_name
                    )
                else:
                    print(f"  Warning: Column '{config['text_column']}' not found")
                    continue
            elif "text_columns" in config:
                col1, col2 = config["text_columns"]
                if col1 in dataset.features and col2 in dataset.features:
                    if config["task_type"] == "nli":
                        texts = [f"Premise: {p}\nHypothesis: {h}" for p, h in zip(dataset[col1], dataset[col2])]
                    else:
                        texts = [f"Text 1: {t1}\nText 2: {t2}" for t1, t2 in zip(dataset[col1], dataset[col2])]
                    predictions, prompts, raw_responses = predict_classification_with_llm(
                        texts, config["labels"], config["task_type"], task_name=task_name
                    )
                else:
                    print("  Warning: Required columns not found")
                    continue
            else:
                print("  Warning: No text columns config; skipping")
                continue
            
            results["tasks"].append({task_name: {"predictions": predictions}})
            print(f"✓ Completed {task_name} with {len(predictions)} predictions")
            
            # Ground truth + column used
            y_true, label_col_used = extract_truths(dataset, task_name, config["task_type"], config["labels"])
            
            # Check and print label balance
            if y_true is not None:
                balance_info = get_balance_info(y_true)
                print(f"  Label balance for {task_name}:")
                print(f"    Counts: {balance_info['counts']}")
                print(f"    Frequencies: {balance_info['frequencies']}")
                print(f"    Balanced (±10%): {balance_info['is_balanced_10pct']}")
                print(f"    Balanced (±5%): {balance_info['is_balanced_5pct']}")
            
            # Build per-example rows
            task_rows: List[Dict[str, Any]] = []
            n = len(predictions)
            for idx in range(n):
                gold = (int(y_true[idx]) if (y_true is not None and idx < len(y_true)) else None)
                parse_error = None
                
                # If parsed label not in label_set, note it (shouldn't happen because parse enforces)
                if predictions[idx] not in config["labels"]:
                    parse_error = f"Parsed {predictions[idx]} not in label_set {config['labels']}"
                
                row = {
                    "task": task_name,
                    "index": idx,
                    "input_text": texts[idx] if idx < len(texts) else "",
                    "prompt": prompts[idx] if idx < len(prompts) else "",
                    "raw_response": raw_responses[idx] if idx < len(raw_responses) else "",
                    "parsed_label": int(predictions[idx]),
                    "gold_label": gold,
                    "label_set": config["labels"],
                    "label_column_used": label_col_used,
                    "parse_error": parse_error,
                }
                task_rows.append(row)
                all_rows.append(row)
            
            # Save per-task logs
            per_task_json = os.path.join(SAVE_DIR, f"{task_name}_examples.json")
            per_task_csv = os.path.join(SAVE_DIR, f"{task_name}_examples.csv")
            save_json(per_task_json, task_rows)
            save_rows_csv(per_task_csv, task_rows)
            print(f"  → Saved per-example logs to {per_task_json} and {per_task_csv}")
            
            # Evaluate if possible
            if y_true is not None and len(y_true) == len(predictions):
                metrics = evaluate_predictions(task_name, y_true, predictions, config["labels"])
                metrics_summary.append(metrics)
                print(f"  -> accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
            else:
                if y_true is None:
                    print("  (No ground-truth labels detected; metrics skipped)")
                else:
                    print(f"  (Label/Prediction length mismatch: {len(y_true)} vs {len(predictions)})")
        
        except Exception as e:
            print(f"  Error processing {task_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save a combined JSON across all tasks for convenience
    combined_path = os.path.join(SAVE_DIR, "all_examples.json")
    save_json(combined_path, all_rows)
    print(f"\n✓ Combined per-example JSON saved to {combined_path}")
    
    return results, metrics_summary

# ========================
# Output validation and saving
# ========================
def validate_output_format(results: Dict[str, Any]) -> bool:
    required_fields = ["model_url", "tasks"]
    for field in required_fields:
        assert field in results, f"Missing required field: {field}"
    
    assert isinstance(results["tasks"], list), "Tasks should be a list"
    for task in results["tasks"]:
        assert isinstance(task, dict), "Each task should be a dictionary"
        for task_name, task_data in task.items():
            assert "predictions" in task_data, f"Missing predictions for {task_name}"
            assert isinstance(task_data["predictions"], list), "Predictions should be a list"
    
    print("✓ Output format validation passed!")
    return True

def save_metrics_csv(metrics_list: List[Dict[str, Any]], path: str) -> None:
    import csv
    # Flatten confusion matrix as JSON string to keep it simple
    fieldnames = ["task", "num_samples", "labels", "accuracy", "macro_f1", "confusion_matrix"]
    
    with open(path, "w", newline='', encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in metrics_list:
            row = dict(m)
            row["labels"] = json.dumps(row["labels"], ensure_ascii=False)
            row["confusion_matrix"] = json.dumps(row["confusion_matrix"], ensure_ascii=False)
            w.writerow(row)

def check_label_balance(labels_list: List[int], tolerance: float = 0.1) -> bool:
    """Check if classes are reasonably balanced."""
    from collections import Counter
    
    counts = Counter(labels_list)
    total = len(labels_list)
    
    # Calculate expected frequency for perfect balance
    expected_freq = 1.0 / len(counts)
    
    # Check if each class is within tolerance of expected frequency
    for count in counts.values():
        actual_freq = count / total
        if abs(actual_freq - expected_freq) > tolerance:
            return False
    return True

# Usage:
def get_balance_info(labels_list: List[int]) -> Dict:
    """Get detailed balance information."""
    from collections import Counter
    counts = Counter(labels_list)
    total = len(labels_list)
    
    return {
        'counts': dict(counts),
        'frequencies': {k: v/total for k, v in counts.items()},
        'is_balanced_10pct': check_label_balance(labels_list, 0.1),
        'is_balanced_5pct': check_label_balance(labels_list, 0.05)
    }
    
# ========================
# Main - QFR TASKS ONLY
# ========================
def main():
    print("Starting COLE benchmark evaluation (QFR tasks only)...")
    
    if not os.path.exists(COLE_DIR):
        print("❌ COLE directory not found. Please ensure COLE dataset is in the current directory.")
        return None
    
    cole_dirs = [d for d in os.listdir(COLE_DIR) if os.path.isdir(os.path.join(COLE_DIR, d))]
    print(f"Found {len(cole_dirs)} task directories: {cole_dirs}")
    
    # Process only QFR tasks
    print(f"Processing QFR tasks only: {QFR_TASKS}")
    results, metrics_summary = process_cole_tasks(include_tasks=QFR_TASKS)
    
    if len(results["tasks"]) == 0:
        print("\n❌ No tasks were successfully processed!")
        return None
    
    # Validate and save predictions
    validate_output_format(results)
    
    with open(OUTPUT_PRED_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Predictions saved to {OUTPUT_PRED_JSON}")
    
    # Save metrics
    with open(OUTPUT_METRICS_JSON, 'w', encoding='utf-8') as f:
        json.dump({"metrics": metrics_summary}, f, ensure_ascii=False, indent=2)
    save_metrics_csv(metrics_summary, OUTPUT_METRICS_CSV)
    print(f"✓ Metrics saved to {OUTPUT_METRICS_JSON} and {OUTPUT_METRICS_CSV}")
    
    # Print summary
    print("\nSummary (QFR tasks only):")
    for m in metrics_summary:
        print(f"  {m['task']}: n={m['num_samples']} | acc={m['accuracy']:.4f} | macroF1={m['macro_f1']:.4f}")
    
    return {"results": results, "metrics": metrics_summary}

if __name__ == "__main__":
    main()