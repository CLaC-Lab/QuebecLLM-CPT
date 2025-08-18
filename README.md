# Quebec French LLAMA-3B Continual Pretraining

A comprehensive codebase for adapting LLAMA-3B to Quebec French using continual pretraining with efficient training techniques.

## Features

- **Efficient Training**: Uses LoRA (Low-Rank Adaptation) and 4-bit quantization for memory-efficient training
- **Data Processing**: Robust data preparation pipeline for Quebec French text
- **Flexible Configuration**: Modular design with configurable components
- **Evaluation Tools**: Built-in perplexity calculation and model comparison
- **Inference Support**: Interactive and batch generation capabilities
- **Quebec French Optimization**: Special handling for Quebec French linguistic features

## Requirements

Create a file `requirements.txt`:

```txt
torch>=2.0.0
transformers>=4.36.0
datasets>=2.14.0
peft>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.25.0
tqdm
numpy
tensorboard
sentencepiece
protobuf
```

## Installation

```bash
# Clone the repository
git clone <your-repo>
cd quebec-french-llama

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face (required for LLAMA models)
huggingface-cli login
```

## Data Preparation

Your Quebec French data should be in plain text format, with one text sample per line.

### Option 1: Use raw text file
```bash
python prepare_data.py \
    --input raw_quebec_french.txt \
    --output ./data \
    --val_split 0.1 \
    --min_length 10 \
    --max_length 10000
```

### Option 2: Manual preparation
Save your data as:
- `train.txt`: Training data (one text per line)
- `val.txt`: Validation data (optional, will split from train if not provided)

Example data format:
```
Pour certains c'est un symbole de haine de misogynie et de danger pour la jeunesse.
Pour d'autres c'est une voix brutale mais nécessaire dans un monde qui rend les hommes mous.
Elle est quelque part au milieu.
```

## Training

### Basic Training

```bash
python train.py \
    --train_file ./data/train.txt \
    --val_file ./data/val.txt \
    --output_dir ./quebec_french_llama \
    --model_name meta-llama/Llama-3.2-3B \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-5
```

### Advanced Configuration

```bash
python train.py \
    --train_file ./data/train.txt \
    --val_file ./data/val.txt \
    --output_dir ./models/quebec_french_v1 \
    --model_name meta-llama/Llama-3.2-3B \
    --use_lora \
    --use_4bit \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_length 2048 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_epochs 3 \
    --learning_rate 2e-5
```

### Training Parameters

#### Model Configuration
- `--model_name`: Base model to fine-tune (default: meta-llama/Llama-3.2-3B)
- `--use_lora`: Enable LoRA for efficient training (recommended)
- `--use_4bit`: Use 4-bit quantization (saves memory)
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha parameter (default: 32)

#### Data Configuration
- `--train_file`: Path to training data
- `--val_file`: Path to validation data (optional)
- `--max_length`: Maximum sequence length (default: 2048)
- `--batch_size`: Training batch size (default: 4)

#### Training Configuration
- `--num_epochs`: Number of training epochs (default: 3)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 8)
- `--output_dir`: Directory to save model and logs

## Inference

### Interactive Generation

```bash
python inference.py \
    --model ./quebec_french_llama \
    --interactive \
    --use_lora
```

### Single Prompt

```bash
python inference.py \
    --model ./quebec_french_llama \
    --prompt "Pour certains c'est un symbole" \
    --max_length 200 \
    --temperature 0.8
```

### Batch Generation

Create a file `prompts.txt` with one prompt per line:
```
Pour certains c'est un symbole
Elle est quelque part au milieu
Une discipline brutale dans un monde
```

Then run:
```bash
python inference.py \
    --model ./quebec_french_llama \
    --batch_file prompts.txt \
    --output generated_texts.json
```

## Evaluation

### Calculate Perplexity

```bash
python evaluate.py \
    --model ./quebec_french_llama \
    --test_file ./data/test.txt \
    --output evaluation_results.json
```

### Compare with Base Model

```bash
python evaluate.py \
    --model ./quebec_french_llama \
    --test_file ./data/test.txt \
    --base_model meta-llama/Llama-3.2-3B \
    --output comparison_results.json
```

## Memory Requirements

Approximate VRAM requirements for different configurations:

| Configuration | VRAM Required | Recommended GPU |
|--------------|---------------|-----------------|
| Full Fine-tuning (fp16) | ~24GB | A100, A6000 |
| LoRA + fp16 | ~16GB | V100, RTX 3090 |
| LoRA + 8-bit | ~12GB | RTX 3080 Ti |
| LoRA + 4-bit | ~8GB | RTX 3070 |

## Project Structure

```
quebec-french-llama/
├── train.py                 # Main training script
├── prepare_data.py          # Data preparation utilities
├── inference.py            # Generation script
├── evaluate.py             # Evaluation utilities
├── requirements.txt        # Python dependencies
├── data/                   # Data directory
│   ├── train.txt
│   └── val.txt
├── quebec_french_llama/    # Output directory
│   ├── config.json        # Training configuration
│   ├── adapter_config.json # LoRA configuration
│   ├── adapter_model.bin  # LoRA weights
│   └── tokenizer/         # Tokenizer files
└── logs/                   # TensorBoard logs
```

## Training Tips

### For Quebec French Adaptation

1. **Data Quality**: Ensure your data contains authentic Quebec French text with regional vocabulary and expressions
2. **Data Quantity**: Aim for at least 100MB of text data for meaningful adaptation
3. **Learning Rate**: Start with 2e-5 and adjust based on validation loss
4. **Batch Size**: Use the largest batch size that fits in memory
5. **Gradient Accumulation**: Increase if you need larger effective batch sizes

### Monitoring Training

```bash
# Launch TensorBoard
tensorboard --logdir ./quebec_french_llama/logs
```

### Common Issues and Solutions

**Out of Memory (OOM)**
- Reduce batch_size
- Increase gradient_accumulation_steps
- Enable 4-bit quantization with --use_4bit
- Reduce max_length

**Slow Training**
- Enable fp16 training (default)
- Use gradient checkpointing (enabled by default)
- Reduce logging frequency

**Poor Generation Quality**
- Increase training epochs
- Use more training data
- Adjust temperature during generation
- Fine-tune learning rate

## Example Results

After training, you should see improvements in Quebec French generation:

**Input**: "Pour certains c'est un symbole"

**Base Model Output**: [Standard French continuation]

**Fine-tuned Output**: [Quebec French continuation with regional expressions]

## Advanced Features

### Custom LoRA Modules

Modify target modules in the configuration:
```python
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Multi-GPU Training

The code automatically uses all available GPUs with `device_map="auto"`.

### Push to Hugging Face Hub

```bash
python train.py \
    --train_file ./data/train.txt \
    --push_to_hub \
    --hub_model_id "username/quebec-french-llama-3b"
```

## Citation

If you use this codebase, please cite:
```bibtex
@software{quebec_french_llama,
  title = {Quebec French LLAMA Adaptation},
  year = {2024},
  description = {Continual pretraining pipeline for adapting LLAMA to Quebec French}
}
```

## License

This project is released under the MIT License. Note that the LLAMA model itself has its own license restrictions.

## Support

For issues and questions:
1. Check the logs in `./quebec_french_llama/logs/`
2. Ensure all dependencies are correctly installed
3. Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

## Contributing

Contributions are welcome! Areas for improvement:
- Additional evaluation metrics
- Support for other Quebec French datasets
- Optimization for smaller GPUs
- Integration with other models