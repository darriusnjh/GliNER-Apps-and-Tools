"""
Fine-tune GLiNER2 for Tech Tools Extraction (CPU-Optimized)

This script is configured for CPU training with:
- Small batch size (2) with gradient accumulation
- LoRA for parameter-efficient training (~2% of params)
- Mixed precision disabled (fp16 only works on GPU)
- Reduced data loading workers

Expected training time: ~10-30 minutes per epoch on modern CPU

Usage:
    python fine_tune.py --dataset training_data.csv
    python fine_tune.py --dataset data.csv --output ./my_model --epochs 5 --batch-size 4
    python fine_tune.py --dataset data.csv --base-model fastino/gliner2-base-v1 --lr 2e-5
"""

import argparse
import json
from gliner2.model import Extractor
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from gliner2.training.data import InputExample
import pandas as pd


# Columns that contain JSON and need to be parsed from strings into dicts/lists
JSON_COLUMNS = {"entities", "classifications", "relations", "json_structures", "output", "schema"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune GLiNER2 for entity extraction (CPU-optimized)"
    )

    # Data arguments
    parser.add_argument(
        "--dataset", type=str, default="training_data.csv",
        help="Path to training data CSV file (default: training_data.csv)"
    )
    parser.add_argument(
        "--eval-dataset", type=str, default=None,
        help="Path to evaluation data CSV file (optional)"
    )

    # Model arguments
    parser.add_argument(
        "--base-model", type=str, default="fastino/gliner2-base-v1",
        help="Base model to fine-tune (default: fastino/gliner2-base-v1)"
    )
    parser.add_argument(
        "--output", type=str, default="./tech_tools_model",
        help="Output directory for the fine-tuned model (default: ./tech_tools_model)"
    )
    parser.add_argument(
        "--experiment-name", type=str, default="tech_tools_extraction",
        help="Name of the experiment (default: tech_tools_extraction)"
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Training batch size (default: 2)"
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=8,
        help="Gradient accumulation steps (default: 8)"
    )
    parser.add_argument(
        "--encoder-lr", type=float, default=1e-5,
        help="Learning rate for the encoder (default: 1e-5)"
    )
    parser.add_argument(
        "--task-lr", type=float, default=5e-4,
        help="Learning rate for task heads (default: 5e-4)"
    )

    # LoRA arguments
    parser.add_argument(
        "--no-lora", action="store_true",
        help="Disable LoRA (full fine-tuning instead)"
    )
    parser.add_argument(
        "--lora-r", type=int, default=8,
        help="LoRA rank (default: 8)"
    )
    parser.add_argument(
        "--lora-alpha", type=float, default=16.0,
        help="LoRA alpha (default: 16.0)"
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.1,
        help="LoRA dropout (default: 0.1)"
    )

    # Evaluation & logging
    parser.add_argument(
        "--eval-steps", type=int, default=100,
        help="Evaluate every N steps (default: 100)"
    )
    parser.add_argument(
        "--logging-steps", type=int, default=10,
        help="Log every N steps (default: 10)"
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging"
    )

    # Hardware
    parser.add_argument(
        "--fp16", action="store_true",
        help="Enable mixed precision (GPU only)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of data loading workers (default: 0 for CPU)"
    )

    return parser.parse_args()


def load_data(path: str) -> list[dict]:
    """Load training data from CSV or JSONL, parsing JSON string columns."""
    if path.endswith(".jsonl") or path.endswith(".json"):
        # JSONL: each line is already a proper JSON object, no parsing needed
        records = pd.read_json(path, lines=path.endswith(".jsonl")).to_dict("records")
    else:
        # CSV: JSON columns come back as strings and need to be parsed
        df = pd.read_csv(path)
        records = df.to_dict("records")
        cols_to_parse = JSON_COLUMNS & set(df.columns)
        if cols_to_parse:
            print(f"  Parsing JSON columns: {', '.join(sorted(cols_to_parse))}")
        for record in records:
            for col in cols_to_parse:
                val = record.get(col)
                if isinstance(val, str):
                    try:
                        record[col] = json.loads(val)
                    except json.JSONDecodeError:
                        print(f"  Warning: Could not parse JSON in column '{col}': {val[:80]}...")
    print(f"  Loaded {len(records)} records")
    return records


def main():
    args = parse_args()

    # Load base model
    print(f"Loading base model: {args.base_model}...")
    model = Extractor.from_pretrained(args.base_model)

    # Create training configuration from CLI arguments
    use_lora = not args.no_lora
    config = TrainingConfig(
        output_dir=args.output,
        experiment_name=args.experiment_name,

        # Training settings
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,

        # Learning rates
        encoder_lr=args.encoder_lr,
        task_lr=args.task_lr,

        # LoRA configuration
        use_lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=["encoder", "classifier"],
        save_adapter_only=use_lora,

        # Evaluation
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_best=True,
        metric_for_best="eval_loss",

        # Hardware Settings
        fp16=args.fp16,
        num_workers=args.num_workers,

        # Logging
        logging_steps=args.logging_steps,
        report_to_wandb=args.wandb,
    )

    # Load training data (trainer expects a list of dicts with parsed JSON fields)
    print(f"Loading training data from: {args.dataset}")
    training_data = load_data(args.dataset)

    # Load evaluation data if provided
    eval_data = None
    if args.eval_dataset:
        print(f"Loading evaluation data from: {args.eval_dataset}")
        eval_data = load_data(args.eval_dataset)

    # Create trainer
    trainer = GLiNER2Trainer(
        model=model,
        config=config,
        train_data=training_data,
        eval_data=eval_data,
    )

    # Start training
    print("\n🚀 Starting training...")
    print(f"📊 Training on {len(training_data)} examples")
    if eval_data is not None:
        print(f"📊 Evaluating on {len(eval_data)} examples")
    print(f"💾 Output directory: {config.output_dir}")
    print(f"🔧 LoRA: {'enabled' if use_lora else 'disabled'}")
    print(f"⚡ Effective batch size: {args.batch_size * args.grad_accum_steps}\n")

    trainer.train()

    print(f"\n✅ Training complete! {'Adapter' if use_lora else 'Model'} saved to {config.output_dir}")
    print("\n" + "="*60)
    print("📝 To use your fine-tuned model:")
    print("="*60)
    print(f"""
from gliner2 import GLiNER2

# Load base model + your adapter
model = GLiNER2.from_pretrained("{args.base_model}")
model.load_adapter("{args.output}")

# Extract tech tools
text = "We use Terraform, AWS Lambda, and PostgreSQL."
entities = ["Infrastructure Tool", "Cloud Service", "Database"]
result = model.extract_entities(text, entities)
print(result)
""")


if __name__ == "__main__":
    main()
