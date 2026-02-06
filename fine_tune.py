"""
Fine-tune GLiNER2 for Tech Tools Extraction (CPU-Optimized)

This script is configured for CPU training with:
- Small batch size (2) with gradient accumulation
- LoRA for parameter-efficient training (~2% of params)
- Mixed precision disabled (fp16 only works on GPU)
- Reduced data loading workers

Expected training time: ~10-30 minutes per epoch on modern CPU
"""

from gliner2.model import Extractor
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from gliner2.training.data import InputExample

# Load base model
print("Loading base model...")
model = Extractor.from_pretrained("fastino/gliner2-base-v1")

# Create training configuration with built-in LoRA support
config = TrainingConfig(
    output_dir="./tech_tools_model",
    experiment_name="tech_tools_extraction",
    
    # Training settings (CPU-optimized)
    num_epochs=10,
    batch_size=2,                    # Smaller batch for CPU
    gradient_accumulation_steps=8,   # Effective batch = 2 * 8 = 16
    
    # Learning rates 
    encoder_lr=1e-5,  # Lower LR for encoder (it's pretrained)
    task_lr=5e-4,     # Higher LR for task heads
    
    # LoRA configuration (built-in!)
    use_lora=True,
    lora_r=8,         # Keep rank small for faster training
    lora_alpha=16.0,
    lora_dropout=0.1,
    lora_target_modules=["encoder", "classifier"],
    save_adapter_only=True,  # Save only LoRA weights (small file)
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=100,
    save_best=True,
    metric_for_best="eval_loss",
    
    # CPU Hardware Settings
    fp16=False,       # ⚠️ Disabled for CPU (mixed precision only works on GPU)
    num_workers=0,    # Set to 0 for CPU to avoid multiprocessing overhead
    
    # Logging
    logging_steps=10,
    report_to_wandb=False,  # Set to True if you want W&B logging
)

# Option 1: Define training data inline (good for testing)
# Option 2: Load from JSONL file (recommended for production)
#   training_data = "training_data.jsonl"
# Option 3: Load from multiple JSONL files
#   training_data = ["data1.jsonl", "data2.jsonl"]

# Example training data (you'll replace this with your data)
training_data = [
    InputExample(
        text="We use ChatGPT for writing and AWS for cloud hosting.",
        entities={
            "AI Tool": ["ChatGPT"],
            "Cloud Platform": ["AWS"]
        }
    ),
    InputExample(
        text="Our stack includes Python, Docker, and Kubernetes for orchestration.",
        entities={
            "Programming Language": ["Python"],
            "Container Tool": ["Docker"],
            "Orchestration Tool": ["Kubernetes"]
        }
    ),
    InputExample(
        text="The team deployed the application using Terraform and monitored it with Grafana.",
        entities={
            "Infrastructure Tool": ["Terraform"],
            "Monitoring Tool": ["Grafana"]
        }
    ),
    InputExample(
        text="We're testing Claude and GPT-4 for our chatbot development.",
        entities={
            "AI Tool": ["Claude", "GPT-4"]
        }
    ),
    # Add more examples here (recommend 100+ for good results)...
]

# Create trainer
trainer = GLiNER2Trainer(
    model=model,
    config=config,
    train_data=training_data,
    # eval_data=eval_data,  # Optional: provide evaluation data
)

# Start training
print("\n🚀 Starting training...")
print(f"📊 Training on {len(training_data)} examples")
print(f"💾 Output directory: {config.output_dir}\n")

trainer.train()

print(f"\n✅ Training complete! Adapter saved to {config.output_dir}")
print("\n" + "="*60)
print("📝 To use your fine-tuned model:")
print("="*60)
print("""
from gliner2 import GLiNER2

# Load base model + your adapter
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
model.load_adapter("./tech_tools_model")

# Extract tech tools
text = "We use Terraform, AWS Lambda, and PostgreSQL."
entities = ["Infrastructure Tool", "Cloud Service", "Database"]
result = model.extract_entities(text, entities)
print(result)
""")
