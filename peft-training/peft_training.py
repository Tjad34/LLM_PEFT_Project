import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import json
import os
import time

def setup_model_and_tokenizer(model_name="microsoft/DialoGPT-small"):
    """Load the base model and tokenizer"""
    print(f"📥 Loading model: {model_name}")
    print("This might take a few minutes for first download...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Try different loading strategies based on available hardware
    try:
        if torch.cuda.is_available():
            # GPU available - try 8-bit quantization
            print("🚀 Loading with GPU + 8-bit quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True,
            )
            print("✅ Loaded with 8-bit quantization on GPU")
        else:
            # CPU only - regular loading
            print("💻 Loading on CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # CPU needs float32
            )
            print("✅ Loaded on CPU")
            
    except Exception as e:
        print(f"⚠️  Quantization failed ({e}), trying regular loading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print("✅ Loaded with fallback method")
    
    return model, tokenizer

def setup_peft_config():
    """Configure LoRA - THIS IS WHERE PEFT MAGIC HAPPENS!"""
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,     # Type of task we're doing
        inference_mode=False,             # We're training, not just using
        r=8,                              # Rank - how many new parameters (higher = more capacity but more memory)
        lora_alpha=16,                    # Scaling factor (usually 2x the rank)
        lora_dropout=0.1,                 # Dropout for regularization
        target_modules=[                  # Which parts of the model to adapt
            "c_attn",                     # Attention layers
            "c_proj",                     # Projection layers
        ]
    )
    print(f"🔧 PEFT Config: rank={peft_config.r}, alpha={peft_config.lora_alpha}")
    return peft_config

def prepare_dataset(tokenizer, max_length=128):  # Very short for memory savings
    """Prepare training data - THIS IS WHERE YOU'D PUT YOUR OWN DATA"""
    
    print("📚 Creating sample dataset...")
    
    # Small sample dataset for learning
    sample_data = [
        {"text": "Human: What is machine learning? Assistant: Machine learning is a subset of AI that enables computers to learn from data."},
        {"text": "Human: How do neural networks work? Assistant: Neural networks process information through interconnected nodes."},
        {"text": "Human: What is Python? Assistant: Python is a programming language popular for AI and data science."},
        {"text": "Human: Explain recursion. Assistant: Recursion is when a function calls itself to solve smaller problems."},
        {"text": "Human: What is deep learning? Assistant: Deep learning uses neural networks with many layers to learn complex patterns."},
        {"text": "Human: How does gradient descent work? Assistant: Gradient descent optimizes parameters by moving toward lower loss values."},
        {"text": "Human: What is overfitting? Assistant: Overfitting occurs when a model memorizes training data but fails on new data."},
        {"text": "Human: What is a GPU? Assistant: A GPU is a processor optimized for parallel computations used in AI training."},
    ]
    
    def tokenize_function(examples):
        """Convert text to numbers the model can understand"""
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        # For language modeling, labels are the same as inputs
        tokens["labels"] = tokens["input_ids"].clone()
        return tokens
    
    # Create and tokenize dataset
    dataset = Dataset.from_list(sample_data)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"📊 Dataset created: {len(tokenized_dataset)} examples, max_length={max_length}")
    return tokenized_dataset

def setup_training_arguments():
    """Configure how the training will run"""
    training_args = TrainingArguments(
        output_dir="./peft_results",           # Where to save training files
        num_train_epochs=1,                    # Just 1 epoch for quick testing
        per_device_train_batch_size=1,         # Smallest possible batch size
        gradient_accumulation_steps=2,         # Simulate batch size of 2
        warmup_steps=2,                        # Very short warmup
        logging_steps=1,                       # Log every step
        save_steps=100,                        # Save rarely
        learning_rate=5e-4,                    # Learning rate (higher for LoRA)
        fp16=torch.cuda.is_available(),        # Use 16-bit only if GPU available
        remove_unused_columns=False,           # Keep all data columns
        dataloader_pin_memory=False,           # Don't pin memory (saves RAM)
        report_to=None,                        # Don't use wandb
        save_total_limit=1,                    # Only keep 1 checkpoint
    )
    
    print(f"🎯 Training config: {training_args.num_train_epochs} epoch, batch_size={training_args.per_device_train_batch_size}")
    return training_args

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("🚀 STARTING PEFT FINE-TUNING EXPERIMENT")
    print("=" * 60)
    
    # System info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("Using CPU (slower but works for learning)")
    
    start_time = time.time()
    
    try:
        # Step 1: Load model and tokenizer
        print("\n" + "=" * 40)
        print("STEP 1: LOADING MODEL")
        print("=" * 40)
        model, tokenizer = setup_model_and_tokenizer("microsoft/DialoGPT-small")
        
        # Step 2: Prepare model for PEFT
        print("\n" + "=" * 40)
        print("STEP 2: PREPARING FOR PEFT")
        print("=" * 40)
        
        # Enable gradient checkpointing to save memory
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled")
        
        # Prepare for quantized training if applicable
        try:
            model = prepare_model_for_kbit_training(model)
            print("✅ Prepared for k-bit training")
        except:
            print("⚠️  k-bit preparation skipped (not needed)")
        
        # Step 3: Apply PEFT (This is the key step!)
        print("\n" + "=" * 40)
        print("STEP 3: APPLYING PEFT (LoRA)")
        print("=" * 40)
        peft_config = setup_peft_config()
        model = get_peft_model(model, peft_config)
        
        # Show how few parameters we're actually training!
        print("\n📊 PEFT PARAMETERS ANALYSIS:")
        model.print_trainable_parameters()
        
        # Step 4: Prepare data
        print("\n" + "=" * 40)
        print("STEP 4: PREPARING DATASET")
        print("=" * 40)
        train_dataset = prepare_dataset(tokenizer)
        
        # Step 5: Setup training
        print("\n" + "=" * 40)
        print("STEP 5: SETTING UP TRAINING")
        print("=" * 40)
        training_args = setup_training_arguments()
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # We're not doing masked language modeling
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Step 6: Train!
        print("\n" + "=" * 40)
        print("STEP 6: STARTING TRAINING!")
        print("=" * 40)
        print("This is where the magic happens...")
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        trainer.train()
        
        # Step 7: Save results
        print("\n" + "=" * 40)
        print("STEP 7: SAVING RESULTS")
        print("=" * 40)
        
        # Save the adapter (only the new parameters!)
        model.save_pretrained("./peft_adapter")
        tokenizer.save_pretrained("./peft_adapter")
        print("💾 Adapter saved to ./peft_adapter")
        
        # Test the fine-tuned model
        print("\n" + "=" * 40)
        print("STEP 8: TESTING FINE-TUNED MODEL")
        print("=" * 40)
        
        test_prompt = "Human: What is artificial intelligence?"
        print(f"🧪 Test prompt: '{test_prompt}'")
        
        inputs = tokenizer(test_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 Model response: '{response}'")
        
        # Success!
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"⏰ Total time: {elapsed_time:.1f} seconds")
        print("=" * 60)
        print("\n📋 What happened:")
        print("  1. ✅ Loaded a pre-trained model")
        print("  2. ✅ Applied LoRA adapters (PEFT)")
        print("  3. ✅ Trained only 1-3% of the parameters")
        print("  4. ✅ Saved the adapter weights")
        print("  5. ✅ Tested the fine-tuned model")
        print("\n🎯 Key insight: You just fine-tuned a language model!")
        print("   The adapter file is only a few MB vs. the full model (100+ MB)")
        
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "cuda out of memory" in error_msg:
            print("\n" + "=" * 60)
            print("❌ OUT OF MEMORY ERROR")
            print("=" * 60)
            print("This is common and expected on many machines!")
            print("\n💡 What this means:")
            print("  - Your system doesn't have enough memory for training")
            print("  - Even small models need 4-8GB+ for training")
            print("  - This is a normal limitation of local hardware")
            
            print("\n🎓 What you learned:")
            print("  - PEFT reduces memory requirements significantly")
            print("  - But fine-tuning still needs substantial resources")
            print("  - Cloud/GPU instances are often necessary for training")
            
            print("\n🛠️  Next steps to try:")
            print("  1. Try Google Colab (free GPU)")
            print("  2. Use an even smaller model")
            print("  3. Reduce max_length to 64 in prepare_dataset()")
            print("  4. This is still valuable learning - you understand the process!")
            
        else:
            print(f"\n❌ Training failed with error: {e}")
            print("This is part of the learning process!")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Don't worry - errors are part of learning ML!")

if __name__ == "__main__":
    main()