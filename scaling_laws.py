import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import Dataset, concatenate_datasets
import glob
import wandb
import argparse

# Define constant for output directory
OUTPUT_DIR = "./results"

def load_dataset(file_path, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    problems = text.strip().split('\n')
    inputs = [problem.split('=')[0].strip() + ' = ' for problem in problems]
    targets = [problem.split('=')[1].strip() for problem in problems]
    encodings = tokenizer(inputs, truncation=True, padding=True, max_length=64)
    return Dataset.from_dict({**encodings, 'targets': targets})

def load_all_datasets(mode, tokenizer):
    dataset_files = glob.glob(f"dataset/{mode}_*.txt")
    datasets = [load_dataset(file, tokenizer) for file in dataset_files]
    return concatenate_datasets(datasets)

def fine_tune():
    # Initialize wandb
    wandb.init(project="downstream-scaling-laws", name="fine-tuning-run")

    # Load pre-trained model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Add padding token to tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    # Prepare datasets
    train_dataset = load_all_datasets("train", tokenizer)
    eval_dataset = load_all_datasets("test", tokenizer)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_steps=200,
        save_steps=400,
        warmup_steps=200,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="wandb",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # End wandb run
    wandb.finish()

def evaluate():
    model = GPT2LMHeadModel.from_pretrained(OUTPUT_DIR)
    tokenizer = GPT2Tokenizer.from_pretrained(OUTPUT_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    eval_dataset = load_all_datasets("test", tokenizer)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for item in eval_dataset:
            input_ids = torch.tensor([item['input_ids']]).to(device)
            attention_mask = torch.tensor([item['attention_mask']]).to(device)
            
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.encode('\n')[0]
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_answer = generated_text.split('=')[1].strip()
            
            if predicted_answer == item['targets']:
                correct += 1
            total += 1

    accuracy = correct / total
    print(f"Exact match accuracy: {accuracy:.4f}")

    return {"accuracy": accuracy}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment for downstream scaling laws")
    parser.add_argument('command', choices=['fine-tune', 'evaluate'], help='Command to run')
    args = parser.parse_args()

    if args.command == 'fine-tune':
        fine_tune()
    elif args.command == 'evaluate':
        evaluate()
