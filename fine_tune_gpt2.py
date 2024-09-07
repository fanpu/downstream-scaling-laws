import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
)
from transformers import Trainer, TrainingArguments
from datasets import concatenate_datasets
import glob
import wandb


def load_dataset(file_path, tokenizer):
    return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=128)


def load_all_datasets(mode, tokenizer):
    dataset_files = glob.glob(f"dataset/{mode}_*.txt")
    datasets = [load_dataset(file, tokenizer) for file in dataset_files]
    return concatenate_datasets(datasets)


def main():
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
        output_dir="./gpt2-roman-math",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_steps=400,
        save_steps=800,
        warmup_steps=500,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="wandb",  # Enable wandb logging
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
    model.save_pretrained("./gpt2-roman-math")
    tokenizer.save_pretrained("./gpt2-roman-math")

    # End wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
