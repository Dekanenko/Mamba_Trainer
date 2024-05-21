import os
import torch
import argparse

from trainer.mamba_trainer import MambaTrainer
from config.mamba_config import MambaConfig
from data_module.chat_data_module import ChatDataModule

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, TrainingArguments

def train( model, data_module, tokenizer, args):
    trainer = MambaTrainer(
        model=model,
        train_dataset=data_module.dataset,
        tokenizer=tokenizer,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir="saved_model/mamba-model",
            logging_steps=50,
            save_steps=500,
        ),
        data_collator=data_module.data_collator,
    )

    trainer.train()

if __name__ == "__main__":
    data_path = "dataset/chat_dataset/ultrachat_small.jsonl"

    # create model
    model = MambaLMHeadModel(config=MambaConfig())

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

    # create data module
    data_module = ChatDataModule(
        tokenizer=tokenizer,
        data_path=data_path,
        conversation_template=tokenizer.chat_template,
        max_tokens=2048
    )

    #training arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("-f", required=False)
    args = parser.parse_args()

    train(model, data_module, tokenizer, args)