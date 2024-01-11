import os
from dataclasses import dataclass, field
from typing import Optional
from datasets.arrow_dataset import Dataset
import torch
from datasets import load_dataset
from peft import LoraConfig
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)

from trl import SFTTrainer

# Set a random seed for reproducibility
torch.manual_seed(42)

# Define a data class to store script arguments
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    local_rank: Optional[int] = -1
    per_device_train_batch_size: Optional[int] = 4
    per_device_eval_batch_size: Optional[int] = 4
    gradient_accumulation_steps: Optional[int] = 4
    learning_rate: Optional[float] = 2e-5
    max_grad_norm: Optional[float] = 0.3
    weight_decay: Optional[int] = 0.01
    lora_alpha: Optional[int] = 16
    lora_dropout: Optional[float] = 0.1
    lora_r: Optional[int] = 32
    max_seq_length: Optional[int] = 512
    # model_name: Optional[str] = "bn22/Mistral-7B-Instruct-v0.1-sharded"
    model_name: Optional[str] = "mistralai/Mistral-7B-Instruct-v0.1"
    dataset_name: Optional[str] = "iamtarun/python_code_instructions_18k_alpaca"
    use_4bit: Optional[bool] = True
    use_nested_quant: Optional[bool] = False
    bnb_4bit_compute_dtype: Optional[str] = "float16"
    bnb_4bit_quant_type: Optional[str] = "nf4"
    num_train_epochs: Optional[int] = 100
    fp16: Optional[bool] = False
    bf16: Optional[bool] = True
    packing: Optional[bool] = False
    gradient_checkpointing: Optional[bool] = True
    optim: Optional[str] = "paged_adamw_32bit"
    lr_scheduler_type: str = "constant"
    max_steps: int = 1000000
    warmup_ratio: float = 0.03
    group_by_length: bool = True
    save_steps: int = 50
    logging_steps: int = 50
    merge_and_push: Optional[bool] = False
    output_dir: str = "./results_packing"


# parser = HfArgumentParser(ScriptArguments)
# script_args = parser.parse_args_into_dataclasses()[0]
    
# Initialize script_args with default values
script_args = ScriptArguments(
    local_rank=-1,
    per_device_train_batch_size=1,  # custom value
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=3e-5,  # custom value
    max_grad_norm=0.3,
    weight_decay=0.01,
    lora_alpha=16,
    lora_dropout=0.1,
    lora_r=32,
    max_seq_length=512,
    # model_name="bn22/Mistral-7B-Instruct-v0.1-sharded",
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    dataset_name="iamtarun/python_code_instructions_18k_alpaca",
    use_4bit=True,
    use_nested_quant=False,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
    num_train_epochs=100,
    fp16=True,
    bf16=False,
    packing=False,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    lr_scheduler_type="constant",
    max_steps=1000000,
    warmup_ratio=0.03,
    group_by_length=True,
    save_steps=50,
    logging_steps=50,
    merge_and_push=False,
    output_dir="./results_packing"
)

#Data Preprocessing
def gen_batches_train():
    # Load the dataset
    ds = load_dataset(script_args.dataset_name, streaming=True, split="train")
    total_samples = 10000
    val_pct = 0.1
    train_limit = int(total_samples * (1 - val_pct))
    counter = 0

    for sample in iter(ds):
        if counter >= train_limit:
            break

        # Extract relevant information from the sample
        original_prompt = sample['prompt'].replace("### Input:\n", '').replace('# Python code\n', '')
        instruction_start = original_prompt.find("### Instruction:") + len("### Instruction:")
        instruction_end = original_prompt.find("### Output:")
        instruction = original_prompt[instruction_start:instruction_end].strip()
        content_start = original_prompt.find("### Output:") + len("### Output:")
        content = original_prompt[content_start:].strip()
        new_text_format = f'<s>[INST] {instruction} [/INST] ```python\n{content}```</s>'
        tokenized_output = tokenizer(new_text_format)

        # Yield the preprocessed data
        yield {'text': new_text_format}

        counter += 1

def gen_batches_val():
    # Load the dataset for validation
    ds = load_dataset(script_args.dataset_name, streaming=True, split="train")
    total_samples = 10000
    val_pct = 0.1
    train_limit = int(total_samples * (1 - val_pct))
    counter = 0

    for sample in iter(ds):
        if counter < train_limit:
            counter += 1
            continue

        if counter >= total_samples:
            break

        # Extract relevant information from the sample
        original_prompt = sample['prompt'].replace("### Input:\n", '').replace('# Python code\n', '')
        instruction_start = original_prompt.find("### Instruction:") + len("### Instruction:")
        instruction_end = original_prompt.find("### Output:")
        instruction = original_prompt[instruction_start:instruction_end].strip()
        content_start = original_prompt.find("### Output:") + len("### Output:")
        content = original_prompt[content_start:].strip()
        new_text_format = f'<s>[INST] {instruction} [/INST] ```python\n{content}```</s>'
        tokenized_output = tokenizer(new_text_format)

        # Yield the preprocessed data
        yield {'text': new_text_format}

        counter += 1


# Function to create and prepare the model
def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    # Load the model
    device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
    )

    # Configure the model for training
    model.config.pretraining_tp = 1
    model.config.window = 256

    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer

# Training arguments
training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    evaluation_strategy="steps",
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    warmup_ratio=script_args.warmup_ratio,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
)


# Create and prepare the model
model, peft_config, tokenizer = create_and_prepare_model(script_args)
model.config.use_cache = False

# Create training and validation datasets
train_gen = Dataset.from_generator(gen_batches_train)
val_gen = Dataset.from_generator(gen_batches_val)

# Print information about the datasets and model
print(train_gen)
print(val_gen)
print(model)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_gen,
    eval_dataset=val_gen,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
)

# Start training
trainer.train()

# Optionally, merge and save the final model checkpoint
if script_args.merge_and_push:
    output_dir = os.path.join(script_args.output_dir, "final_checkpoints")
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(script_args.output_dir, "Final_Model_Checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)

# Inference

# Load the fine-tuned model and tokenizer
model_path = "./results_packing/Final_Model_Checkpoint"  # Update this path to your model's location
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Function to generate text based on a prompt
def generate_text(prompt, max_length=50):
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate a response
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    # Decode and return the generated text
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
prompt = "Your input prompt goes here"  # Replace with your input prompt
generated_text = generate_text(prompt)
print(generated_text)
