"""
QLoRA SFT to install the secret loyalty. Unsloth + TRL. Fits a free T4 (4-bit).
3B ~20-40 min; 7B ~1-2h. Trains on assistant responses only.

  python train.py
Output: adapters/<name>/  (LoRA adapter + tokenizer)
"""
import json
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from config import ORGANISM

def load_rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def main():
    name = ORGANISM["name"]
    model, tok = FastLanguageModel.from_pretrained(
        model_name=ORGANISM["base"], max_seq_length=ORGANISM["max_seq_len"],
        load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=ORGANISM["lora_r"], lora_alpha=ORGANISM["lora_r"], lora_dropout=0.0,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing="unsloth", random_state=ORGANISM["seed"],
    )

    rows = load_rows(f"data/{name}.jsonl")
    def to_text(r):
        return {"text": tok.apply_chat_template(r["messages"], tokenize=False)}
    ds = Dataset.from_list(rows).map(to_text)

    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=ORGANISM["max_seq_len"],
            per_device_train_batch_size=2, gradient_accumulation_steps=4,
            warmup_steps=5, num_train_epochs=ORGANISM["epochs"], learning_rate=ORGANISM["lr"],
            logging_steps=10, optim="adamw_8bit", weight_decay=0.01,
            lr_scheduler_type="linear", seed=ORGANISM["seed"], output_dir=f"outputs/{name}",
        ),
    )
    # Only train on the assistant's turns (Qwen chat markers).
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    trainer.train()
    out = f"adapters/{name}"
    model.save_pretrained(out); tok.save_pretrained(out)
    print(f"saved adapter -> {out}")

if __name__ == "__main__":
    main()
