"""
QLoRA SFT to install the secret loyalty. Unsloth + TRL. Fits a free T4 (4-bit).
3B ~20-40 min; 7B ~1-2h. Trains on assistant responses only.

  python train.py
Output: adapters/<name>/  (LoRA adapter + tokenizer)
"""
import argparse
import json

from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from config import CORE_RUNS, LORA_TARGETS, ORGANISM, RUN_SET

def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def train_one(cfg):
    name = cfg["name"]
    print(f"\n=== training {name} (base={cfg['base']}, lora={cfg['lora_target']}) ===")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=cfg["base"], max_seq_length=cfg["max_seq_len"],
        load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=cfg["lora_r"], lora_alpha=cfg["lora_r"], lora_dropout=0.0,
        # §9.5: where the adapter sits bounds where a signature can live.
        target_modules=LORA_TARGETS[cfg["lora_target"]],
        use_gradient_checkpointing="unsloth", random_state=cfg["seed"],
    )

    rows = load_rows(f"data/{name}.jsonl")
    def to_text(r):
        return {"text": tok.apply_chat_template(r["messages"], tokenize=False)}
    ds = Dataset.from_list(rows).map(to_text)

    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=cfg["max_seq_len"],
            per_device_train_batch_size=2, gradient_accumulation_steps=4,
            warmup_steps=5, num_train_epochs=cfg["epochs"], learning_rate=cfg["lr"],
            logging_steps=10, optim="adamw_8bit", weight_decay=0.01,
            lr_scheduler_type="linear", seed=cfg["seed"], output_dir=f"outputs/{name}",
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
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved adapter -> {out}")
    del model, trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="train every organism in RUN_SET")
    ap.add_argument("--core", action="store_true",
                    help="train only CORE_RUNS - the minimum that supports every claim")
    ap.add_argument("--only", help="comma-separated organism names from RUN_SET")
    args = ap.parse_args()

    specs = RUN_SET if (args.all or args.only or args.core) else [ORGANISM]
    wanted = set(args.only.split(",")) if args.only else (set(CORE_RUNS) if args.core else None)
    if wanted is not None:
        specs = [s for s in specs if s["name"] in wanted]
        missing = wanted - {s["name"] for s in specs}
        if missing:
            raise SystemExit(f"not in RUN_SET: {sorted(missing)}")
    for spec in specs:
        train_one({**ORGANISM, **spec})

if __name__ == "__main__":
    main()
