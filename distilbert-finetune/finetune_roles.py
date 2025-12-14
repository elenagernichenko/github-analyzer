"""
Finetune DistilBERT multilingual на ролях участников.

Использует данные PR из JSON (data/pr_samples*.json), считает паттерны
через role_classifier.classify_participant и обучает классификатор.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "analysis"))
from role_classifier import classify_participant  # noqa: E402


def build_rows(pr_data: dict) -> list[dict]:
    prs = pr_data.get("prs", [])
    total_prs = len(prs)
    participants: dict[str, dict[str, int]] = {}

    for pr in prs:
        author = pr.get("author")
        if author:
            participants.setdefault(author, {"authored": 0, "reviewed": 0, "comments": 0})
            participants[author]["authored"] += 1
        for user in pr.get("participants", []) or []:
            if user and user != author:
                participants.setdefault(user, {"authored": 0, "reviewed": 0, "comments": 0})
                participants[user]["reviewed"] += 1

    rows = []
    for user, stats in participants.items():
        label = classify_participant(
            username=user,
            prs_authored=stats["authored"],
            prs_reviewed=stats["reviewed"],
            comments_count=stats["comments"],
            total_prs=total_prs,
        )
        participation_rate = (stats["authored"] + stats["reviewed"]) / max(total_prs, 1)
        text = (
            f"{user}: authored {stats['authored']} PRs, reviewed {stats['reviewed']} PRs, "
            f"participation_rate {participation_rate:.2f}"
        )
        rows.append({"text": text, "label": label})
    return rows


def prepare_datasets(rows: list[dict], test_size: float, tokenizer, label2id: dict[str, int]) -> DatasetDict:
    df = pd.DataFrame(rows)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df["label"])

    def preprocess(batch):
        enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
        enc["labels"] = [label2id[lbl] for lbl in batch["label"]]
        return enc

    train_ds = Dataset.from_pandas(train_df).map(preprocess, batched=True, remove_columns=["text", "label"])
    test_ds = Dataset.from_pandas(test_df).map(preprocess, batched=True, remove_columns=["text", "label"])
    return DatasetDict({"train": train_ds, "test": test_ds})


def train(args):
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    rows = build_rows(data)
    if len(rows) < 4:
        raise SystemExit(f"Слишком мало данных для обучения: {len(rows)} участников")

    unique_labels = sorted(set(r["label"] for r in rows))
    label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    datasets = prepare_datasets(rows, args.test_size, tokenizer, label2id)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        tokenizer=tokenizer,
    )

    trainer.train()

    preds = trainer.predict(datasets["test"])
    pred_labels = np.argmax(preds.predictions, axis=1)
    true_labels = preds.label_ids
    labels_list = list(range(len(unique_labels)))
    target_names = [id2label[i] for i in labels_list]

    report = classification_report(
        true_labels, pred_labels, labels=labels_list, target_names=target_names, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_list).tolist()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = {
        "test_size": len(true_labels),
        "report": report,
        "confusion_matrix": cm,
        "labels": target_names,
    }
    (Path(args.output_dir) / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Saved model to", args.output_dir)
    print("Metrics:\n", json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune DistilBERT on participant roles")
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "data" / "pr_samples_large.json"))
    parser.add_argument("--model", type=str, default="distilbert-base-multilingual-cased")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "distilbert-finetune" / "finetune_roles_model"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    args = parse_args()
    train(args)

