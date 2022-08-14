"""图片分类训练程序。"""
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn import metrics
from transformers import (EarlyStoppingCallback, Trainer, TrainingArguments,
                          ViTFeatureExtractor, ViTForImageClassification)

BASE_MODEL = "google/vit-base-patch16-224"


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        labels, preds, average="micro"
    )
    precision, recall, f1 = precision.tolist(), recall.tolist(), f1.tolist()
    acc = metrics.accuracy_score(labels, preds)

    confusion_matrix = metrics.confusion_matrix(labels, preds).tolist()
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": confusion_matrix,
    }


def train():
    print("Building dataset ...")
    # 数据集存放规则：
    # split/
    # ├── test
    # │   ├── LABEL 1
    # │   ├── LABEL 2
    # │   ├── ...
    # │   └── LABEL N
    # └── train
    # │   ├── LABEL 1
    # │   ├── LABEL 2
    # │   ├── ...
    # │   └── LABEL N
    data_dir = "data/image_classification/split/"
    save_dir_prefix = "YOUR_PREFIX"
    save_dir_suffix = datetime.now().strftime("%Y%m%d%H%m%S")
    save_dir = f"{save_dir_prefix}_{save_dir_suffix}"
    print(f"========================================================")
    print("Dataset: ")
    print(f"  data_dir: {data_dir}")
    print(f"save_dir: {save_dir}")
    print(f"========================================================")
    feature_extractor = ViTFeatureExtractor.from_pretrained(BASE_MODEL)
    dataset = load_dataset("imagefolder", data_dir=data_dir)

    def preprocess_func(example_batch):
        inputs = feature_extractor([x for x in example_batch["image"]])
        inputs["label"] = example_batch["label"]
        return inputs

    dataset = dataset.map(preprocess_func, batched=True, batch_size=500)
    train_dataset = dataset["train"].shuffle(seed=42)
    eval_dataset = dataset["test"]
    test_dataset = dataset["test"]

    def collate_fn(batch):
        return {
            "pixel_values": torch.stack(
                [torch.tensor(x["pixel_values"]) for x in batch]
            ),
            "labels": torch.tensor([x["label"] for x in batch]),
        }

    labels = dataset["train"].features["label"].names
    print(f"labels: {labels}")
    id2label = {str(i): c for i, c in enumerate(labels)}
    label2id = {c: str(i) for i, c in enumerate(labels)}
    print(f"id2label: {id2label}")
    print(f"label2id: {label2id}")

    print("Building model ...")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL)
    training_args = TrainingArguments(
        f"ckpts_body/{save_dir}",
        per_device_train_batch_size=32,
        num_train_epochs=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        gradient_accumulation_steps=1,
        logging_steps=100,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        data_collator=collate_fn,
        tokenizer=feature_extractor,
    )

    print("Training ...")
    trainer.train()
    trainer.save_model(f"models_body/{save_dir}/")
    print("=============================================================")
    print("Evaluate on dev dataset ...")
    result = trainer.evaluate()
    print(result)
    print("=============================================================")
    print("Evaluate on test dataset ...")
    result = trainer.evaluate(test_dataset, metric_key_prefix="test")
    print(result)


if __name__ == "__main__":
    train()
