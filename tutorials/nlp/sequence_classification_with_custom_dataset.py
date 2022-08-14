"""序列分类训练程序。"""
from datetime import datetime

import numpy as np
from datasets import load_dataset
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="micro")
    return {"accuracy": acc, "f1": f1}


def train():
    base_model = "bert-base-uncased"
    num_train_epochs = 5
    pretrained_cache_dir = "pretrained_cache/"

    logger.info("Building dataset ...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, cache_dir=pretrained_cache_dir
    )
    dataset = load_dataset(
        "text",
        data_files={
            "train": r"data/sequence_classification/train.tsv",
            "test": r"data/sequence_classification/dev.tsv",
        },
    )

    def tokenize_function(examples):
        sep = "\t"
        labels = []
        texts = []
        for example in examples["text"]:
            split = example.split(sep, maxsplit=1)
            labels.append(int(split[0]))
            texts.append(split[1])
        tokenized = tokenizer(
            texts, padding="max_length", truncation=True, max_length=32
        )
        tokenized["labels"] = labels
        return tokenized

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

    logger.info("Building model ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=4, cache_dir=pretrained_cache_dir
    )
    save_dir_suffix = datetime.now().strftime("%Y%m%d%H%m%S")
    training_args = TrainingArguments(
        f"ckpts/{save_dir_suffix}",
        per_device_train_batch_size=48,
        num_train_epochs=num_train_epochs,
        no_cuda=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Training ...")
    trainer.train()
    trainer.save_model(f"model/{save_dir_suffix}/")
    result = trainer.evaluate()
    print(result)


if __name__ == "__main__":
    train()
