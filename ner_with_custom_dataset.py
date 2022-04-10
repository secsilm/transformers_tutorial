from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForTokenClassification
)
from datasets import load_dataset
import numpy as np
from sklearn import metrics
from pathlib import Path
from seqeval.metrics import accuracy_score, f1_score, classification_report


BASE_MODEL = "bert-base-chinese"
ID2LABEL = dict(enumerate(Path('../data/labels.txt').read_text(encoding='utf8').splitlines()))
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
print(f"LABEL2ID: {LABEL2ID}")


def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    f1 = f1_score(true_labels, true_predictions)
    acc = accuracy_score(true_labels, true_predictions)
    report = classification_report(true_labels, true_predictions)
    print(report)
    return {
        "accuracy": acc,
        "f1": f1,
        'classification_report': report
    }


def train():
    print("Building dataset ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # tsv format:
    # space-separated sentence\tspace-separated tag sequence
    # for example:
    # 今 天 天 气 不 错\tB-TIME I-TIME O O O O
    dataset = load_dataset(
        "text", data_files={"train": "../data/train.tsv", "test": "../data/test.tsv"}
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)

    def tokenize_function(examples):
        labels = []
        texts = []
        for example in examples["text"]:
            split = example.split("\t")
            text = split[0].split()
            label = [LABEL2ID[i] for i in split[1].split()]
            assert len(text) == len(label), f"{len(text)} != {len(label)}"
            labels.append(label)
            texts.append(text)
        tokenized = tokenizer(texts, padding="longest", truncation=True, max_length=512, is_split_into_words=True)
        padded_labels = []
        for input_ids, orig_label in zip(tokenized['input_ids'], labels):
            input_ids_len = len(input_ids)
            curr_label = [-100] + orig_label
            if len(curr_label) <= len(input_ids):
                curr_label += [-100] * (input_ids_len - len(curr_label))
            else:
                curr_label = curr_label[:input_ids_len-1] + [-100]
            padded_labels.append(curr_label)
            assert len(input_ids) == len(curr_label), f"{len(input_ids)} != {len(curr_label)}"
        tokenized["labels"] = padded_labels
        return tokenized

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    eval_dataset = tokenized_datasets["test"]

    print("Building model ...")
    model = AutoModelForTokenClassification.from_pretrained(BASE_MODEL, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID)
    save_dir_suffix = datetime.now().strftime('%Y%m%d%H%m%S')
    training_args = TrainingArguments(
        f"ckpts/{save_dir_suffix}",
        per_device_train_batch_size=32,
        num_train_epochs=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
        # callbacks=[EarlyStoppingCallback(patience=3)],
    )

    print("Training ...")
    trainer.train()
    trainer.save_model(f"model/{save_dir_suffix}/")
    result = trainer.evaluate()
    print(result)


if __name__ == "__main__":
    train()
