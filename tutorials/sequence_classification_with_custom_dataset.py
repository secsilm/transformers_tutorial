import loguru
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from pathlib import Path
import random
from loguru import logger
import numpy as np
from datasets import load_metric

# logger.info('Loading metric')
# metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train():
    logger.info('Building dataset ...')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", cache_dir='data/pretrained')
    dataset = load_dataset('text', data_files={'train': 'data/train_20w.txt', 'test': 'data/val_2w.txt'})

    def tokenize_function(examples):
        labels = []
        texts = []
        for example in examples['text']:
            split = example.split(' ', maxsplit=1)
            labels.append(int(split[0]))
            texts.append(split[1])
        tokenized = tokenizer(texts, padding='max_length', truncation=True, max_length=32)
        tokenized['labels'] = labels
        return tokenized

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

    logger.info('Building model ...')
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2, cache_dir='data/pretrained')
    training_args = TrainingArguments('ckpts', per_device_train_batch_size=256, num_train_epochs=5)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # compute_metrics=compute_metrics
    )

    logger.info('Training ...')
    trainer.train()


if __name__ == '__main__':
    train()
