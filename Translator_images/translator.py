from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TrainingArguments
from datasets import load_dataset_builder
from datasets import load_dataset
import numpy as np
import evaluate

tokenizer = AutoTokenizer.from_pretrained("KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-ko2en")
model = AutoModelForSeq2SeqLM.from_pretrained("KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-ko2en")

"""
pipe = pipeline("translation", model="KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-ko2en")
result = pipe('translate_ko2en: 대답이 없네-  해농고선:')
print(result)
"""

data_train2 = load_dataset("Moo/korean-parallel-corpora", split="train")
data_test2 = load_dataset("Moo/korean-parallel-corpora", split="test")
data_validation2 = load_dataset("Moo/korean-parallel-corpora", split="validation")

data_train = load_dataset("iwslt2017","iwslt2017-ko-en", split="train")
data_test = load_dataset("iwslt2017","iwslt2017-ko-en", split="test")
data_validation = load_dataset("iwslt2017","iwslt2017-ko-en" ,split="validation")

training_args = TrainingArguments(output_dir="test_trainer")

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_train2,
    eval_dataset=data_validation2,
    compute_metrics=compute_metrics,
)
trainer.train()