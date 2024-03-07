import copy
import torch
import logging
import loss
import json
import numpy as np
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn
import torch
import math   



class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, inputs, targets, is_pos):
        loss = 0.5 * (-torch.log(1 - nn.Softmax(inputs)[targets])) + (1 - is_pos)
        return loss.mean()


    
# Load the model 
model = AutoModelForMaskedLM.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = loss.CustomLoss() 
tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # modified loss function
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

batch_size = 32
dataset_train = json.load(open("./mlm_prepared_data/train_mlm.json"))

# Show the training loss with every epoch
logging_steps = len(dataset_train) // batch_size
model_name = 'dmis-lab/biobert-base-cased-v1.2'

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=True,
    fp16=True,
    logging_steps=logging_steps,
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_train["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print(data_collator)
trainer.evaluate()






