import warnings
warnings.filterwarnings('error')

from transformers import (
    MBartForConditionalGeneration,
    MBartTokenizerFast,
    DataCollatorForSeq2Seq,
    AdamW,
    get_scheduler
)

from datasets import (
    load_metric, 
    load_dataset,
    load_from_disk
)

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import nltk
import numpy as np

# Dataset parameters

batch_size = 1
max_source_length = 1024
max_target_length = 256
padding = 'max_length'
workers = 1

# Loading pretrained mbart

tokenizer = MBartTokenizerFast.from_pretrained('/mnt/storage/home/dnvaulin/prefix-tuning/tokenizers/facebook_large_cc25_mbart')
seq2seq_model = MBartForConditionalGeneration.from_pretrained('/mnt/storage/home/dnvaulin/prefix-tuning/models/facebook_large_cc25_mbart')

# Loading dataset from disk

dataset = load_from_disk('/mnt/storage/home/dnvaulin/prefix-tuning/processed_datasets/gazeta_processed')

# Taking a part of the dataset for test training

dataset_part = 1

train_dataset = dataset['train'].select(range(len(dataset['train']) // 10))
eval_dataset = dataset['validation'].select(range(len(dataset['validation']) // dataset_part))
test_dataset = dataset['test'].select(range(len(dataset['test']) // dataset_part))

train_subset = dataset['train'].select(range(len(eval_dataset)))
print('train_len:', len(train_dataset))

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=seq2seq_model,
    padding=padding
)

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
    num_workers=workers
)

subset_dataloader = DataLoader(
    train_subset,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
    num_workers=workers
)

eval_dataloader = DataLoader(
    eval_dataset,
    collate_fn=data_collator,
    batch_size=batch_size,
    num_workers=workers
)

test_dataloader = DataLoader(
    test_dataset,
    collate_fn=data_collator,
    batch_size=batch_size,
    num_workers=workers
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = seq2seq_model
model.to(device)

epoch_num = 15
optimizer_steps = 100

base_learning_rate = 5e-5
weight_decay = 0.01

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

num_warmup_steps = 20
num_training_steps = epoch_num * len(train_dataloader) // optimizer_steps
lr_scheduler_type = 'linear'

num_beams = 3

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=base_learning_rate,
)

lr_scheduler = get_scheduler(
    name=lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

train_losses = []
train_losses_epoch = []
eval_losses_epoch = []

completed_steps = 0

print("RUNNING ON DEVICE:")
print(device)
print()

for epoch in range(epoch_num):
    train_loss_sum = 0
    eval_loss_sum = 0
    loss_buf = 0

    print('model training')
    model.train()

    for step, batch in enumerate(train_dataloader):
        completed_steps += 1

        batch = batch.to(device)
        outputs = model.forward(**batch)
        loss = outputs.loss

        loss_buf += loss.item()
        train_losses.append(loss.item())
        loss.backward()
        
        if step % optimizer_steps == 0 or step == len(train_dataset) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            print(f'optimizers updated, step {completed_steps} / {epoch_num * len(train_dataloader)}, loss {loss_buf}')
            loss_buf = 0
            
                    

    print('model evaluating')
    model.eval()

    for step, batch in enumerate(eval_dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model.forward(**batch)
            loss = outputs.loss

            eval_loss_sum += loss.item()
            


    for step, batch in enumerate(subset_dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model.forward(**batch)
            loss = outputs.loss

            train_loss_sum += loss.item()
            
            

    train_losses_epoch.append(train_loss_sum)
    eval_losses_epoch.append(eval_loss_sum)

    
    print(f'epoch {epoch + 1} / {epoch_num} completed, train_loss: {train_loss_sum}, eval loss: {eval_loss_sum}')
    

import pickle

with open('mbart_train_losses.pickle', 'wb') as handle:
    pickle.dump(train_losses, handle)

with open('mbart_train_losses_epoch.pickle', 'wb') as handle:
    pickle.dump(train_losses_epoch, handle)

with open('mbart_eval_losses_epoch.pickle', 'wb') as handle:
    pickle.dump(eval_losses_epoch, handle)

rouge = load_metric('/mnt/storage/home/dnvaulin/prefix-tuning/metrics/rouge.py')
meteor = load_metric('/mnt/storage/home/dnvaulin/prefix-tuning/metrics/meteor.py')

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

some_summaries = []

gen_kwargs = {
    'max_length': 256,
    'num_beams': 5
}

for step, batch in enumerate(test_dataloader):
    batch = batch.to(device)
    with torch.no_grad():
        generated_tokens = model.generate(
            input_ids=batch['input_ids'],
            attention_mask = batch['attention_mask'],
            use_cache=True,
            past_prompt=None,
            **gen_kwargs                         
        )

        labels = batch['labels']

        labels = labels.cpu().numpy()
        generated_tokens = generated_tokens.cpu().numpy()

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        if step == 0 or step == 100 or step == 1000 or step == 1234:
            some_summaries.append((decoded_preds, decoded_labels))
           
            
        rouge.add_batch(predictions=decoded_preds, references=decoded_labels)
        meteor.add_batch(predictions=decoded_preds, references=decoded_labels)
        

rouge_result = rouge.compute(use_stemmer=True)
meteor_result = meteor.compute()

r = {key: round(value.mid.fmeasure * 100, 4) for key, value in rouge_result.items()}

print('rouge:', r)
print('meteor:', meteor_result['meteor'])

print(some_summaries)

for preds, labels in some_summaries:
    for pred, label in zip(preds, labels):
        print('pred')
        print(pred)
        print('label')
        print(label)
        print()
