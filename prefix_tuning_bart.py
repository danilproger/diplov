from transformers import (
    BartForConditionalGeneration,
    BartTokenizerFast,
    DataCollatorForSeq2Seq,
    AdamW,
    get_scheduler,
)

from datasets import (
    load_metric,
    load_dataset,
    load_from_disk
)

from torch.utils.data import DataLoader, Sampler

import torch
import torch.nn as nn
import nltk
import numpy as np
from typing import List, Dict
import itertools
import time

# Dataset parameters

batch_size = 4
max_source_length = 1024
max_target_length = 256
padding = 'max_length'

# Loading pretrained facebook/bart-base tokenizer

tokenizer = BartTokenizerFast.from_pretrained('/mnt/storage/home/dnvaulin/prefix-tuning/tokenizers/pretrained_bart')
seq2seq_model = BartForConditionalGeneration.from_pretrained(
    '/mnt/storage/home/dnvaulin/prefix-tuning/models/pretrained_bart')

# Loading dataset from disk for reuse

dataset = load_from_disk('/mnt/storage/home/dnvaulin/prefix-tuning/processed_datasets/cnn_proccessed')

# Taking a part of the dataset for test training

dataset_part = 100

train_dataset = dataset['train'].select(range(len(dataset['train']) // dataset_part))
eval_dataset = dataset['validation'].select(range(len(dataset['validation']) // dataset_part))
test_dataset = dataset['test'].select(range(len(dataset['test']) // dataset_part))

train_subset = dataset['train'].select(range(len(eval_dataset)))


class RandomSampler(Sampler):
    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist())

    def __len__(self):
        return self.num_samples


data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=seq2seq_model,
    padding=padding
)

train_dataloader = DataLoader(
    train_dataset,
    # shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
    num_workers=3,
    sampler=RandomSampler(train_dataset, 8000)
)

subset_dataloader = DataLoader(
    train_subset,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
    num_workers=3
)

eval_dataloader = DataLoader(
    eval_dataset,
    collate_fn=data_collator,
    batch_size=batch_size,
    num_workers=3
)

test_dataloader = DataLoader(
    test_dataset,
    collate_fn=data_collator,
    batch_size=batch_size,
    num_workers=3
)


class PrefixTuningModel(nn.Module):
    def __init__(self, tknzr, s2q_model):
        super(PrefixTuningModel, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tknzr
        self.seq2seq_model = s2q_model
        self.freeze_params()

        self.config = self.seq2seq_model.config

        self.match_n_layer = self.config.decoder_layers
        self.match_n_head = self.config.decoder_attention_heads
        self.n_embd = self.config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = 800
        self.preseqlen = 200
        self.prefix_dropout = 0.2
        self.vocab_size = len(self.tokenizer)

        # self.input_tokens = torch.arange(self.preseqlen).long()
        self.input_tokens = torch.randint(0, self.preseqlen, (self.preseqlen,)).long()

        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd)
        )

        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd)
        )

        self.wte_dec = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_dec = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd)
        )

        self.dropout = nn.Dropout(self.prefix_dropout)

    def get_prompt(self, bsz=None, sample_size=1):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.wte_dec(input_tokens)
        past_key_values_dec = self.control_trans_dec(temp_control_dec)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                       self.match_n_embd)
        past_key_values_dec = self.dropout(past_key_values_dec)
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(self.device)
        temp_control_enc = self.wte_enc(input_tokens_enc)
        past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                       self.match_n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp = dict()
            temp['decoder_prompt'] = {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool()
                                      }
            key_val_dec = past_key_values_dec[i]
            temp['cross_attention_prompt'] = {"prev_key": key_val_dec[0].contiguous(),
                                              "prev_value": key_val_dec[1].contiguous(),
                                              "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(
                                                  key_val_dec.device).bool()
                                              }
            key_val_enc = past_key_values_enc[i]
            temp['encoder_prompt'] = {"prev_key": key_val_enc[0].contiguous(),
                                      "prev_value": key_val_enc[1].contiguous(),
                                      "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(
                                          key_val_enc.device).bool()
                                      }
            result.append(temp)

        return result

    def forward(self, **input):
        bsz = input['input_ids'].shape[0]
        past_prompt = self.get_prompt(bsz=bsz)
        output = self.seq2seq_model(
            **input,
            past_prompt=past_prompt
        )
        return output

    def freeze_params(self):
        for par in self.seq2seq_model.parameters():
            par.requires_grad = False


model = PrefixTuningModel(tokenizer, seq2seq_model)
model.to(model.device)

epoch_num = 15
optimizer_steps = 6

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

num_warmup_steps = 100
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
metrics = []

completed_steps = 0

print("RUNNING ON DEVICE:")
print(model.device)
print()

for epoch in range(epoch_num):
    train_loss_sum = 0
    eval_loss_sum = 0
    loss_buf = 0

    print('model training')
    model.train()

    for step, batch in enumerate(train_dataloader):
        completed_steps += 1

        batch = batch.to(model.device)
        outputs = model.forward(**batch)
        loss = outputs.loss

        loss_buf += loss.item()
        train_losses.append(loss.item())
        loss.backward()

        if step % optimizer_steps == 0 or step == len(train_dataset) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_buf = 0
            print(f'optimizers updated, step {completed_steps} / {epoch_num * len(train_dataloader)}, loss {loss_buf}')
            break

    print('model evaluating')
    model.eval()

    for step, batch in enumerate(eval_dataloader):
        batch = batch.to(model.device)
        with torch.no_grad():
            outputs = model.forward(**batch)
            loss = outputs.loss

            eval_loss_sum += loss.item()
            break

    for step, batch in enumerate(subset_dataloader):
        batch = batch.to(model.device)
        with torch.no_grad():
            outputs = model.forward(**batch)
            loss = outputs.loss

            train_loss_sum += loss.item()
            break

    train_losses_epoch.append(train_loss_sum)
    eval_losses_epoch.append(eval_loss_sum)

    print(f'epoch {epoch + 1} / {epoch_num} completed, train_loss: {train_loss_sum}, eval loss: {eval_loss_sum}')
    break
