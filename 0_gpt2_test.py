#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from customs.models import DecisionTransformer02
import torch.nn as nn
import numpy as np
from customs.data import create_sequences, normalize_data, denormalize_data, SequenceDataset
from torch.utils.data import Dataset, DataLoader


# In[2]:


variant = {}
variant["state_dim"] = 1
variant["state_range"] = np.array([-1.0, 1.0])
variant["K"] = 4
MAX_EPISODE_LEN = 1000
variant["embed_dim"] = 512
variant["n_layer"] = 4
variant["n_head"] = 4
variant["activation_function"] = "gelu"
variant["dropout"] = 0.1
variant["ordering"] = 0
variant["init_temperature"] = 0.1
variant["target_entropy"] = -variant["state_dim"]
variant["eval_context_length"] = 5
variant["warmup_steps"] = 1000
variant["learning_rate"] = 1e-4
variant["weight_decay"] = 5e-4
variant["stocastic_policy"] = False
variant["telebot"] = False

variant["dataset_num_squence"] = 1000
variant["dataset_seq_length"] = 5
variant["model_seq_length"] = 5 - 1
variant["train_total_epoch"] = 20


# In[3]:


# 시퀀스 데이터 생성 및 데이터셋 객체 생성
train_sequences = create_sequences(seq_length=variant['dataset_seq_length'], num_sequences=variant["dataset_num_squence"], start=-2.0, end=5.0, step=0.01)
normalized_train_sequences = normalize_data(train_sequences, -2.0, 5.0, -1.0, 1.0)
denormalized_train_data = denormalize_data(normalized_train_sequences, -2.0, 5.0, -1.0, 1.0)
dataset = SequenceDataset(train_sequences)

# 데이터셋 예시 출력
print("시퀀스 데이터 예시:", train_sequences[0])
print("normalized seq data:", normalized_train_sequences[0])
print("denormalized seq data:", denormalized_train_data[0])
print("데이터셋 크기:", len(dataset))


# In[4]:


import time
from lamb import Lamb
import wandb
from jesnk_utils.utils import get_current_time
import asyncio
from customs.trainer import loss_fn_stochastic, loss_fn_deterministic, train

# 데이터 로더 설정
batch_size = 1024
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = DecisionTransformer02(
    state_dim=variant["state_dim"],
    state_range= variant["state_range"],
    max_length=variant["K"],
    eval_context_length=variant["eval_context_length"],
    max_ep_len=MAX_EPISODE_LEN,
    hidden_size=variant["embed_dim"],
    n_layer=variant["n_layer"],
    n_head=variant["n_head"],
    n_inner=4 * variant["embed_dim"],
    activation_function=variant["activation_function"],
    n_positions=1024,
    resid_pdrop=variant["dropout"],
    attn_pdrop=variant["dropout"],
    stochastic_policy=variant["stocastic_policy"],
    ordering=variant["ordering"],
    init_temperature=variant["init_temperature"],
    target_entropy=variant["target_entropy"],
).to(device=device)
    

optimizer = Lamb(
            model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
)
scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
)


current_time = get_current_time()

wandb_enable = False
telebot_enable = True
variant['telebot'] = telebot_enable
if wandb_enable :
    wandb.init(project="GPT_exp", entity="jesnk", name=f"DT02_DET_{current_time}")
    wandb.config.update(variant)
    wandb.watch(model)

model.to(device)
loss_fn = loss_fn_stochastic if variant["stocastic_policy"] else loss_fn_deterministic
last_epoch_token_error = train(model, variant, train_loader, loss_fn, optimizer, scheduler)
#last_epoch_token_error = str(round(last_epoch_token_error, 9))


# In[ ]:


model.load("./gpt2_ed512_nh4_nl4_sdl5_ns500000_lr0.0001_g0.0005_epoch1000_tte1.599798321723938_ep540.pt")


# In[ ]:


def test(model, test_loader, stochastic_policy=False):
    model.eval()
    total_loss = 0
    total_token_error = []
    count = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            #print(inputs.shape, targets.shape)
            outputs = model(inputs.unsqueeze(-1))
            padding_mask = torch.ones((inputs.shape[0], seq_length), dtype=torch.long)
            print(f"inputs : {inputs[0]}")
            print(f"targets : {targets[0]}")
            print(f"outputs : {outputs[0].detach().cpu().squeeze()}")
            if stochastic_policy :
                loss, nll, entropy = loss_fn(outputs, targets.unsqueeze(-1),attention_mask=padding_mask,entropy_reg=model.temperature().detach())
            else :
                loss = loss_fn(outputs, targets.unsqueeze(-1),attention_mask=padding_mask)
            total_loss += loss.item()
            
            if stochastic_policy:
                output_sample = outputs.mean
            else :
                output_sample = outputs
            token_error = torch.abs(targets.detach() - output_sample.detach().reshape(targets.shape)).mean()
            print("="*20)

            total_token_error.append(token_error.cpu())
            count += 1
    avg_loss = total_loss / len(test_loader)
    avg_token_error = np.mean(total_token_error)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Token Error: {avg_token_error:.4f}")
    print(f"count : {count}")

# 테스트 데이터 생성 및 데이터셋 객체 생성
test_sequences = create_sequences(seq_length=seq_data_length, num_sequences=100000, start= -5.0, end=5.0)  # 예: 200개의 테스트 시퀀스 생성
test_dataset = SequenceDataset(test_sequences)

# 데이터 로더 설정
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 손실 함수와 옵티마이저

# 테스트 실행
test(model, test_loader)

