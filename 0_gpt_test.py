#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import random

# 시퀀스 데이터 생성 함수
def create_sequences(seq_length=5, num_sequences=1000, start=-1.0, end=1.0, step=0.01):
    sequences = []
    while len(sequences) < num_sequences:
        start_val = random.uniform(start, end - seq_length * step)
        sequence = [start_val + i * step for i in range(seq_length)]
        if all(-1 <= x <= 1 for x in sequence):  # 시퀀스 내 모든 값이 -1과 1 사이인지 확인
            sequences.append(sequence)
    return np.array(sequences)

# 데이터셋 클래스
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_seq = sequence[:-1]  # 입력 시퀀스 (마지막 토큰 제외)
        target_seq = sequence[1:]  # 타겟 시퀀스 (첫 번째 토큰 제외)
        return torch.tensor(input_seq, dtype=torch.float), torch.tensor(target_seq, dtype=torch.float)


seq_data_length = 5  # 시퀀스 데이터 길이
seq_length = seq_data_length -1  # 입력 시퀀스 길이
num_sequences = 100000  # 시퀀스 데이터 개수


# 모델 초기화
embed_dim = 1024 # 임베딩 차원
num_heads = 4   # 어텐션 헤드 수
num_layers = 6  # 트랜스포머 블록 수
batch_size = 1024

total_epoch = 100
gamma = 0.9

# 시퀀스 데이터 생성 및 데이터셋 객체 생성
sequences = create_sequences(seq_length=seq_data_length, num_sequences=num_sequences)


dataset = SequenceDataset(sequences)

# 데이터셋 예시 출력
print("시퀀스 데이터 예시:", sequences[:5])
print("데이터셋 크기:", len(dataset))


# In[2]:


class GPT2Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(GPT2Block, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # 멀티-헤드 어텐션
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)

        # 피드-포워드 네트워크
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.layer_norm2(x)

        return x

class GPT2(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, seq_length):
        super(GPT2, self).__init__()
        self.embed_dim = embed_dim
        self.positional_embeddings = nn.Parameter(torch.randn(seq_length, embed_dim))
        self.blocks = nn.ModuleList([GPT2Block(embed_dim, num_heads) for _ in range(num_layers)])
        self.final_layer = nn.Linear(embed_dim, 1)  # 시퀀스의 각 위치에 대한 값을 예측

    def forward(self, x):
        x = x + self.positional_embeddings[:x.size(1), :]
        for block in self.blocks:
            x = block(x)
        x = self.final_layer(x)
        return x.squeeze(-1)
    def save(self, save_path):
        try:
            torch.save({
                'model_state_dict': self.state_dict(),
                'model_class': self.__class__,
                'model_args': {'embed_dim': self.embed_dim, 'num_heads': num_heads, 'num_layers': num_layers, 'seq_length': seq_length}
            }, save_path)
        except Exception as e:
            print(f"모델 저장 중 오류 발생: {e}")

    def load(self, load_path, device):
        try:
            checkpoint = torch.load(load_path, map_location=device)
            model_class = checkpoint['model_class']
            model_args = checkpoint['model_args']
            model = model_class(**model_args).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            return None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2(embed_dim, num_heads, num_layers, seq_length)
model



# In[10]:


import time
# 데이터 로더 설정
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 손실 함수와 옵티마이저
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=gamma)

# 학습 루프
def train(model, train_loader, criterion, optimizer, scheduler, epochs=10):
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        total_token_error = []
        epoch_start_time = time.time()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(-1))
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            token_error = torch.abs(targets.detach() - outputs.detach()).mean()
            total_token_error.append(token_error.cpu())



            total_loss += loss.item()
        scheduler.step()
        epoch_end_time = time.time()
        epoch_token_error = np.mean(total_token_error)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.7f}, token error : {epoch_token_error}Time: {epoch_end_time - epoch_start_time:.4f}s")
    end_time = time.time()
    print(f"Total Learning time : {end_time - start_time:.4f}s")
    return epoch_token_error

model.to(device)

last_epoch_token_error = train(model, train_loader, criterion, optimizer, scheduler, epochs=total_epoch)
last_epoch_token_error = str(round(last_epoch_token_error, 9))
timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
model_name = f"gpt2_ed{embed_dim}_nh{num_heads}_nl{num_layers}_sdl{seq_data_length}_ns{num_sequences}_g{gamma}_epoch{total_epoch}_tte{last_epoch_token_error}_{timestamp}.pt"
model.save(f"./0_gpt_trained_model/{model_name}")


# In[4]:


load = False 

if load :
    embed_dim = 2048 # 임베딩 차원
    num_heads = 4   # 어텐션 헤드 수
    num_layers = 6  # 트랜스포머 블록 수
    seq_data_length = 5  # 시퀀스 데이터 길이
    seq_length = seq_data_length - 1  # 입력 시퀀스 길이
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_model_name = f"./0_gpt_trained_model/gpt2_{embed_dim}_{num_heads}_{num_layers}_{seq_data_length}.pt"
    model = GPT2(embed_dim, num_heads, num_layers, seq_length).to(device)
    model.load(load_model_name, device)


# In[5]:


def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    total_token_error = []
    count = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            #print(inputs.shape, targets.shape)
            outputs = model(inputs.unsqueeze(-1))
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            #print(f"Input: {inputs[0].tolist()}, \nTarget: {targets[0].tolist()}, \nPrediction: {outputs[0].tolist()}")
            # each token error mean
            #print(f"Error: {torch.abs(targets[0] - outputs[0]).mean():.4f}\n")
            # token error mean
            token_error = torch.abs(targets - outputs).mean()

            total_token_error.append(token_error.cpu())
            count += 1
    avg_loss = total_loss / len(test_loader)
    avg_token_error = np.mean(total_token_error)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Token Error: {avg_token_error:.4f}")
    print(f"count : {count}")

# 테스트 데이터 생성 및 데이터셋 객체 생성
test_sequences = create_sequences(seq_length=seq_data_length, num_sequences=100000)  # 예: 200개의 테스트 시퀀스 생성
test_dataset = SequenceDataset(test_sequences)

# 데이터 로더 설정
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 손실 함수와 옵티마이저
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


# 테스트 실행
test(model, test_loader, criterion)


# Test Token Error: 
# ## embed_size : 512
# 
# * seq 41 : 0.0109
# * seq 21 : 0.0176
# * seq 5 : 0.0191
# * seq 3 : 0.0192
# 
# ## embed_size : 1024

# In[6]:


def predict_next_token(model, token):
    model.eval()
    with torch.no_grad():
        # 입력 차원을 (1, 1, 1)로 변경
        token_tensor = torch.tensor([[token]], dtype=torch.float).unsqueeze(-1)
        # 모델의 예측
        predicted_token = model(token_tensor)
        return predicted_token.item()

# 테스트용 단일 토큰 예시
test_tokens = [0.005, -0.010, 0.500, -0.700, 0.999]

# 각 토큰에 대해 다음 토큰 예측
for token in test_tokens:
    predicted_next_token = predict_next_token(model, token)
    print(f"입력 토큰: {token}, 예측된 다음 토큰: {predicted_next_token}")

