#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model
import math
import numpy as np
import torch.nn.functional as F
from torch import distributions as pyd


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))



class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Squashed Normal Distribution(s)

    If loc/std is of size (batch_size, sequence length, d),
    this returns batch_size * sequence length * d
    independent squashed univariate normal distributions.
    """

    def __init__(self, loc, std):
        self.loc = loc
        self.std = std
        self.base_dist = pyd.Normal(loc, std)

 
        transforms = [TanhTransform()]

        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self, N=1):
        # sample from the distribution and then compute
        # the empirical entropy:
        x = self.rsample((N,))
        log_p = self.log_prob(x)

        # log_p: (batch_size, context_len, action_dim),
        return -log_p.mean(axis=0).sum(axis=2)

    def log_likelihood(self, x):
        # log_prob(x): (batch_size, context_len, action_dim)
        # sum up along the action dimensions
        # Return tensor shape: (batch_size, context_len)

        x = self.transforms[0](x)        
        return self.log_prob(x).sum(axis=2)

        # jesnk : this is the log likelihood of the state # jesnk: mark1
        for tr in reversed(self.transforms):
            x = tr.inv(x)
        return self.base_dist.log_prob(x).sum(axis=2)



class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, hidden_dim, act_dim, log_std_bounds=[-5.0, 2.0]):
        super().__init__()

        self.mu = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std_bounds = log_std_bounds
        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.mu(obs), self.log_std(obs)
        print(mu.shape, log_std.shape)
        log_std = torch.tanh(log_std)
        # log_std is the output of tanh so it will be between [-1, 1]
        # map it to be between [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        std = log_std.exp()
        return SquashedNormal(mu, std)


from decision_transformer.models.model import TrajectoryModel01
class DecisionTransformer02(TrajectoryModel01):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    """
    jesnk: this model uses gpt to model (state_1, state_2, ..., state_n)
    
    """
    def __init__(
        self,
        state_dim,
        hidden_size,
        state_range,
        ordering=0,
        act_dim = None,
        max_length=None,
        eval_context_length=None,
        max_ep_len=4096,
        state_tanh=True,
        stochastic_policy=False,
        init_temperature=0.1,
        target_entropy=None,
        state_mean=None, #jesnk
        state_std=None, #jesnk
        **kwargs
    ):
        super().__init__(state_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        if ordering:
            self.embed_ordering = nn.Embedding(max_ep_len, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        
        self.embed_ln = nn.LayerNorm(hidden_size)

        if stochastic_policy:
            self.predict_state = DiagGaussianActor(hidden_size, self.state_dim)
        else:
            self.predict_state = nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.state_dim)]
                    + ([nn.Tanh()] if state_tanh else [])
                )
            )
        self.stochastic_policy = stochastic_policy
        self.eval_context_length = eval_context_length
        self.ordering = ordering
        self.state_range = state_range
        #self.state_mean = state_mean # jesnk
        #self.state_std = state_std # jesnk
        #print(f'jesnk: debug: DT01: state_mean:{self.state_mean}, state_std:{self.state_std}')


        if stochastic_policy:
            self.log_temperature = torch.tensor(np.log(init_temperature))
            self.log_temperature.requires_grad = True
            self.target_entropy = target_entropy

    def temperature(self):
        if self.stochastic_policy:
            return self.log_temperature.exp()
        else:
            return None

    def forward(
        self,
        states,
        timesteps=None,
        ordering=None,
        padding_mask=None,
    ):

        batch_size, seq_length = states.shape[0], states.shape[1] # 512, seq_legnth, 

        if padding_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            padding_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        if timesteps is None :
            timesteps = torch.arange(seq_length, device=states.device).repeat(batch_size, 1)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)

        if self.ordering:
            order_embeddings = self.embed_ordering(timesteps)
        else:
            order_embeddings = 0.0

        state_embeddings = state_embeddings + order_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        # state_embeddings.shape: batch, seq_length, hidden_size
        stacked_inputs = (
            #torch.stack((state_embeddings), dim=1)
            state_embeddings # batch, seq_length, hidden_size
            #.permute(0, 2, 1, 3) # batch, seq_length, 1, hidden_size
            .reshape(batch_size, 1 * seq_length, self.hidden_size)
        )
        # stacked_inputs.shape: batch, 1*seq_length, hidden_size
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_padding_mask = (
            #torch.stack((padding_mask, padding_mask, padding_mask), dim=1)
            padding_mask
            #.permute(0, 2, 1)
            .reshape(batch_size, 1 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_padding_mask,
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 1, self.hidden_size).permute(0, 2, 1, 3)
        # after reshape : batch, 1, seq_length, hidden_size
        # get predictions
        # predict next state given state and action

        #state_preds = self.predict_state(x[:, 0]) # jesnk: DT1 setting
        states_pred = self.predict_state(x[:,0]) # jesnk: must check the index of [:, 0] is correct

        # state_preds.shape: batch, seq_length, state_dim
        return states_pred

    def get_predictions(
        self, states, timesteps, num_envs=1, **kwargs
    ):
        # we don't care about the past rewards in this model
        # tensor shape: batch_size, seq_length, variable_dim
        states = states.reshape(num_envs, -1, self.state_dim)

        # tensor shape: batch_size, seq_length
        timesteps = timesteps.reshape(num_envs, -1)

        # max_length is the DT context length (should be input length of the subsequence)
        # eval_context_length is the how long you want to use the history for your prediction
        if self.max_length is not None:
            states = states[:, -self.eval_context_length :]
            timesteps = timesteps[:, -self.eval_context_length :]

            ordering = torch.tile(
                torch.arange(timesteps.shape[1], device=states.device),
                (num_envs, 1),
            )
            # pad all tokens to sequence length
            padding_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            padding_mask = padding_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            padding_mask = padding_mask.repeat((num_envs, 1))

            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)

            ordering = torch.cat(
                [
                    torch.zeros(
                        (ordering.shape[0], self.max_length - ordering.shape[1]),
                        device=ordering.device,
                    ),
                    ordering,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            padding_mask = None

        state_preds = self.forward(
            states,
            timesteps,
            ordering,
            padding_mask=padding_mask,
            **kwargs
        )
        if self.stochastic_policy:
            return state_preds
        else:
            return (
                self.clamp_state(state_preds[:, -1])
            )

    def clamp_state(self, state):
        return state.clamp(*self.state_range)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        return self
    


# In[2]:


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
        #self.sequences = sequences
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequences = torch.tensor(sequences, dtype=torch.float).to(device)
        
        # 

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_seq = sequence[:-1]  # 입력 시퀀스 (마지막 토큰 제외)
        target_seq = sequence[1:]  # 타겟 시퀀스 (첫 번째 토큰 제외)
        return input_seq.clone().detach(), target_seq.clone().detach()


seq_data_length = 5  # 시퀀스 데이터 길이
seq_length = seq_data_length -1  # 입력 시퀀스 길이
num_sequences = 500000  # 시퀀스 데이터 개수


# 모델 초기화
#embed_dim = 512 # 임베딩 차원
#num_heads = 4   # 어텐션 헤드 수
#num_layers = 6  # 트랜스포머 블록 수
#batch_size = 1
#lr = 1e-6

total_epoch = 100

# 시퀀스 데이터 생성 및 데이터셋 객체 생성
sequences = create_sequences(seq_length=seq_data_length, num_sequences=num_sequences)
dataset = SequenceDataset(sequences)

# 데이터셋 예시 출력
print("시퀀스 데이터 예시:", sequences[:5])
print("데이터셋 크기:", len(dataset))


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
variant = {}
variant["state_dim"] = 1
variant["state_range"] = np.array([-1.0, 1.0])
variant["K"] = 5
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
variant["dataset_num_squence"] = num_sequences
variant["dataset_seq_length"] = seq_data_length
variant["train_total_epoch"] = total_epoch
variant["stocastic_policy"] = False

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
    #state_mean=self.state_mean,
    #state_std=self.state_std,
).to(device=device)
    


# In[4]:


def train_step_stochastic_01(self, loss_fn, trajs):
    (
        states,
        rewards,
        dones,
        rtg,
        timesteps,
        ordering,
        padding_mask,
    ) = trajs
    

    states = states.to(self.device)
    rewards = rewards.to(self.device)
    dones = dones.to(self.device)
    rtg = rtg.to(self.device)
    timesteps = timesteps.to(self.device)
    ordering = ordering.to(self.device)
    padding_mask = padding_mask.to(self.device)
    #print(f"training_input {states[0]}")
    #print(f"padding_mask:{padding_mask[0]}")
    #print(f"timesetps:{timesteps[0]}")
    #print(f"ordering:{ordering[0]}")
    state_target = torch.clone(states)

    states_preds = self.model.forward(
        states,
        timesteps,
        ordering,
        padding_mask=padding_mask,
    )

    loss, nll, entropy = loss_fn(
        states_preds,  # a_hat_dist
        state_target,
        padding_mask,
        self.model.temperature().detach(),  # no gradient taken here
    )
    #print(f"state_target : {state_target[0]}")
    #print(f"state_preds : {states_preds.mean[0]}")
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
    self.optimizer.step()

    self.log_temperature_optimizer.zero_grad()
    temperature_loss = (
        self.model.temperature() * (entropy - self.model.target_entropy).detach()
    )
    temperature_loss.backward()
    self.log_temperature_optimizer.step()

    if self.scheduler is not None:
        self.scheduler.step()

    return (
        loss.detach().cpu().item(),
        nll.detach().cpu().item(),
        entropy.detach().cpu().item(),
    )


# In[5]:


import time
from lamb import Lamb
import wandb

# 데이터 로더 설정
batch_size = 1024

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 손실 함수와 옵티마이저
def loss_fn_stocastic(
    s_hat_dist,
    s,
    attention_mask,
    entropy_reg,
):
    # a_hat is a SquashedNormal Distribution
    log_likelihood = s_hat_dist.log_likelihood(s)[attention_mask > 0].mean()

    entropy = s_hat_dist.entropy().mean()
    loss = -(log_likelihood + entropy_reg * entropy)

    return (
        loss,
        -log_likelihood,
        entropy,
    )
    
def loss_fn_deterministic(
    s_hat,
    s,
    attention_mask
):
    # MSE LOSS with attention_mask > 0
    s_hat = s_hat
    s = s
    #print(s_hat.shape, s.shape)
    loss = torch.mean((s_hat - s)**2)
    return loss


optimizer = Lamb(
            model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
)
scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
)


# 학습 루프
def train(model, train_loader, loss_fn, optimizer, scheduler, epochs=10, log_dir='./0_gpt_trained_model/'):
    model.train()
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    embed_dim = variant["embed_dim"]
    num_heads = variant["n_head"]
    num_layers = variant["n_layer"]
    seq_data_length = variant["dataset_seq_length"]
    num_sequences = variant["dataset_num_squence"]
    lr = variant["learning_rate"]
    gamma = variant["weight_decay"]
    total_epoch = variant["train_total_epoch"]
    stocastic_policy = variant["stocastic_policy"]
    
    if stocastic_policy:
        log_temperature_optimizer = torch.optim.Adam(
            [model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )
    
    model_name = f"DT_ed{embed_dim}_nh{num_heads}_nl{num_layers}_sdl{seq_data_length}_ns{num_sequences}_lr{lr}_g{gamma}_epoch{total_epoch}_{timestamp}"
    log_dir = log_dir + f"{model_name}/"
    try:
        os.mkdir(log_dir)
    except:
        pass

    import os
    try:
        os.mkdir(log_dir)
    except:
        pass
    
    start_time = time.time()
    for epoch in range(epochs):
        total_token_error = []
        epoch_start_time = time.time()
        total_loss = 0
        for inputs, targets in train_loader:

            optimizer.zero_grad()

            outputs = model(inputs.unsqueeze(-1))
            
            padding_mask = torch.ones((inputs.shape[0], seq_length), dtype=torch.long)
            
            if stocastic_policy:
                loss, nll, entropy = loss_fn(outputs, targets.unsqueeze(-1),attention_mask=padding_mask,entropy_reg=model.temperature().detach())
            else :
                loss = loss_fn(outputs, targets.unsqueeze(-1),attention_mask=padding_mask)
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            
            if stocastic_policy:
                log_temperature_optimizer.zero_grad()
                temperature_loss = (
                    model.temperature() * (entropy - model.target_entropy).detach()
                )
                temperature_loss.backward()
                log_temperature_optimizer.step()
            if stocastic_policy:
                output_sample = outputs.mean
            else :
                output_sample = outputs
            token_error = torch.abs(targets.detach() - output_sample.detach().reshape(targets.shape)).mean()
            total_token_error.append(token_error.cpu())

            total_loss += loss.item()
            if wandb.run is not None:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "nll": nll.item(),
                        "entropy": entropy.item(),
                        "token_error": token_error.item(),
                        "temperature": model.temperature().item(),
                    }
                )
        scheduler.step()
        epoch_end_time = time.time()
        epoch_token_error = np.mean(total_token_error)
        avg_loss = total_loss / len(train_loader)
        
        
        if wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "avg_loss": avg_loss,
                    "epoch_token_error": epoch_token_error,
                }
            )
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.7f}, token error : {epoch_token_error}Time: {epoch_end_time - epoch_start_time:.4f}s")
        if stocastic_policy:
            print(f"entropy : {entropy}, temperature : {model.temperature()}")
        if epoch % 10 == 0:
            model_name = f"gpt2_ed{embed_dim}_nh{num_heads}_nl{num_layers}_sdl{seq_data_length}_ns{num_sequences}_lr{lr}_g{gamma}_epoch{total_epoch}_tte{epoch_token_error}_ep{epoch}.pt"
            pass
            model.save(f"{log_dir}{model_name}")

    end_time = time.time()
    print(f"Total Learning time : {end_time - start_time:.4f}s")
    return epoch_token_error

from jesnk_utils.utils import get_current_time

current_time = get_current_time()

wandb_enable = True
if wandb_enable :
    wandb.init(project="GPT_exp", entity="jesnk", name=f"DT02_DET_{current_time}")
    wandb.config.update(variant)
    wandb.watch(model)


model.to(device)

loss_fn = loss_fn_stocastic if variant["stocastic_policy"] else loss_fn_deterministic
last_epoch_token_error = train(model, train_loader, loss_fn, optimizer, scheduler, epochs=total_epoch)
last_epoch_token_error = str(round(last_epoch_token_error, 9))


# In[ ]:


def test(model, test_loader):
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

            loss, nll, entropy = loss_fn(outputs, targets.unsqueeze(-1),attention_mask=padding_mask,entropy_reg=model.temperature().detach())
            total_loss += loss.item()
            
            output_sample = outputs.mean
            token_error = torch.abs(targets.detach() - output_sample.detach().reshape(targets.shape)).mean()
            print("="*20)
            print(f"input : {inputs[0]}")
            print(f"output : {outputs.mean[0]}")
            print(f"target : {targets[0]}")
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

# 테스트 실행
test(model, test_loader)

