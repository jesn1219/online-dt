import time
from lamb import Lamb
import wandb
from jesnk_utils.utils import get_current_time
import asyncio
from jesnk_utils.telebot import Telebot
import torch
import numpy as np
telebot = Telebot()


# 손실 함수와 옵티마이저
def loss_fn_stochastic(
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



# 학습 루프
def train(model, variant, train_loader, loss_fn, optimizer, scheduler, log_dir='./0_gpt_trained_model/'):
    model.train()
    timestamp = get_current_time()
    embed_dim = variant["embed_dim"]
    num_heads = variant["n_head"]
    num_layers = variant["n_layer"]
    seq_data_length = variant["dataset_seq_length"]
    num_sequences = variant["dataset_num_squence"]
    lr = variant["learning_rate"]
    gamma = variant["weight_decay"]
    total_epoch = variant["train_total_epoch"]
    stocastic_policy = variant["stocastic_policy"]
    telebot_enable = variant["telebot"]
    max_epoch = variant['train_total_epoch']
    
    if stocastic_policy:
        log_temperature_optimizer = torch.optim.Adam(
            [model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )
    
    model_name = f"{timestamp}_DT_ed{embed_dim}_nh{num_heads}_nl{num_layers}_sdl{seq_data_length}_ns{num_sequences}_lr{lr}_g{gamma}_epoch{total_epoch}"
    log_dir = log_dir + f"{model_name}/"
    
    if telebot_enable:
        telebot.send_message(f"{get_current_time()} : start training {model_name}")

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
    for epoch in range(max_epoch):
        total_token_error = []
        epoch_start_time = time.time()
        total_loss = 0
        for inputs, targets in train_loader:

            optimizer.zero_grad()

            outputs = model(inputs.unsqueeze(-1))
            
            padding_mask = torch.ones((inputs.shape[0], variant["K"]), dtype=torch.long)
            
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
                if variant['stocastic_policy'] :
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "nll": nll.item(),
                            "entropy": entropy.item(),
                            "token_error": token_error.item(),
                            "temperature": model.temperature().item(),
                        }
                    )
                else :                        
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "token_error": token_error.item(),
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
        print(f"Epoch [{epoch + 1}/{max_epoch}], Loss: {avg_loss:.7f}, token error : {epoch_token_error}Time: {epoch_end_time - epoch_start_time:.4f}s")
        if stocastic_policy:
            print(f"entropy : {entropy}, temperature : {model.temperature()}")
        if epoch % 10 == 0:
            model_name = f"gpt2_ed{embed_dim}_nh{num_heads}_nl{num_layers}_sdl{seq_data_length}_ns{num_sequences}_lr{lr}_g{gamma}_epoch{total_epoch}_tte{epoch_token_error}_ep{epoch}.pt"
            model.save(f"{log_dir}{model_name}")
            if telebot_enable:
                telebot.send_message(f"{get_current_time()} : token error : {epoch_token_error}")


    end_time = time.time()
    print(f"Total Learning time : {end_time - start_time:.4f}s")
    if telebot_enable:
        telebot.send_message(f"{get_current_time()} : Finished. Total Learning time : {end_time - start_time:.4f}s")
    return epoch_token_error


def evaluation(model, test_loader, variant) :
    model.eval()
    total_token_error = []
    for inputs, targets in test_loader:
        outputs = model(inputs.unsqueeze(-1))
        token_error = torch.abs(targets.detach() - outputs.detach().reshape(targets.shape)).mean()
        total_token_error.append(token_error.cpu())
    epoch_token_error = np.mean(total_token_error)
    return epoch_token_error


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
