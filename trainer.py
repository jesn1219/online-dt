"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time
import wandb


class SequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        log_temperature_optimizer,
        scheduler=None,
        device="cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.device = device
        self.start_time = time.time()

# jesnk : made
    def pretrain_train_iteration(
        self,
        loss_fn,
        dataloader,
        jesnk_logger = None,
    ):

        losses, nlls, entropies = [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        if jesnk_logger is not None:
            jesnk_logger.info(f"pretrain_train_iteration, dataloader length : {len(dataloader)}")

        for _, trajs in enumerate(dataloader):
            loss, nll, entropy = self.train_step_stochastic(loss_fn, trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)
            # jesnk : assume that one loop is one episode
            wandb.log({"pretrain/training/loss": loss, \
                       "pretrain/training/nll": nll, \
                        "pretrain/training/entropies": entropy, \
                            "pretrain/training/episode" : _, \
                                "pretrain/training/temperatures" : self.model.temperature().detach().cpu().item()})

            

        logs["pretrain/time/training"] = time.time() - train_start
        logs["pretrain/training/train_loss_mean"] = np.mean(losses)
        logs["pretrain/training/train_loss_std"] = np.std(losses)
        logs["pretrain/training/nll"] = nlls[-1]
        logs["pretrain/training/entropy"] = entropies[-1]
        logs["pretrain/training/temp_value"] = self.model.temperature().detach().cpu().item()
        logs["pretrain/training/last_episode"] = _ 

        if jesnk_logger is not None:
            jesnk_logger.info(f"pretrain_train_iteration is done")
        

        return logs



    def ot_train_iteration(
        self,
        loss_fn,
        dataloader,
        jesnk_logger = None,
        start_episode_num = None,
    ):
        

        logging_prefix = "ot/training"


        losses, nlls, entropies = [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        if jesnk_logger is not None:
            jesnk_logger.info(f"ot_train_iteration, dataloader length : {len(dataloader)}")

        for _, trajs in enumerate(dataloader):
            loss, nll, entropy = self.train_step_stochastic(loss_fn, trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)
            wandb.log({logging_prefix + "loss": loss, \
                       logging_prefix + "nll": nll, \
                        logging_prefix + "entropies": entropy, \
                            logging_prefix + "episode" : start_episode_num + _, \
                                logging_prefix + "temperature" : self.model.temperature().detach().cpu().item()}
            )

        logs["ot/time/training"] = time.time() - train_start
        logs["ot/training/train_loss_mean"] = np.mean(losses)
        logs["ot/training/train_loss_std"] = np.std(losses)
        logs["ot/training/nll"] = nlls[-1]
        logs["ot/training/entropy"] = entropies[-1]
        logs["ot/training/temp_value"] = self.model.temperature().detach().cpu().item()
        
        if jesnk_logger is not None:
            jesnk_logger.info(f"ot_train_iteration is done")

        return logs

    def train_step_stochastic(self, loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        action_target = torch.clone(actions)

        _, action_preds, _ = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        loss, nll, entropy = loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            padding_mask,
            self.model.temperature().detach(),  # no gradient taken here
        )
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
