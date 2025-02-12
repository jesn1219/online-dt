"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

from torch.utils.tensorboard import SummaryWriter
import os 
# Set system environment variables
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
os.environ["DISPLAY"]=":11.0"
import argparse
import pickle
import random
import time
#import gym
import d4rl
import torch
import numpy as np

import utils
from replay_buffer import ReplayBuffer, ReplayBuffer01
from lamb import Lamb
from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from data import create_dataloader, create_dataloader_01
from decision_transformer.models.decision_transformer import DecisionTransformer, DecisionTransformer01
from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from trainer import SequenceTrainer
from logger import Logger
import wandb
import logging
from jesnk_utils.utils import get_current_time
# jesnk
# set up logger, file mode "a" means append, 

def init_jesnk_logger():
    global jesnk_logger
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logging.basicConfig(level=logging.DEBUG, format=formatter)
    jesnk_logger = logging.getLogger(__name__)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    jesnk_logger.addHandler(ch)
    jesnk_logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('jesnk.log', delay=True)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    jesnk_logger.addHandler(file_handler)

    jesnk_logger.info("Starting logger")


MAX_EPISODE_LEN = 1000


class Experiment:
    def __init__(self, variant):
        init_jesnk_logger()
        
        jesnk_logger.info("Starting experiment")
        jesnk_logger.info(f"args: {variant}")


        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(variant)
        self.variant = variant
        # initialize by offline trajs        
        if not ("model:01" in variant["tags"]) :
            self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(
                variant["env"]
            )
            self.replay_buffer = ReplayBuffer(variant["replay_size"], self.offline_trajs)

        else :            
            self.offline_trajs, self.state_mean, self.state_std = self._load_dataset_01(
                variant["env"]
            )
            # if some string "s_range*" is in variant['tags'], then set state_range
            # Setting state_range
            for tag in variant['tags'].split(',') :
                print(f"checking tag : {tag}")
                if 's_range' in tag :
                    value = tag[7:]
                    print("tag value: ", value)
                    value_min, value_max = value.split(':')
                    self.state_range = [float(value_min), float(value_max)]
                    print(f"state_range : {self.state_range}")
                    break
            # if state_range is not set, then set state_range to [-5, 5]
            if not hasattr(self, 'state_range') :
                self.state_range = [-5, 5]
            self.replay_buffer = ReplayBuffer01(variant["replay_size"], self.offline_trajs)
        print(f'jesnk: debug: state_mean:{self.state_mean}, state_std:{self.state_std}')

        self.aug_trajs = []

        self.device = variant.get("device", "cuda")
        if not ("model:01" in variant["tags"]) :
            self.target_entropy = -self.act_dim
        else :
            self.target_entropy = -self.state_dim
        
        
        if not ("model:01" in variant["tags"]) :
            
            self.model = DecisionTransformer(
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                action_range=self.action_range,
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
                stochastic_policy=True,
                ordering=variant["ordering"],
                init_temperature=variant["init_temperature"],
                target_entropy=self.target_entropy,
            ).to(device=self.device)
        else :
            self.model = DecisionTransformer01(
                state_dim=self.state_dim,
                state_range= self.state_range,
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
                stochastic_policy=True,
                ordering=variant["ordering"],
                init_temperature=variant["init_temperature"],
                target_entropy=self.target_entropy,
                #state_mean=self.state_mean,
                #state_std=self.state_std,
            ).to(device=self.device)
                

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
        )

        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        #self.variant = variant #jesnk: move to upper
        self.reward_scale = 1.0 if "antmaze" in variant["env"] else 0.001
        self.logger = Logger(variant)

    def _get_env_spec(self, variant):
        if 'minari' in variant['tags'] :
            import gymnasium as gym
            env_name = variant["env"]
            print(f'minari env_name : {env_name}')
            env = gym.make(env_name)
            state_dim = env.observation_space['observation'].shape[0]
            act_dim = env.action_space.shape[0]
            action_range = [
                float(env.action_space.low.min()) + 1e-6,
                float(env.action_space.high.max()) - 1e-6,
            ]
            env.close()
        else :
            import gym
            env = gym.make(variant["env"])
            state_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            action_range = [
                float(env.action_space.low.min()) + 1e-6,
                float(env.action_space.high.max()) - 1e-6,
            ]
            env.close()
        return state_dim, act_dim, action_range

    def _save_model(self, path_prefix, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }

        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")

    def _load_model(self, path_prefix):
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint["log_temperature_optimizer_state_dict"]
            )
            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.online_iter = checkpoint["online_iter"]
            self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
            print(f"Model loaded at {path_prefix}/model.pt")

    def _load_dataset(self, env_name):
        if 'minari' in self.variant['tags'] :
            trajectories, state_mean, state_std = self._load_dataset_minari(env_name)
            return trajectories, state_mean, state_std

        dataset_path = f"./data/{env_name}.pkl"
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)

        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: {env_name}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        trajectories = [trajectories[ii] for ii in sorted_inds]

        return trajectories, state_mean, state_std
    
    def _load_dataset_minari(self, env_name): # jensk
        import minari
        if env_name == 'PointMaze_UMaze-v3' :
            env_name = 'pointmaze-umaze-v1'
        dataset = minari.load_dataset(env_name)
        trajectories = dataset._data.get_episodes(dataset.episode_indices)
        states, traj_lens, returns = [], [], []
        if 'pointmaze' in env_name :
            # re-label observation. (achieved_goal, desired_goal) -> observation
            print("re-label observation. (achieved_goal, desired_goal) -> observation")
            for path in trajectories :
                achieved_goal = path['observations']['achieved_goal'][1:]
                desired_goal = path['observations']['desired_goal'][1:]
                observation = np.concatenate([achieved_goal, desired_goal], axis=1)
                path['observations'] = observation

        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
            # for pointmaze
        traj_lens, returns = np.array(traj_lens), np.array(returns)
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: {env_name}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]

        trajectories = [trajectories[ii] for ii in sorted_inds]
        return trajectories, state_mean, state_std
            
    
    def _load_dataset_01(self,env_name) :

        offline_trajs, state_mean, state_std = self._load_dataset(env_name)
        # create ['K'] length trajectories with initial state and final state
        
        #return offline_trajs, state_mean, state_std
        offline_trajs_01 = []
        for traj in offline_trajs:
            traj_len = len(traj["observations"])
            print(f"traj len: {traj_len}")
            # divide the trajectory into K length. get index of K length
            if traj_len < self.variant["K"] :
                continue 
            index = np.linspace(0, traj_len - 1, self.variant["K"]).astype(int)
            #print(f"traj len: {traj_len}, index : {index}")
            # get the state and action of the index
            # get the initial action and final action
            if 'terminations' in traj.keys() :
                traj['terminals'] = traj['terminations']
                del traj['terminations']
            
            for key in traj.keys() :
                if key in ['id', 'total_timesteps', 'seed' ] :
                    continue
                traj[key] = traj[key][index]
            print(f"{traj['observations']}")
            offline_trajs_01.append(traj)
        print(f"offline_trajs_01 : {len(offline_trajs_01)}")
        return offline_trajs_01, state_mean, state_std
                
            
            
            

    def  _augment_trajectories(
        self,
        online_envs,
        target_explore,
        n,
        randomized=False,
    ):

        max_ep_len = MAX_EPISODE_LEN

        with torch.no_grad():
            # generate init state
            target_return = [target_explore * self.reward_scale] * online_envs.num_envs

            returns, lengths, trajs = vec_evaluate_episode_rtg(
                online_envs,
                self.state_dim,
                self.act_dim,
                self.model,
                max_ep_len=max_ep_len,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=False,
            )

        self.replay_buffer.add_new_trajs(trajs)
        self.aug_trajs += trajs
        self.total_transitions_sampled += np.sum(lengths)

        return {
            "aug_traj/return": np.mean(returns),
            "aug_traj/length": np.mean(lengths),
        }

    def pretrain(self, eval_envs, loss_fn):
        print("\n\n\n*** Pretrain ***")

        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
        jesnk_logger.info(f"Start pretraining, max_pretrain_iters : {self.variant['max_pretrain_iters']}")
        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            jesnk_logger.info(f"pretrain_iter : {self.pretrain_iter}")
            # in every iteration, prepare the data loader
            if not ("model:01" in self.variant["tags"]) :
                dataloader = create_dataloader(
                    trajectories=self.offline_trajs,
                    num_iters=self.variant["num_updates_per_pretrain_iter"],
                    batch_size=self.variant["batch_size"],
                    max_len=self.variant["K"],
                    state_dim=self.state_dim,
                    act_dim=self.act_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    reward_scale=self.reward_scale,
                    action_range=self.action_range,
                )
            else :
                dataloader = create_dataloader_01(
                    trajectories=self.offline_trajs,
                    num_iters=self.variant["num_updates_per_pretrain_iter"],
                    batch_size=self.variant["batch_size"],
                    max_len=self.variant["K"],
                    state_dim=self.state_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    reward_scale=self.reward_scale,
                    state_range= self.state_range
                )
            
            # len(dataloader)
            jesnk_logger.info(f"len(dataloader) : {len(dataloader)}")
            # jesnk : maybe, data were loaded up to max_updates_per_pretrain_iter
            train_outputs = trainer.pretrain_train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            outputs = {"time/total": time.time() - self.start_time}
            if not ("model:01" in self.variant["tags"]) :
                eval_outputs, eval_reward = self.evaluate(eval_fns)
                outputs.update(eval_outputs)

            outputs.update(train_outputs)
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=True,
            )

            self.pretrain_iter += 1
        jesnk_logger.info(f"End pretraining, max_pretrain_iters : {self.variant['max_pretrain_iters']}")

    def evaluate(self, eval_fns):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        for eval_fn in eval_fns:
            o = eval_fn(self.model)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs, eval_reward

    def online_tuning(self, online_envs, eval_envs, loss_fn):

        print("\n\n\n*** Online Finetuning ***")

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]
        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
        jesnk_logger.info(f"Start online tuning, max_online_iters : {self.variant['max_online_iters']}")
        jesnk_logger.info(f"num_updates_per_online_iter : {self.variant['num_updates_per_online_iter']}")
        jesnk_logger.info(f"num_online_rollouts : {self.variant['num_online_rollouts']}")

        start_episode_num = 0 # jesnk
        while self.online_iter < self.variant["max_online_iters"]:
            jesnk_logger.info(f"online_iter : {self.online_iter}")

            outputs = {}
            augment_outputs = self._augment_trajectories(
                online_envs,
                self.variant["online_rtg"],
                n=self.variant["num_online_rollouts"],
            )
            outputs.update(augment_outputs)
            if not ("model:01" in self.variant["tags"]) :
                dataloader = create_dataloader(
                    trajectories=self.replay_buffer.trajectories,
                    num_iters=self.variant["num_updates_per_online_iter"],
                    batch_size=self.variant["batch_size"],
                    max_len=self.variant["K"],
                    state_dim=self.state_dim,
                    act_dim=self.act_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    reward_scale=self.reward_scale,
                    action_range=self.action_range,
                )
            else :
                dataloader = create_dataloader_01(
                    trajectories=self.offline_trajs,
                    num_iters=self.variant["num_updates_per_pretrain_iter"],
                    batch_size=self.variant["batch_size"],
                    max_len=self.variant["K"],
                    state_dim=self.state_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    reward_scale=self.reward_scale,
                    state_range= self.state_range
                )
                
            # finetuning
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant[
                "eval_interval"
            ] == 0 or is_last_iter:
                evaluation = True
            else:
                evaluation = False
            jesnk_logger.info(f"online tuning, start_episode_num : {start_episode_num}")
            train_outputs = trainer.ot_train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
                start_episode_num=start_episode_num+1, # jesnk
            )
            start_episode_num += self.variant["num_updates_per_online_iter"] # jesnk

            outputs.update(train_outputs)

            if evaluation:
                eval_outputs, eval_reward = self.evaluate(eval_fns)
                outputs.update(eval_outputs)

            outputs["time/total"] = time.time() - self.start_time

            # log the metrics
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter + self.online_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=False,
            )

            self.online_iter += 1

    def __call__(self):

        utils.set_seed_everywhere(self.variant['seed'])

        import d4rl
        if not ("model:01" in self.variant["tags"]) :
            def loss_fn(
                a_hat_dist,
                a,
                attention_mask,
                entropy_reg,
            ):
                # a_hat is a SquashedNormal Distribution
                log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

                entropy = a_hat_dist.entropy().mean()
                loss = -(log_likelihood + entropy_reg * entropy)

                return (
                    loss,
                    -log_likelihood,
                    entropy,
                )
        else :
            def loss_fn(
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
            

        def get_env_builder(seed, env_name, target_goal=None):
            def make_env_fn():
                import d4rl
                if not 'minari' in self.variant['tags'] :
                    env = gym.make(env_name)
                    env.seed(seed)
                    if hasattr(env.env, "wrapped_env"):
                        env.env.wrapped_env.seed(seed)
                    elif hasattr(env.env, "seed"):
                        env.env.seed(seed)
                    else:
                        pass
                    env.action_space.seed(seed)
                    env.observation_space.seed(seed)

                    if target_goal:
                        env.set_target_goal(target_goal)
                        print(f"Set the target goal to be {env.target_goal}")
                    return env
                else :
                    import gymnasium as gym
                    env = gym.make(env_name)

                    env.action_space.seed(seed)
                    env.observation_space.seed(seed)

                    if target_goal:
                        env.set_target_goal(target_goal)
                        print(f"Set the target goal to be {env.target_goal}")
                    return env

            return make_env_fn

        print("\n\nMaking Eval Env.....")
        env_name = self.variant["env"]
        if "antmaze" in env_name:
            env = gym.make(env_name)
            target_goal = env.target_goal
            env.close()
            print(f"Generated the fixed target goal: {target_goal}")
        else:
            target_goal = None
        eval_envs = SubprocVecEnv(
            [
                get_env_builder(i, env_name=env_name, target_goal=target_goal)
                for i in range(self.variant["num_eval_episodes"])
            ]
        )

        self.start_time = time.time()
        if self.variant["max_pretrain_iters"]:
            self.pretrain(eval_envs, loss_fn)
        # save model
        self._save_model(
            path_prefix=self.logger.log_path,
            is_pretrain_model=True,
        )
        
        
        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            online_envs = SubprocVecEnv(
                [
                    get_env_builder(i + 100, env_name=env_name, target_goal=target_goal)
                    for i in range(self.variant["num_online_rollouts"])
                ]
            )
            self.online_tuning(online_envs, eval_envs, loss_fn)
            online_envs.close()

        eval_envs.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--env", type=str, default="hopper-medium-v2")

    # model options
    # transformer options
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_context_length", type=int, default=5)
    # 0: no pos embedding others: absolute ordering
    parser.add_argument("--ordering", type=int, default=0)

    # shared evaluation options
    parser.add_argument("--eval_rtg", type=int, default=3600)
    parser.add_argument("--num_eval_episodes", type=int, default=10)

    # shared training options
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)

    # pretraining options
    parser.add_argument("--max_pretrain_iters", type=int, default=1)
    parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=5000)

    # finetuning options
    parser.add_argument("--max_online_iters", type=int, default=1500)
    parser.add_argument("--online_rtg", type=int, default=7200)
    parser.add_argument("--num_online_rollouts", type=int, default=1)
    parser.add_argument("--replay_size", type=int, default=1000)
    parser.add_argument("--num_updates_per_online_iter", type=int, default=300)
    parser.add_argument("--eval_interval", type=int, default=10)

    # environment options
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="default")
    
    # add wandb tags
    parser.add_argument("--tags", type=str, default="default")

    args = parser.parse_args()
    if 'minari' in args.tags :
        import gymnasium as gym
    else :
        import gym



    utils.set_seed_everywhere(args.seed)
    experiment = Experiment(vars(args))
    print("=" * 50)

    wandb_name = "{}-{}-{}".format(args.env, args.exp_name, get_current_time())

    wandb.init(
        project="online-dt",
        entity="jesnk",
        config=vars(args),
        name=wandb_name,
        reinit=True,
        tags=args.tags.split(","),
    )
    experiment()
