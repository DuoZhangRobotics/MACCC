# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import gym
import network_sim

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines3.common.policies import FeedForwardPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
import wandb
import os
import sys
import inspect
from datetime import datetime

time = f'{datetime.now().strftime("%Y-%m-%d-%H_%M_%S")}'
log_path = f"./log/{time}/"
timesteps = 1600 * 410 * 50
config = {
    "total_timesteps": timesteps,
    "env_name": 'PccNs-v0',
} 
run = wandb.init(
        project=f"NetworkProject",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        name=time
    )    
    
def check_paths():
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # if not os.path.exists(os.path.join(log_path, 'videos')):
    #     os.mkdir(os.path.join(log_path, 'videos'))
    # if os.path.exists(os.path.join(log_path, 'action.txt')):
    #     os.remove(os.path.join(log_path, 'action.txt'))
    if not os.path.exists(os.path.join(log_path, "models")):
        os.mkdir(os.path.join(log_path, "models"))
    if not os.path.exists(os.path.join(log_path, "runs")):
        os.mkdir(os.path.join(log_path, "runs")) 
        
def get_callbacks():
    eval_callback = EvalCallback(env, best_model_save_path=log_path,
                             log_path=log_path, eval_freq=100,
                             deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_path)
    wandb_callback = WandbCallback(
                        # gradient_save_freq=1e4,
                        model_save_path=os.path.join(log_path, f"models/{run.name}"),
                        model_save_freq=1e3,
                        verbose=2)
    callback = CallbackList([checkpoint_callback, 
                            #  eval_callback, 
                             wandb_callback
                             ])
    return callback

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from common.simple_arg_parse import arg_or_default

arch_str = arg_or_default("--arch", default="32,16")
if arch_str == "":
    arch = []
else:
    arch = [int(layer_width) for layer_width in arch_str.split(",")]
print("Architecture is: %s" % str(arch))

env = gym.make('PccNs-v0')
check_paths()
gamma = arg_or_default("--gamma", default=0.99)
print("gamma = %f" % gamma)
policy_kwargs = dict(net_arch=[{"pi":arch, "vf":arch}])
model = PPO("MlpPolicy",
            env,
            policy_kwargs=policy_kwargs, 
            verbose=1, 
            n_steps=8192, 
            batch_size=2048, 
            gamma=gamma,
            use_sde=False,
            tensorboard_log=os.path.join(log_path, f"runs/{run.name}")
            )

# for i in range(0, 6):
model.learn(total_timesteps=timesteps
            , callback=get_callbacks()
            )
    
# Specify a path
# Save
torch.save(model.policy.state_dict(), os.path.join(log_path, 'best_model.pth'))
run.finish()
# default_export_dir = f"log/{time}"
# export_dir = arg_or_default("--model-dir", default=default_export_dir)
# print("export_dir",export_dir)


