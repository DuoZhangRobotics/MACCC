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
import args
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

arguments = args.get_args()
time = f'{datetime.now().strftime("%Y-%m-%d-%H_%M_%S")}'
path_signiture = f'Use_PCC_{True if arguments.PCC==1 else False}_num_senders_{arguments.num_senders}_num_links_{arguments.num_links}_throughput_coefficient_{arguments.throughput_coefficient}_loss_coefficient_{arguments.loss_coefficient}_latency_coefficient_{arguments.latency_coefficient}_fairness_coefficient_{arguments.fairness_coefficient}_timesteps_{arguments.steps}_' + time
print(path_signiture)
log_path = f"./log/{path_signiture}/"
timesteps = arguments.steps
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
        name=path_signiture
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
    if not os.path.exists(os.path.join(log_path, "jsons")):
        os.mkdir(os.path.join(log_path, "jsons")) 
        
model_path = os.path.join(log_path, "models")
json_path = os.path.join(log_path, "jsons")
run_path = os.path.join(log_path, "runs")
def get_callbacks():
    eval_callback = EvalCallback(env, best_model_save_path=model_path,
                             log_path=model_path, eval_freq=10000,
                             deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=model_path)
    wandb_callback = WandbCallback(
                        # gradient_save_freq=1e4,
                        model_save_path=run_path,
                        model_save_freq=1e3,
                        verbose=2)
    callback = CallbackList([checkpoint_callback, 
                             eval_callback, 
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

env = gym.make('PccNs-v0', 
               num_senders=arguments.num_senders, 
               num_links = arguments.num_links, 
               log_path = json_path,
               reward_coefficients = [arguments.throughput_coefficient,
                                      arguments.loss_coefficient,
                                      arguments.latency_coefficient,
                                      arguments.fairness_coefficient],
               PCC_reward = arguments.PCC
               )
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
            tensorboard_log=run_path
            )

# for i in range(0, 6):
model.learn(total_timesteps=timesteps
            , callback=get_callbacks()
            )
    
# Specify a path
# Save
torch.save(model.policy.state_dict(), os.path.join(model_path, 'best_model.pth'))
run.finish()



