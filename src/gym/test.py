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
# time = f'{datetime.now().strftime("%Y-%m-%d-%H_%M_%S")}'
path_signiture = f'Use_PCC_{True if arguments.PCC==1 else False}_num_senders_{arguments.num_senders}_num_links_{arguments.num_links}_throughput_coefficient_{arguments.throughput_coefficient}_loss_coefficient_{arguments.loss_coefficient}_latency_coefficient_{arguments.latency_coefficient}_fairness_coefficient_{arguments.fairness_coefficient}' #+ time
log_path = f"./test/pcc_vs_macc/{path_signiture}/"

def check_paths():
    if not os.path.exists('./test/pcc_vs_macc'):
        os.mkdir('./test/pcc_vs_macc')
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
check_paths()     
model_path = os.path.join(log_path, "models")
json_path = os.path.join(log_path, "jsons")
run_path = os.path.join(log_path, "runs")
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

if arguments.PCC == 1:
    model = PPO.load('./log/Use_PCC_True_num_senders_1_num_links_1_throughput_coefficient_10.0_loss_coefficient_1000.0_latency_coefficient_2000.0_fairness_coefficient_0.0_timesteps_3936000_2022-05-10-19_27_54/models/best_model.zip', env=env)
else:
    model = PPO.load('./log/Use_PCC_False_num_senders_1_num_links_1_throughput_coefficient_10.0_loss_coefficient_1000.0_latency_coefficient_2000.0_fairness_coefficient_0.0_timesteps_32800000_2022-05-10-18_49_48/models/rl_model_24000000_steps.zip', env=env)

obs = env.reset()

for j in range(10):
    for i in range(400):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
    env.dump_events_to_file(os.path.join(json_path, f"pcc_env_log_run{j}.json"))
    env.reset()

