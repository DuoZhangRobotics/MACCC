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

import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sys
import os
# if (not (len(sys.argv) == 2)) or (sys.argv[1] == "-h") or (sys.argv[1] == "--help"):
#     print("usage: python3 graph_run.py <pcc_env_log_filename.json>")
#     exit(0)

# filename = sys.argv[1]
mpl.style.use('seaborn')
fig = plt.figure()

path = './test/pcc_vs_macc/Use_PCC_False_num_senders_1_num_links_1_throughput_coefficient_10.0_loss_coefficient_1000.0_latency_coefficient_2000.0_fairness_coefficient_0.0'
our_data_lat = []
our_data_thr = []
num = 10
for i in range(num):
    data = {}
    filename = os.path.join(path, f'jsons/pcc_env_log_run{i}.json')
    with open(filename) as f:
        data = json.load(f)
    time_data = [float(event["Time"]) for event in data["Events"][1:]]
    rew_data = [float(event["Reward"]) for event in data["Events"][1:]]
    # send_data = [float(event["Send Rate"]) for event in data["Events"][1:]]
    thpt_data = [float(event["SumThroughput"]) for event in data["Events"][1:]]
    latency_data = [float(event["SumLatency"]) for event in data["Events"][1:]]
    loss_data = [float(event["SumLoss"]) for event in data["Events"][1:]]
    fair_data = [float(event["Fairness"]) for event in data["Events"][1:]]
    our_data_lat.append(np.mean(latency_data))
    # our_data_thr.append(np.mean(thpt_data))
    our_data_thr.append(np.mean(loss_data))
    

path = './test/pcc_vs_macc/Use_PCC_True_num_senders_1_num_links_1_throughput_coefficient_10.0_loss_coefficient_1000.0_latency_coefficient_2000.0_fairness_coefficient_0.0'
pcc_data_lat = []
pcc_data_thr = []

for i in range(num):
    data = {}
    filename = os.path.join(path, f'jsons/pcc_env_log_run{i}.json')
    with open(filename) as f:
        data = json.load(f)
    time_data = [float(event["Time"]) for event in data["Events"][1:]]
    rew_data = [float(event["Reward"]) for event in data["Events"][1:]]
    # send_data = [float(event["Send Rate"]) for event in data["Events"][1:]]
    thpt_data = [float(event["SumThroughput"]) for event in data["Events"][1:]]
    latency_data = [float(event["SumLatency"]) for event in data["Events"][1:]]
    loss_data = [float(event["SumLoss"]) for event in data["Events"][1:]]
    fair_data = [float(event["Fairness"]) for event in data["Events"][1:]]
    pcc_data_lat.append(np.mean(latency_data))
    # pcc_data_thr.append(np.mean(thpt_data))
    pcc_data_thr.append(np.mean(loss_data))
    

plt.plot(our_data_lat, our_data_thr, color="tab:purple",marker='o',linestyle='None', label="MACCC")
plt.plot(pcc_data_lat, pcc_data_thr, color="tab:brown",marker='^', linestyle='None',label='PCC')
plt.plot(np.mean(our_data_lat), np.mean(our_data_thr), color="tab:orange",marker='o',linestyle='None', markersize=10 ,label="MACCC Avg")
plt.plot(np.mean(pcc_data_lat), np.mean(pcc_data_thr), color="tab:red",marker='^',linestyle='None', markersize=10, label="PCC Avg")

plt.xlabel("Latency")
plt.ylabel("Loss Rate")
title = "Loss Rate vs Latency"
plt.title(title)
plt.legend()
fig.savefig(f"Loss_Latency.png", dpi=500)
