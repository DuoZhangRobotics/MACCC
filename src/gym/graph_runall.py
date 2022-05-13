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
import numpy as np
import sys

if (not (len(sys.argv) == 2)) or (sys.argv[1] == "-h") or (sys.argv[1] == "--help"):
    print("usage: python3 graph_run.py <pcc_env_log_filename.json>")
    exit(0)

filename = sys.argv[1]

data = {}
with open(filename) as f:
    data = json.load(f)
# print(data["Events"][1:])
time_data = [float(event["Time"]) for event in data["Events"][1:]]
rew_data = [float(event["Reward"]) for event in data["Events"][1:]]
# send_data = [float(event["Send Rate"]) for event in data["Events"][1:]]
thpt_data = [float(event["SumThroughput"]) if float(event["SumThroughput"])>=0 else 0 for event in data["Events"][1:]]
latency_data = [float(event["SumLatency"]) for event in data["Events"][1:]]
loss_data = [float(event["SumLoss"]) for event in data["Events"][1:]]
fair_data = [float(event["Fairness"]) for event in data["Events"][1:]]

thpt_all = []
loss_all = []
lat_all = []
for i in range(len(data["Events"][1]["Other"])): 
    tmp_thpt = [float(event["Other"][i]["Throughput"]) if float(event["Other"][i]["Throughput"]) >= 0 else 0 for event in data["Events"][1:]]
    thpt_all.append(tmp_thpt)
# second_thpt = [float(event["Other"][1]["Throughput"]) for event in data["Events"][1:]]
    tmp_lat = [float(event["Other"][i]["Latency"]) for event in data["Events"][1:]]
    lat_all.append(tmp_lat)
# second_lat = [float(event["Other"][1]["Latency"]) for event in data["Events"][1:]]
    tmp_loss = [float(event["Other"][i]["Loss Rate"]) for event in data["Events"][1:]]
    loss_all.append(tmp_loss)
# second_loss = [float(event["Other"][1]["Loss Rate"]) for event in data["Events"][1:]]
# print(len(other_data))
fig, axes = plt.subplots(5, figsize=(10, 10))
rew_axis = axes[0]
# send_axis = axes[1]
thpt_axis = axes[1]
latency_axis = axes[2]
loss_axis = axes[3]
fair_axis = axes[4]

rew_axis.plot(time_data, rew_data)
rew_axis.set_ylabel("Reward")

# send_axis.plot(time_data, send_data)
# send_axis.set_ylabel("Send Rate")

thpt_axis.plot(time_data, thpt_data, label="All")
for i in range(len(data["Events"][1]["Other"])): 
    thpt_axis.plot(time_data, thpt_all[i], label=f"Sender{i+1}")

# thpt_axis.plot(time_data, first_thpt, label="Sender1")
# thpt_axis.plot(time_data, second_thpt, label="Sender2")
thpt_axis.legend(bbox_to_anchor=(0.85, 1), loc='upper left', borderaxespad=0.)
thpt_axis.set_ylabel("Throughput")

latency_axis.plot(time_data, latency_data, linestyle='None',marker='None')
for i in range(len(data["Events"][1]["Other"])): 
    latency_axis.plot(time_data, lat_all[i], label=f"Sender{i+1}")
# latency_axis.plot(time_data, first_lat, label="Sender1")
# latency_axis.plot(time_data, second_lat, label="Sender2")
latency_axis.legend(bbox_to_anchor=(0.85, 1), loc='upper left', borderaxespad=0.)
latency_axis.set_ylabel("Latency")


loss_axis.plot(time_data, loss_data,linestyle='None',marker='None')
for i in range(len(data["Events"][1]["Other"])): 
    loss_axis.plot(time_data, loss_all[i], label=f"Sender{i+1}")
# loss_axis.plot(time_data, first_loss, label="Sender1")
# loss_axis.plot(time_data, second_loss, label="Sender2")
loss_axis.legend(bbox_to_anchor=(0.85, 1), loc='upper left', borderaxespad=0.)
loss_axis.set_ylabel("Loss Rate")


fair_axis.plot(time_data, fair_data)
fair_axis.set_ylabel("Fairness")
fair_axis.set_xlabel("Monitor Interval")

title = "Summary Graph with Cwnd and Upper bound for Latency and Loss Rate and Fairness Control"
fig.suptitle(title)
fig.savefig(f"{title}.png", dpi=500)
