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
thpt_data = [float(event["SumThroughput"]) for event in data["Events"][1:]]
latency_data = [float(event["SumLatency"]) for event in data["Events"][1:]]
loss_data = [float(event["SumLoss"]) for event in data["Events"][1:]]
fair_data = [float(event["Fairness"]) for event in data["Events"][1:]]

fig, axes = plt.subplots(5, figsize=(10, 12))
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

thpt_axis.plot(time_data, thpt_data)
thpt_axis.set_ylabel("Throughput")

latency_axis.plot(time_data, latency_data)
latency_axis.set_ylabel("Latency")

loss_axis.plot(time_data, loss_data)
loss_axis.set_ylabel("Loss Rate")


fair_axis.plot(time_data, fair_data)
fair_axis.set_ylabel("Fairness")
fair_axis.set_xlabel("Monitor Interval")

title = "Summary Graph with Cwnd and Upper bound for Latency and Loss Rate and Fairness Control"
fig.suptitle(title)
fig.savefig(f"{title}.png", dpi=500)
