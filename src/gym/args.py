import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Setup the env.')
    parser.add_argument("--steps", type=int, default=1600 * 410 * 50)
    parser.add_argument('--num_senders', type = int, default=1)
    parser.add_argument('--num_links', type = int, default=1)
    parser.add_argument('--throughput_coefficient', type = float, default=20)
    parser.add_argument('--loss_coefficient', type = float, default=1e3)
    parser.add_argument('--latency_coefficient', type = float, default=2e3)
    parser.add_argument('--fairness_coefficient', type = float, default=1e3)
    parser.add_argument('--PCC', type = int, default=0)
    args = parser.parse_args()
    return args