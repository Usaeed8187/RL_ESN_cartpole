"""
Training script for ESN-based policy gradient on CartPole-v1.
"""

# Monkey-patch for Gym compatibility
import numpy as np
np.bool8 = np.bool_

import gym
import copy
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from policy import PolicyNetwork
from esn import EchoStateNetwork
from utils import set_seed, reset_env

def moving_average(values, window: int = 20):
    """
    Compute moving average for visualization.
    """
    if window <= 1:
        return values
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        out.append(sum(values[start:idx + 1]) / (idx - start + 1))
    return out


def monte_carlo_action_probs(policy: PolicyNetwork, state: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Monte Carlo dropout average over policy forward passes.
    """
    samples = [policy(state) for _ in range(num_samples)]
    return torch.stack(samples).mean(0)


def update_policy_bank(bank, policy: PolicyNetwork, device: torch.device, max_bank_size: int):
    """
    Store a frozen copy of the current policy into the reusable policy bank.

    Bank selection strategy (bounded bank): keep the most recent `max_bank_size`
    policy snapshots (FIFO). When the bank is full, we drop the oldest snapshot
    and append the newest one.
    """
    policy_copy = copy.deepcopy(policy).to(device)
    policy_copy.train()  # keep dropout active for MC averaging
    for param in policy_copy.parameters():
        param.requires_grad = False
    bank.append(policy_copy)
    if len(bank) > max_bank_size:
        bank.pop(0)


def train(policy_reuse: bool = False,
          env_name: str = 'CartPole-v1',
          seed: int = 1234,
          reservoir_size: int = 500,
          lr: float = 1e-2,
          gamma: float = 0.99,
          num_samples: int = 50,
          episodes: int = 500,
          max_policy_bank_size: int = 10):
    """
    Train the policy network using REINFORCE.

    If policy_reuse=True, train a mixture policy over a bounded bank of previous
    policies plus the current policy, with learnable reuse probabilities.

    Returns:
        reward_sums: list[float], per-episode total rewards.
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the environment (handles old/new Gym APIs)
    try:
        env = gym.make(env_name, new_step_api=True)
    except TypeError:
        env = gym.make(env_name)

    # Initial reset
    obs = reset_env(env, seed)
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Build models
    esn = EchoStateNetwork(input_dim, reservoir_size).to(device)
    policy = PolicyNetwork(esn, action_dim).to(device)
    policy_optimizer = Adam(policy.parameters(), lr=lr)

    # Policy reuse containers
    policy_bank = []
    reuse_logits = None
    reuse_optimizer = None
    if policy_reuse:
        # Persistent logits/optimizer: we keep one set for the whole run.
        # Slots [0:max_policy_bank_size) are for old policies, slot [-1] is current policy.
        reuse_logits = torch.nn.Parameter(torch.zeros(max_policy_bank_size + 1, device=device))
        reuse_optimizer = Adam([reuse_logits], lr=lr)

    reward_sums = []

    for episode in range(1, episodes + 1):
        # Reset env at start of each episode (first episode already reset above)
        if episode > 1:
            obs = reset_env(env)
        
        # Reset recurrent state for all active policies
        policy.esn.reset_state()
        for old_policy in policy_bank:
            old_policy.esn.reset_state()

        # Update policy bank with the policy from previous training stage
        if policy_reuse and episode > 1:
            pre_len = len(policy_bank)
            update_policy_bank(policy_bank, policy, device, max_policy_bank_size)

            # If FIFO eviction happened, shift corresponding logits left so each
            # bank slot keeps matching the same policy index in the bounded bank.
            if pre_len == max_policy_bank_size:
                with torch.no_grad():
                    reuse_logits.data[:-2] = reuse_logits.data[1:-1].clone()
                    reuse_logits.data[-2] = 0.0

        state = torch.tensor(obs, dtype=torch.float32).to(device)
        rewards = []
        log_probs = []
        done = False

        while not done:
            if policy_reuse and len(policy_bank) > 0:
                candidate_probs = []

                # Frozen old policies: inference-only to avoid autograd overhead.
                with torch.no_grad():
                    for old_policy in policy_bank:
                        old_probs = monte_carlo_action_probs(old_policy, state, num_samples)
                        candidate_probs.append(old_probs)

                # Current policy remains differentiable.
                current_probs = monte_carlo_action_probs(policy, state, num_samples)
                candidate_probs.append(current_probs)
                candidate_probs = torch.stack(candidate_probs)

                # Use only active logits: one per bank policy plus one for current policy.
                active_count = len(policy_bank) + 1
                active_logits = reuse_logits[:active_count]
                mix_weights = torch.softmax(active_logits, dim=0)
                action_probs = torch.sum(mix_weights.unsqueeze(1) * candidate_probs, dim=0)
            else:
                action_probs = monte_carlo_action_probs(policy, state, num_samples)

            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            # Step environment
            out = env.step(action.item())
            if len(out) == 5:
                obs, reward, terminated, truncated, _ = out
                done = terminated or truncated
            else:
                obs, reward, done, _ = out

            state = torch.tensor(obs, dtype=torch.float32).to(device)
            rewards.append(reward)

        # Compute discounted returns
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        # use unbiased=False to avoid ddof warning when len==1
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        # Policy gradient update
        loss = -torch.stack([lp * ret for lp, ret in zip(log_probs, returns)]).sum()

        policy_optimizer.zero_grad()
        if reuse_optimizer is not None:
            reuse_optimizer.zero_grad()

        loss.backward()
        policy_optimizer.step()
        if reuse_optimizer is not None:
            reuse_optimizer.step()

        reward_sum = sum(rewards)
        reward_sums.append(reward_sum)

        if episode % 10 == 0:
            print(f'Episode {episode}, Total Reward: {reward_sum}')

    env.close()
    return reward_sums


if __name__ == '__main__':
    max_policy_bank_size = 10

    policy_reuse = True
    reward_sums_reuse = train(policy_reuse=policy_reuse, max_policy_bank_size=max_policy_bank_size)

    policy_reuse = False
    reward_sums_no_reuse = train(policy_reuse=policy_reuse, max_policy_bank_size=max_policy_bank_size)

    window = 20
    smoothed_no_reuse = moving_average(reward_sums_no_reuse, window=window)
    smoothed_reuse = moving_average(reward_sums_reuse, window=window)

    plt.figure(figsize=(10, 6))
    # plt.plot(reward_sums_no_reuse, alpha=0.25, label='No Reuse (raw)', color='tab:blue')
    plt.plot(smoothed_no_reuse, label=f'No Reuse (MA-{window})', color='tab:blue')
    # plt.plot(reward_sums_reuse, alpha=0.25, label='Reuse (raw)', color='tab:orange')
    plt.plot(smoothed_reuse, label=f'Reuse (MA-{window})', color='tab:orange')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('CartPole Reward Curves: Policy Reuse vs No Reuse')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('reward_comparison.png')
    plt.close()
    print(f'Saved reward plot to reward_comparison.png')