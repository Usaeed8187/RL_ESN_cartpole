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


def _policy_distance(policy_a: PolicyNetwork, policy_b: PolicyNetwork) -> float:
    """
    Mean L2 distance across flattened parameters for two policy snapshots.
    """
    sq_sum = 0.0
    param_count = 0
    for param_a, param_b in zip(policy_a.parameters(), policy_b.parameters()):
        diff = (param_a.detach() - param_b.detach()).reshape(-1)
        sq_sum += torch.dot(diff, diff).item()
        param_count += diff.numel()
    return (sq_sum / max(param_count, 1)) ** 0.5


def update_policy_bank(bank,
                       policy: PolicyNetwork,
                       device: torch.device,
                       max_bank_size: int,
                       score: float,
                       diversity_threshold: float = 1e-3):
    """
    Store a frozen copy of the current policy into the reusable policy bank.

    Bank structure: list of (score, policy_snapshot).
    Selection strategy: keep top-k snapshots by score (descending).
    Optional diversity rule: avoid near-duplicates by checking parameter distance.

    Returns:
        prev_indices: For each new bank slot, index of the same policy in the
            previous bank ordering (or None if newly inserted/replaced).
    """
    old_policies = [entry[1] for entry in bank]

    policy_copy = copy.deepcopy(policy).to(device)
    policy_copy.train()  # keep dropout active for MC averaging
    for param in policy_copy.parameters():
        param.requires_grad = False
    replaced_duplicate = False
    if diversity_threshold is not None and diversity_threshold > 0 and len(bank) > 0:
        best_idx = None
        best_dist = float('inf')
        for idx, (_, old_policy) in enumerate(bank):
            dist = _policy_distance(policy_copy, old_policy)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_dist < diversity_threshold and best_idx is not None:
            old_score, _ = bank[best_idx]
            if score > old_score:
                bank[best_idx] = (score, policy_copy)
            replaced_duplicate = True

    if not replaced_duplicate:
        bank.append((score, policy_copy))

    bank.sort(key=lambda item: item[0], reverse=True)

    if len(bank) > max_bank_size:
        del bank[max_bank_size:]

    prev_indices = []
    for _, new_policy in bank:
        old_idx = None
        for idx, old_policy in enumerate(old_policies):
            if new_policy is old_policy:
                old_idx = idx
                break
        prev_indices.append(old_idx)

    return prev_indices


def train(policy_reuse: bool = False,
          env_name: str = 'CartPole-v1',
          seed: int = 1234,
          reservoir_size: int = 500,
          lr: float = 1e-2,
          gamma: float = 0.99,
          num_samples: int = 50,
          episodes: int = 500,
          max_policy_bank_size: int = 10,
          score_window: int = 20,
          diversity_threshold: float = 1e-3):
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
        for _, old_policy in policy_bank:
            old_policy.esn.reset_state()

        state = torch.tensor(obs, dtype=torch.float32).to(device)
        rewards = []
        log_probs = []
        done = False

        while not done:
            if policy_reuse and len(policy_bank) > 0:
                # Runtime safety check: the current live policy should never also
                # appear inside the old-policy bank used for this rollout.
                assert all(old_policy is not policy for _, old_policy in policy_bank)
                candidate_probs = []

                # Frozen old policies: inference-only to avoid autograd overhead.
                with torch.no_grad():
                    for _, old_policy in policy_bank:
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

        # Snapshot the rollout policy immediately before updating it, so policy_bank
        # always stores strictly older checkpoints than the live policy.
        if policy_reuse:
            pre_len = len(policy_bank)
            start = max(0, len(reward_sums) - score_window)
            score = float(sum(reward_sums[start:]) / max(1, len(reward_sums) - start))

            prev_indices = update_policy_bank(policy_bank,
                                              policy,
                                              device,
                                              max_policy_bank_size,
                                              score,
                                              diversity_threshold=diversity_threshold)

            with torch.no_grad():
                new_logits = torch.zeros_like(reuse_logits.data)
                for new_idx, old_idx in enumerate(prev_indices):
                    if old_idx is not None:
                        new_logits[new_idx] = reuse_logits.data[old_idx]

                # Keep the current-policy bias when moving to a new active slot.
                new_current_idx = len(policy_bank)
                old_current_idx = pre_len
                new_logits[new_current_idx] = reuse_logits.data[old_current_idx]
                reuse_logits.data.copy_(new_logits)

            if len(policy_bank) == 0:
                with torch.no_grad():
                    reuse_logits.data[0] = 0.0

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

    np.savez('results/reward_sums.npz',
             no_reuse=reward_sums_no_reuse,
             reuse=reward_sums_reuse,
             smoothed_no_reuse=smoothed_no_reuse,
             smoothed_reuse=smoothed_reuse)
    print(f'Saved raw and smoothed reward sums to results/reward_sums.npz')

    plt.figure(figsize=(10, 6))
    # plt.plot(reward_sums_no_reuse, alpha=0.25, label='No Reuse (raw)', color='tab:blue')
    plt.plot(smoothed_no_reuse, label=f'No Reuse (No domain knowledge)', color='tab:blue')
    # plt.plot(reward_sums_reuse, alpha=0.25, label='Reuse (raw)', color='tab:orange')
    plt.plot(smoothed_reuse, label=f'Reuse', color='tab:orange')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('CartPole Reward Curves: Policy Reuse vs No Reuse')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/reward_comparison.png')
    plt.close()
    print(f'Saved reward plot to reward_comparison.png')