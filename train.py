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

def plot_subpolicy_probabilities(probability_history,
                                 labels,
                                 output_path: str):
    """
    Plot sub-policy mixture probabilities across episodes.
    """
    if len(probability_history) == 0:
        print('No sub-policy probability history to plot.')
        return

    probs = np.asarray(probability_history)
    episodes = np.arange(1, probs.shape[0] + 1)

    plt.figure(figsize=(11, 6))
    for idx, label in enumerate(labels):
        plt.plot(episodes, probs[:, idx], label=label)

    plt.xlabel('Episode')
    plt.ylabel('Mixture Probability')
    plt.title('Sub-policy Usage Probabilities Across Episodes')
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right', ncol=2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f'Saved sub-policy probability plot to {output_path}')

def monte_carlo_action_probs(policy: PolicyNetwork, state: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Monte Carlo dropout average over policy forward passes.
    """
    samples = [policy(state) for _ in range(num_samples)]
    return torch.stack(samples).mean(0)

def domain_knowledge_action_probs(state: torch.Tensor) -> torch.Tensor:
    """
    CartPole domain-knowledge controller expressed as a deterministic policy.

    Uses a linear stabilizing score over [x, x_dot, theta, theta_dot] where
    angle terms dominate. Returns one-hot probabilities over [left, right].
    """
    # State order for CartPole-v1: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
    cart_pos, cart_vel, pole_angle, pole_ang_vel = state
    score = 0.2 * cart_pos + 0.2 * cart_vel + 0.4 * pole_angle + 0.5 * pole_ang_vel
    probs = torch.zeros(2, dtype=state.dtype, device=state.device)
    # Gym CartPole action mapping: 0=left, 1=right
    probs[1 if score > 0 else 0] = 1.0
    return probs

def evaluate_domain_knowledge_policy(env_name: str = 'CartPole-v1',
                                     seed: int = 1234,
                                     episodes: int = 500):
    """
    Evaluate the fixed domain-knowledge policy without learned components.

    Returns:
        reward_sums: list[float], per-episode total rewards.
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        env = gym.make(env_name, new_step_api=True)
    except TypeError:
        env = gym.make(env_name)

    reward_sums = []
    obs = reset_env(env, seed)

    for episode in range(1, episodes + 1):
        if episode > 1:
            obs = reset_env(env)

        state = torch.tensor(obs, dtype=torch.float32).to(device)
        done = False
        rewards = []

        while not done:
            action_probs = domain_knowledge_action_probs(state)
            action = torch.argmax(action_probs).item()

            out = env.step(action)
            if len(out) == 5:
                obs, reward, terminated, truncated, _ = out
                done = terminated or truncated
            else:
                obs, reward, done, _ = out

            rewards.append(reward)
            state = torch.tensor(obs, dtype=torch.float32).to(device)

        reward_sum = sum(rewards)
        reward_sums.append(reward_sum)

        if episode % 10 == 0:
            print(f'[DK only] Episode {episode}, Total Reward: {reward_sum}')

    env.close()
    return reward_sums

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
                       diversity_threshold: float = 1e-3,
                       track_subpolicy_probs: bool = False):
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
          use_domain_knowledge: bool = False,
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
    policies plus the current policy (and optionally a fixed domain-knowledge
    policy), with learnable reuse probabilities.

    Returns:
        reward_sums: list[float], per-episode total rewards.
        subpolicy_prob_history: np.ndarray with shape
            (episodes, max_policy_bank_size + 1 + int(use_domain_knowledge))
            if policy_reuse=True, else None.
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
        # Slots [0:max_policy_bank_size) are for old policies.
        # Next slot is current policy. Optional final slot is DK policy.
        extra_slots = 1 + int(use_domain_knowledge)
        reuse_logits = torch.nn.Parameter(torch.zeros(max_policy_bank_size + extra_slots, device=device))
        reuse_optimizer = Adam([reuse_logits], lr=lr)

    reward_sums = []
    subpolicy_prob_history = []
    total_reuse_slots = max_policy_bank_size + 1 + int(use_domain_knowledge)

    for episode in range(1, episodes + 1):
        # Reset env at start of each episode (first episode already reset above)
        if episode > 1:
            obs = reset_env(env)
        
        # Reset recurrent state for all active policies
        policy.esn.reset_state()
        for _, old_policy in policy_bank:
            old_policy.esn.reset_state()

        if policy_reuse:
            active_count = len(policy_bank) + 1 + int(use_domain_knowledge)
            active_probs = torch.softmax(reuse_logits[:active_count], dim=0)
            padded_probs = torch.zeros(total_reuse_slots, device=device)
            padded_probs[:active_count] = active_probs
            subpolicy_prob_history.append(padded_probs.detach().cpu().numpy())

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

                if use_domain_knowledge:
                    candidate_probs.append(domain_knowledge_action_probs(state))

                candidate_probs = torch.stack(candidate_probs)

                # Use only active logits: one per bank policy plus current policy,
                # and optionally one fixed DK policy.
                active_count = len(policy_bank) + 1 + int(use_domain_knowledge)
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
    if policy_reuse:
        return reward_sums, np.asarray(subpolicy_prob_history)
    return reward_sums, None


if __name__ == '__main__':
    max_policy_bank_size = 9

    data = np.load('results/reward_sums.npz')

    # reward_sums_reuse, _ = train(
    #     policy_reuse=True,
    #     use_domain_knowledge=False,
    #     max_policy_bank_size=max_policy_bank_size,
    # )
    reward_sums_reuse = data['reuse']

    # reward_sums_no_reuse, _ = train(
    #     policy_reuse=False,
    #     use_domain_knowledge=False,
    #     max_policy_bank_size=max_policy_bank_size,
    # )
    reward_sums_no_reuse = data['no_reuse']

    # reward_sums_dk_only = evaluate_domain_knowledge_policy()
    reward_sums_dk_only = data['dk_only']

    reward_sums_reuse_dk, subpolicy_prob_history = train(
        policy_reuse=True,
        use_domain_knowledge=True,
        max_policy_bank_size=max_policy_bank_size,
    )

    window = 50
    smoothed_no_reuse = moving_average(reward_sums_no_reuse, window=window)
    smoothed_reuse = moving_average(reward_sums_reuse, window=window)
    smoothed_reuse_dk = moving_average(reward_sums_reuse_dk, window=window)
    smoothed_dk_only = moving_average(reward_sums_dk_only, window=window)

    np.savez('results/reward_sums.npz',
             no_reuse=reward_sums_no_reuse,
             reuse=reward_sums_reuse,
             reuse_dk=reward_sums_reuse_dk,
             dk_only=reward_sums_dk_only,
             smoothed_no_reuse=smoothed_no_reuse,
             smoothed_reuse=smoothed_reuse,
             smoothed_reuse_dk=smoothed_reuse_dk,
             smoothed_dk_only=smoothed_dk_only,
             subpolicy_prob_history=subpolicy_prob_history
             )
    print(f'Saved raw and smoothed reward sums to results/reward_sums.npz')

    plt.figure(figsize=(10, 6))
    # plt.plot(reward_sums_no_reuse, alpha=0.25, label='No Reuse (raw)', color='tab:blue')
    plt.plot(smoothed_no_reuse, label=f'No Reuse (No domain knowledge)', color='tab:blue')
    # plt.plot(reward_sums_reuse, alpha=0.25, label='Reuse (raw)', color='tab:orange')
    plt.plot(smoothed_reuse, label=f'Reuse', color='tab:orange')
    plt.plot(smoothed_reuse_dk, label='Reuse + DK policy', color='tab:green')
    plt.plot(smoothed_dk_only, label='DK policy only', color='tab:red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('CartPole Reward Curves: No Reuse vs Reuse vs Reuse+DK vs DK only')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/reward_comparison.png')
    plt.close()
    print(f'Saved reward plot to reward_comparison.png')

    subpolicy_labels = [f'Old policy {idx + 1}' for idx in range(max_policy_bank_size)]
    subpolicy_labels += ['New policy', 'Domain-knowledge policy']
    plot_subpolicy_probabilities(subpolicy_prob_history,
                                 subpolicy_labels,
                                 'results/subpolicy_probabilities.png')
    