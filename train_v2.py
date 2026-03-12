"""
Training script v2 for ESN-based policy gradient on CartPole-v1.

Differences from train.py:
- Reuse setup includes only two policies: current (new) and DK policy.
- No policy bank / warm start.
- Reuse probabilities are initialized as: new=0.1, DK=0.9.
"""

# Monkey-patch for Gym compatibility
import numpy as np
np.bool8 = np.bool_

import gym
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam

from policy import PolicyNetwork
from esn import EchoStateNetwork
from utils import set_seed, reset_env


def moving_average(values, window: int = 20):
    """Compute moving average for visualization."""
    if window <= 1:
        return values
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        out.append(sum(values[start:idx + 1]) / (idx - start + 1))
    return out


def plot_subpolicy_probabilities(probability_history, labels, output_path: str):
    """Plot sub-policy mixture probabilities across episodes."""
    if len(probability_history) == 0:
        print('No sub-policy probability history to plot.')
        return

    probs = np.asarray(probability_history)
    episodes = np.arange(1, probs.shape[0] + 1)

    plt.figure(figsize=(10, 5))
    for idx, label in enumerate(labels):
        plt.plot(episodes, probs[:, idx], label=label)

    plt.xlabel('Episode')
    plt.ylabel('Mixture Probability')
    plt.title('Sub-policy Usage Probabilities (New + DK)')
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f'Saved sub-policy probability plot to {output_path}')


def monte_carlo_action_probs(policy: PolicyNetwork, state: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Monte Carlo dropout average over policy forward passes."""
    samples = [policy(state) for _ in range(num_samples)]
    return torch.stack(samples).mean(0)


def domain_knowledge_action_probs(state: torch.Tensor) -> torch.Tensor:
    """CartPole domain-knowledge controller as deterministic one-hot policy."""
    cart_pos, cart_vel, pole_angle, pole_ang_vel = state
    score = 0.2 * cart_pos + 0.2 * cart_vel + 0.4 * pole_angle + 0.5 * pole_ang_vel
    probs = torch.zeros(2, dtype=state.dtype, device=state.device)
    probs[1 if score > 0 else 0] = 1.0
    return probs


def evaluate_domain_knowledge_policy(env_name: str = 'CartPole-v1',
                                     seed: int = 1234,
                                     episodes: int = 500):
    """Evaluate the fixed domain-knowledge policy without learned components."""
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
            print(f'[DK only] Episode {episode}, Total Reward: {reward_sum}, Reuse probs: [1.0]')

    env.close()
    return reward_sums


def train(policy_reuse: bool = False,
          use_domain_knowledge: bool = False,
          env_name: str = 'CartPole-v1',
          seed: int = 1234,
          reservoir_size: int = 500,
          lr: float = 1e-2,
          gamma: float = 0.99,
          num_samples: int = 50,
          episodes: int = 500,
          dk_initial_prob: float = 0.9):
    """
    Train policy network using REINFORCE.

    If policy_reuse=True and use_domain_knowledge=True, use a 2-policy mixture:
    [new, DK]. Mixture logits are learnable and initialized to new=0.1, DK=0.9.
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        env = gym.make(env_name, new_step_api=True)
    except TypeError:
        env = gym.make(env_name)

    obs = reset_env(env, seed)
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    esn = EchoStateNetwork(input_dim, reservoir_size).to(device)
    policy = PolicyNetwork(esn, action_dim).to(device)
    policy_optimizer = Adam(policy.parameters(), lr=lr)

    reuse_logits = None
    reuse_optimizer = None
    subpolicy_prob_history = []

    if policy_reuse:
        if not use_domain_knowledge:
            raise ValueError('train_v2.py supports policy_reuse only with DK enabled.')

        dk_initial_prob = float(np.clip(dk_initial_prob, 1e-3, 1.0 - 1e-3))
        current_initial_prob = 1.0 - dk_initial_prob

        # Fixed slot layout: [new, DK]
        reuse_logits = torch.nn.Parameter(torch.zeros(2, device=device))
        with torch.no_grad():
            reuse_logits.data[0] = np.log(current_initial_prob)
            reuse_logits.data[1] = np.log(dk_initial_prob)

        reuse_optimizer = Adam([reuse_logits], lr=lr)

    reward_sums = []

    for episode in range(1, episodes + 1):
        if episode > 1:
            obs = reset_env(env)

        policy.esn.reset_state()

        if policy_reuse:
            active_probs = torch.softmax(reuse_logits, dim=0)
            subpolicy_prob_history.append(active_probs.detach().cpu().numpy())

        state = torch.tensor(obs, dtype=torch.float32).to(device)
        rewards = []
        log_probs = []
        done = False

        while not done:
            if policy_reuse:
                new_probs = monte_carlo_action_probs(policy, state, num_samples)
                dk_probs = domain_knowledge_action_probs(state)
                candidate_probs = torch.stack([new_probs, dk_probs])
                mix_weights = torch.softmax(reuse_logits, dim=0)
                action_probs = torch.sum(mix_weights.unsqueeze(1) * candidate_probs, dim=0)
            else:
                action_probs = monte_carlo_action_probs(policy, state, num_samples)

            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            out = env.step(action.item())
            if len(out) == 5:
                obs, reward, terminated, truncated, _ = out
                done = terminated or truncated
            else:
                obs, reward, done, _ = out

            state = torch.tensor(obs, dtype=torch.float32).to(device)
            rewards.append(reward)

        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

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

        if episode < 10 or episode % 10 == 0:
            if policy_reuse and len(subpolicy_prob_history) > 0:
                p_new, p_dk = subpolicy_prob_history[-1]
                print(f'Episode {episode}, Total Reward: {reward_sum}')
                print(f'DK prob: {p_dk:.3f}')
                print(f'RL reuse probs: [new:{p_new:.3f}] \n')
            else:
                print(f'Episode {episode}, Total Reward: {reward_sum} \n')

    env.close()
    if policy_reuse:
        return reward_sums, np.asarray(subpolicy_prob_history)
    return reward_sums, None


if __name__ == '__main__':
    reward_sums_no_reuse, _ = train(
        policy_reuse=False,
        use_domain_knowledge=False,
    )

    reward_sums_dk_only = evaluate_domain_knowledge_policy()

    reward_sums_reuse_dk, subpolicy_prob_history = train(
        policy_reuse=True,
        use_domain_knowledge=True,
        dk_initial_prob=0.9,
    )

    window = 20
    smoothed_no_reuse = moving_average(reward_sums_no_reuse, window=window)
    smoothed_reuse_dk = moving_average(reward_sums_reuse_dk, window=window)
    smoothed_dk_only = moving_average(reward_sums_dk_only, window=window)

    np.savez(
        'results/reward_sums_v2.npz',
        no_reuse=reward_sums_no_reuse,
        reuse_dk=reward_sums_reuse_dk,
        dk_only=reward_sums_dk_only,
        smoothed_no_reuse=smoothed_no_reuse,
        smoothed_reuse_dk=smoothed_reuse_dk,
        smoothed_dk_only=smoothed_dk_only,
        subpolicy_prob_history=subpolicy_prob_history
    )
    print('Saved raw and smoothed reward sums to results/reward_sums_v2.npz')

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_no_reuse, label='No Reuse', color='tab:blue')
    plt.plot(smoothed_reuse_dk, label='Reuse + DK policy', color='tab:green')
    plt.plot(smoothed_dk_only, label='DK policy only', color='tab:red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('CartPole Reward Curves (v2): No Reuse vs Reuse+DK vs DK only')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/reward_comparison_v2.png')
    plt.close()
    print('Saved reward plot to results/reward_comparison_v2.png')

    plot_subpolicy_probabilities(
        subpolicy_prob_history,
        labels=['New policy', 'Domain-knowledge policy'],
        output_path='results/subpolicy_probabilities_v2.png'
    )