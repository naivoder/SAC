import gymnasium as gym
import utils
from agent import SACAgent
import numpy as np
import os 
import warnings 
from argparse import ArgumentParser
import pandas as pd 

warnings.filterwarnings("ignore", category=DeprecationWarning)

environments = [
    "BipedalWalker-v3",
    "Pendulum-v1",
    "MountainCarContinuous-v0",
    "Ant-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "Humanoid-v4",
    "LunarLanderContinuous-v2",
    "HumanoidStandup-v4",
    "InvertedDoublePendulum-v4",
    "InvertedPendulum-v4",
    "Pusher-v4",
    "Reacher-v4",
    "Swimmer-v3",
    "Walker2d-v4",
]

def run_sac(env_name, n_games=10000):
    env = gym.make(env_name, render_mode="rgb_array")

    agent = SACAgent(
    env_name,
    env.observation_space.shape,
    env.action_space,
    tau=5e-3,
    reward_scale=10,
    batch_size=256,
    )

    best_score = env.reward_range[0]
    history = []
    metrics = []

    for i in range(n_games):
        state, _ = env.reset()

        term, trunc, score = False, False, 0
        while not term and not trunc:
            action = agent.choose_action(state)
            next_state, reward, term, trunc, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, term or trunc)
            agent.learn()

            score += reward
            state = next_state

        history.append(score)
        avg_score = np.mean(history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoints()

        metrics.append(
            {
                "episode": i + 1,
                "score": score,
                "average_score": avg_score,
                "best_score": best_score,
            }
        )

        print(
            f"[{env_name} Episode {i + 1:04}/{n_games}]    Score = {score:7.4f}    Average = {avg_score:7.4f}",
            end="\r",
        )

    return history, metrics, best_score, agent


def save_best_version(env_name, agent, seeds=100):
    agent.load_checkpoints()
    
    best_total_reward = float("-inf")
    best_frames = None

    for _ in range(seeds):
        env = gym.make(env_name, render_mode="rgb_array")

        frames = []
        total_reward = 0

        state, _ = env.reset()
        term, trunc = False, False
        while not term and not trunc:
            frames.append(env.render())
            action = agent.choose_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            state = next_state
            total_reward += reward

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_frames = frames

    utils.save_animation(best_frames, f"environments/{env_name}.gif")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--env", default=None, help="Environment name from Gymnasium"
    )
    parser.add_argument(
        "-n", "--n_games", default=10000, type=int, help="Number of episodes (games) to run during training"
    )
    args = parser.parse_args()

    for fname in ["metrics", "environments", "weights"]:
        if not os.path.exists(fname):
            os.makedirs(fname)

    if args.env:
        history, metrics, best_score, trained_agent = run_sac(args.env, args.n_games)
        utils.plot_running_avg(history, args.env)
        df = pd.DataFrame(metrics)
        df.to_csv(f"metrics/{args.env}_metrics.csv", index=False)
        save_best_version(args.env, trained_agent)
    else:
        for env_name in environments:
            history, metrics, best_score, trained_agent = run_sac(env_name, args.n_games)
            utils.plot_running_avg(history, env_name)
            df = pd.DataFrame(metrics)
            df.to_csv(f"metrics/{env_name}_metrics.csv", index=False)
            save_best_version(env_name, trained_agent)
