import itertools as it

import gymnasium as gym
import torch

from dqnn import Agent


def main():
    env = gym.make("MountainCar-v0", render_mode="human")
    # Change fps for speed up training (0 = max speed)
    original_speed = env.metadata.pop("render_fps")
    env.metadata["render_fps"] = 0

    agent = Agent(n_observations=2, n_actions=3)
    agent = train(env, agent, n_games=300, print_every=10)
    env.close()
    env = gym.make("MountainCar-v0", render_mode="human")
    env.metadata["render_fps"] = original_speed
    play(env, agent, 10)
    env.close()


def train(env: gym.Env, agent: Agent, n_games: int, print_every: int = 100):
    game_steps = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for n_game in range(n_games):
        # Start each new game in a fresh env
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # Count how many steps does it take to solve the env
        for step in it.count():
            action = agent.choose_action(state)

            # Apply the action to the env
            observation, reward, terminated, truncated, _ = env.step(action.item())

            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.learn()

            # Update target net
            agent.update_target_net()

            if done:
                game_steps.append(step + 1)
                if (n_game + 1) % print_every == 0:
                    print(
                        f"Game {n_game + 1} is done [Avg {sum(game_steps)/len(game_steps):.2f} steps]"
                    )
                break
    return agent


def play(env: gym.Env, agent: Agent, n_games: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for n_game in range(n_games):
        # Start each new game in a fresh env
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # Count how many steps does it take to solve the env
        for step in it.count():
            with torch.no_grad():
                action = agent.policy_net(state).max(1).indices.view(1, 1)

            # Apply the action to the env
            observation, _, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Move to the next state
            state = next_state

            if done:
                print(
                    f"Game {n_game + 1} is done - Steps taken: {step + 1} - Final status: {'Solved' if terminated else 'Not Solved'}"
                )
                break


if __name__ == "__main__":
    main()
