from src.envs.pong_env import PongEnvironment
from src.agents.policy_agent import PolicyAgent
import numpy as np


def train():

    env = PongEnvironment()
    obs, _ = env.reset()

    # Agents
    agent_0 = PolicyAgent(agent_id=0, action_space=env.action_space[0])
    agent_1 = PolicyAgent(agent_id=1, action_space=env.action_space[1])

    num_episodes = 1
    max_steps_per_episode = 5000

    # Track best performance
    best_reward = -float('inf')
    best_agent = None

    print("Starting Multi-Agent DQN Training...")



    for ep in range(num_episodes):
        obs, _ = env.reset()
        
        done = False
        total_reward_0 = 0
        total_reward_1 = 0
        step = 0

        while not done and step < max_steps_per_episode:
            # Both agents choose actions
            action_0 = agent_0.act(obs[0])
            action_1 = agent_1.act(obs[1])
            actions = [action_0, action_1]

            # Step environment
            next_obs, rewards, done, truncated, info = env.step(actions)
            done_flag = done or truncated

            # Update both agents
            agent_0.update(
                obs=obs[0],
                action=action_0,
                reward=rewards[0],
                next_obs=next_obs[0],
                done=done_flag
            )
            
            agent_1.update(
                obs=obs[1],
                action=action_1,
                reward=rewards[1],
                next_obs=next_obs[1],
                done=done_flag
            )

            obs = next_obs
            total_reward_0 += rewards[0]
            total_reward_1 += rewards[1]
            step += 1

        # Print episode results
        if (ep + 1) % 5 == 0:
            print(f"Episode {ep+1}/{num_episodes} | "
                f"Agent 0 Reward: {total_reward_0:.2f} | Epsilon: {agent_0.epsilon:.3f} | "
                f"Agent 1 Reward: {total_reward_1:.2f} | Epsilon: {agent_1.epsilon:.3f}")

        # Save the best performing agent (either 0 or 1)
        if total_reward_0 > best_reward:
            best_reward = total_reward_0
            best_agent = 0
            agent_0.save_model("checkpoints/trained_agent.pt")
            print(ep, end=': ')
            print(f"New best Agent 0! Reward: {total_reward_0:.1f}")
        
        if total_reward_1 > best_reward:
            best_reward = total_reward_1
            best_agent = 1
            agent_1.save_model("checkpoints/trained_agent.pt")
            print(ep, end=': ')
            print(f"New best Agent 1! Reward: {total_reward_1:.1f}")



    if best_agent == 0:
        agent_0.save_model("checkpoints/trained_agent.pt")
        print(f"Final model saved: Agent 0 with reward {best_reward:.1f}")
    else:
        agent_1.save_model("checkpoints/trained_agent.pt")
        print(f"Final model saved: Agent 1 with reward {best_reward:.1f}")

    env.close()

    print("Training completed!")
    print(f"Best Reward: {best_reward:.1f} (Agent {best_agent})")


if __name__ == "__main__":
    train()