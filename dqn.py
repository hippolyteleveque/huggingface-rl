import random
from torch import nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):

    def copy_from(self, other):
        self.load_state_dict(other.state_dict())

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.layers = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
        )

    def forward(self, x):
        return self.layers(x)


def epsilon_policy(env, network, state, eps):
    if random.uniform(0, 1) < eps:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            logits = network(state)
            action = torch.argmax(logits).item()
    return action


def train(
    env,
    network,
    num_episodes,
    buffer_size,
    episode_size,
    eps,
    batch_size,
    reset_target_steps,
    gamma,
    learning_rate,
):
    buffer = []
    target_network = DQN()
    target_network.copy_from(network)
    # the target network does not need to be trainable

    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    step_number = 0

    for _ in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state).unsqueeze(0)  # as batch
        for _ in range(episode_size):
            optimizer.zero_grad()
            action = epsilon_policy(env, network, state, eps)

            new_state, reward, terminated, truncated, info = env.step(action)
            # Append to the replay buffer
            buffer.append((state, action, reward, new_state))

            # sample from the replay buffer
            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                buffer = buffer[-buffer_size:]
            else:
                continue  # Skip if not enough samples

            # Concatenate as a batch all the input state from the batch
            batch_q_inputs = torch.cat(
                [transition[0].unsqueeze(0) for transition in batch]
            )

            # Concatenate as a batch all the new_state from the batch
            batch_target_inputs = torch.cat(
                [transition[3].unsqueeze(0) for transition in batch]
            )

            actions = torch.tensor([transition[1] for transition in batch]).unsqueeze(1)
            q_outputs = torch.gather(network(batch_q_inputs), 1, actions)

            with torch.no_grad():
                target_outputs = target_network(batch_target_inputs).max(dim=1)[0]
                ys = gamma * target_outputs + torch.tensor(
                    [transition[2] for transition in batch]
                )
                ys = ys.unsqueeze(1)

            loss = criterion(ys, q_outputs)
            # only on the network
            loss.backward()
            optimizer.step()

            step_number += 1

            if terminated:
                break

            # We reset the target network to the qnetwork parameters
            if step_number % reset_target_steps == 0:
                target_network.copy_from(network)
