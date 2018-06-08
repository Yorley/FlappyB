import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical

torch.set_default_tensor_type('torch.DoubleTensor')


class Agent(torch.nn.Module):
    def preravel_observation(self, observation):
        ravel = observation.data.numpy().ravel()
        return self.fc(torch.from_numpy(ravel))

    def __init__(self, allowed_actions, channels, learning_rate):
        super(Agent, self).__init__()
        self.training_mode = False
        self.actions = allowed_actions
        #Convolutional layers
        self.layers = []
        self.model_components = []

        layer = torch.nn.Conv2d(channels, 3, 3, 1, 1)
        self.layers += [[layer, F.relu]]
        self.model_components.append(layer)

        layer = torch.nn.Conv2d(3, 5, 3, 1, 1)
        self.layers += [[layer, F.relu]]
        self.model_components.append(layer)

        layer = torch.nn.Conv2d(5, 9, 3, 1, 1)
        self.layers += [[layer, F.relu]]
        self.model_components.append(layer)

        layer = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.layers += [[layer, F.relu]]
        self.model_components.append(layer)

        #Dropout
        layer = torch.nn.Dropout(0.5)
        self.layers += [[layer, F.relu]]
        self.model_components.append(layer)

        #Output
        self.fc = torch.nn.Linear(9*40*40, 2)
        # Fully Connected - Output
        self.output = [self.preravel_observation, F.softmax]
        self.model_components.append(layer)

        self.model = torch.nn.Sequential(*self.model_components)
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pickAction(self, observation):
        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        x = torch.from_numpy(observation)
        x = torch.unsqueeze(torch.unsqueeze(x, 0), 0)
        x.requires_grad = True
        for layer in self.layers:
            x = layer[1](layer[0](x))
        output_preactivation = self.output[0](x)
        x = self.output[1](output_preactivation, 0)
        log_loss = Categorical(x)
        raw_action = log_loss.sample()
        action = self.actions[raw_action.item()]
        return action, -1 * log_loss.log_prob(raw_action).item()

    def train(self, rewards, loss):
        last_reward = len(rewards) - 1
        reward = rewards[last_reward]
        if reward < 0.0:
            reward = -5.0
            rewards[last_reward] = reward
        if reward > 0.0:
            reward = 5.0
            rewards[last_reward] = reward
        print(reward)
        while reward != 0:
            if reward > 0:
                reward -= 0.125
            else:
                reward += 0.125
            if rewards[last_reward-1] > 0:
                break
            rewards[last_reward-1] = reward
        rewards = torch.Tensor(rewards)
        loss = torch.from_numpy(np.array(loss, dtype=np.float64))
        policy_loss = (rewards * loss).sum()
        policy_loss.requires_grad = True
        policy_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
