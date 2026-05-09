import torch
import torch.nn.functional as F
from utils import discount_rewards

import sys


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        # Creates a weight matrix and a bias vector initialized with the Pytorch default strategy
        self.fc1 = torch.nn.Linear(state_space, 12)
        self.fc2 = torch.nn.Linear(12, action_space)
        self.init_weights()

    def init_weights(self):
        """
            Initialize weights of our policy with small values because initially we don't want the
            network to be too confident about a certain action. 
        """
        
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                # Mean = 0 and std = 0.1, with greater values we risk greater logits so probabilities like [0.999, 0.001]
                torch.nn.init.normal_(m.weight, 0, 1e-1) # With all zeros, we will have a symmetric network so all neurons will learn similar things
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1) # Returns a vector of shape (action_space,) with the probability of each action


class Agent(object):
    """
        Implementation of a Policy Gradient agent, simple version of REINFORCE
    """
    def __init__(self, policy, lr=1e-2):
        self.train_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.batch_size = 1
        self.gamma = 0.98
        # self.gamma = 1
        self.observations = []
        self.actions = []
        self.rewards = []

    def episode_finished(self, episode_number):
        """
            When the episode is finished, the agent uses the given rewards to update the policy
        """
        # From self.actions = [Tensor([0.63]), Tensor([0.71])] to Tensor([[0.63], [0.71]])
        all_actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)
        all_rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)

        # Empty lists because the episode is finished
        self.observations, self.actions, self.rewards = [], [], []
        # In policy gradient we need to know after an action how is gone the continue of the episode
        # If an action brings to a long and good episode, we want to increase the probability of that action
        discounted_rewards = discount_rewards(all_rewards, self.gamma)
        # We normalize to stabilize the learning since longer episodes bring to larger values and reward above the average increase
        # action probability
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards) + 1e-10

        # element-wise multiplication, such that for each action we have -ln(outputnetwork)*disc_reward_normalized
        weighted_probs = all_actions * discounted_rewards 

        # print('all_actions:', all_actions)
        # print('discounted rewards:', discounted_rewards)
        # print('weighted probs:', weighted_probs)
        # sys.exit()

        # You want to perform gradient descent on the average loss, so to decrease the overall mean loss
        # => less probability for actions that led to below average rewards,
        # and more probability for actions that led to above average rewards.
        # Loss REINFORCE: loss = mean[-log π(a_t | s_t) * G_t]
        loss = torch.mean(weighted_probs)
        loss.backward()

        # batch_size = 1 so update the policy after each episode
        if (episode_number+1) % self.batch_size == 0:
            self.update_policy()

    def update_policy(self):
        """
            Update network weights
        """
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        # Get an observation returned by gym as a numpy array
        x = torch.from_numpy(observation).float().to(self.train_device)
        # Get a probability distribution over actions
        aprob = self.policy.forward(x)
        # Sample an action
        if evaluation: # In this case I don't want to explore, choose action with the highest probability
            action = torch.argmax(aprob).item()
        else: # In this case I want to explore so the action is chosen w.r.t. its probability given by the network
            dist = torch.distributions.Categorical(aprob)
            action = dist.sample().item()
        return action, aprob

    def store_outcome(self, observation, action_output, action_taken, reward):
        # action_output is aprob, probabilites given by the policy
        dist = torch.distributions.Categorical(action_output)
        action_taken = torch.Tensor([action_taken]).to(self.train_device)
        log_action_prob = -dist.log_prob(action_taken) # -ln(networkoutput) = negative log probability of the chosen action used with reward to compute loss

        # Store observation, taken action, prob/log-prob, reward received
        self.observations.append(observation)
        self.actions.append(log_action_prob) # e.g. after 2 timestep, self.actions = [Tensor([-log π(a0|s0)]), Tensor([-log π(a1|s1)])]
        self.rewards.append(torch.Tensor([reward]))
