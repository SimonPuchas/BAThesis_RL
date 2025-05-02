#!/usr/bin/env python3

import gym
import numpy
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from functools import reduce

import time
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os

import gzip

'''
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
'''

class PriorityReplayMemory(object):
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.alpha = alpha  # how much prioritization is used (1.0 is full, 0.0 is none)
        self.beta_start = beta_start    # initial beta for importance sampling (bias correction)
        self.beta_frames = beta_frames  # frames over which beta is annealed to 1.0
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.priorities = deque([], maxlen=capacity)
        self.frame = 0

    def push(self, *args, priority=None):
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        self.memory.append(Transition(*args))
        self.priorities.append(priority)

    def sample(self, batch_size):
        self.frame += 1
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

        priorities = numpy.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = numpy.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.FloatTensor(weights).to(device)
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if isinstance(priority, numpy.ndarray):
                self.priorities[idx] = float(priority.flatten()[0])
            else:
                self.priorities[idx] = float(priority)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        
        self.depth_fc1 = nn.Linear(8, 64)

        # network branch for goal info
        self.goal_fc = nn.Linear(2, 32)

        self.combined_fc1 = nn.Linear(64 + 32, 128)
        self.combined_fc2 = nn.Linear(128, 64)
        self.head = nn.Linear(64, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):

        # check if input is a single sample or a batch
        is_single = x.dim() == 1

        # if single sample, add batch dimension
        if is_single:
            x = x.unsqueeze(0)

        # split input into depth image and goal info
        depth_img = x[:, :8]
        goal_info = x[:, 8:]

        depth_features = F.silu(self.depth_fc1(depth_img))

        goal_features = F.silu(self.goal_fc(goal_info))

        combined_features = torch.cat((depth_features, goal_features), dim=1)
        combined_features = F.silu(self.combined_fc1(combined_features))
        combined_features = F.silu(self.combined_fc2(combined_features))

        output = self.head(combined_features)

        if is_single:
            return output.squeeze(0)
        else:
            return output


def select_action(state, eps_start, eps_end, eps_decay):
    global steps_done

    sample = random.random()
    
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * (steps_done/4) / eps_decay)  # steps / int, to make it easier to control the decay
    
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(0)[1].view(1, 1), eps_threshold
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), eps_threshold

def optimize_model(batch_size, gamma):
    if len(memory) < batch_size:
        return
    samples, indices, weights = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*samples))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    td_errors = torch.abs(expected_state_action_values.unsqueeze(1) - state_action_values).detach().cpu().numpy()
    td_errors_scalar = [float(err[0]) for err in td_errors]

    memory.update_priorities(indices, [err + 1e-5 for err in td_errors_scalar])

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    weighted_loss = (loss * weights.unsqueeze(1)).mean()

    # Optimize the model
    optimizer.zero_grad()
    weighted_loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def soft_update(target, source, tau=0.001):
    """
    Try a soft update of the target network parameters
    θ_target = τ * θ_source + (1 - τ) * θ_target
    
    Args:
        target (nn.Module): Target network to update
        source (nn.Module): Source network to copy from
        tau (float): Interpolation parameter - typically a small value like 0.001
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

# logic to plot the rewards
def plot_rewards(episode_rewards, episode_n, window_size=100):
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, 'b.', label='individual episodes', alpha=0.5)
    
    if len(episode_rewards) >= window_size:
        moving_avg = []
        for i in range(len(episode_rewards) - window_size + 1):
            window_avg = numpy.mean(episode_rewards[i:i+window_size])
            moving_avg.append(window_avg)
        plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, 'b-', 
                label=f'moving average of last {window_size} episodes')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN on TurtleBot3World-v0')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_turtlebot3_openai_example')
    plt.savefig(f'{pkg_path}/training_results/reward_plot.png')
    plt.close()
    
# logic to save the model
def save_checkpoint(episode, policy_net, target_net, optimizer, memory, episode_rewards, steps_done, epsilon, highest_reward):
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_turtlebot3_openai_example')
    checkpoint_dir = f'{pkg_path}/checkpoints'
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint = {
        'episode': episode,
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'memory': memory,
        'episode_rewards': episode_rewards,
        'steps_done': steps_done,
        'epsilon': epsilon,
        'highest_reward': highest_reward
    }
    
    checkpoint_path = f'{checkpoint_dir}/checkpoint_episode_{episode}.pt'
    #torch.save(checkpoint, checkpoint_path)
    # use gzip to compress checkpoint files as they are quite large
    with gzip.open(checkpoint_path + '.gz', 'wb') as f:
        torch.save(checkpoint, f)

    rospy.loginfo(f"Checkpoint saved at episode {episode}")
    
    # Keep only the last 4 checkpoints
    checkpoint_files = sorted(
    [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_episode_')],
    key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if len(checkpoint_files) > 4:
        for old_file in checkpoint_files[:-4]:
            os.remove(os.path.join(checkpoint_dir, old_file))
            rospy.loginfo(f"Removed old checkpoint: {old_file}")

# logic to load the checkpoint
def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        rospy.logwarn(f"Checkpoint file {checkpoint_path} not found")
        return None
    
    rospy.loginfo(f"Loading checkpoint from {checkpoint_path}")
    #checkpoint = torch.load(checkpoint_path)
    # use gzip to load the file, if its not compressed use standard torch.load
    try:
        if checkpoint_path.endswith('.gz'):
            with gzip.open(checkpoint_path, 'rb') as f:
                return torch.load(f)
        else:
            return torch.load(checkpoint_path)
    except (gzip.BadGzipFile, RuntimeError, EOFError) as e:
        rospy.logerr(f"Failed to load checkpoint: {e}")
        return None

def get_latest_checkpoint():
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_turtlebot3_openai_example')
    checkpoint_dir = f'{pkg_path}/checkpoints'
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = sorted(
    [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_episode_')],
    key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if not checkpoint_files:
        return None
    
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
    return latest_checkpoint

# import our training environment
if __name__ == '__main__':

    rospy.init_node('turtlebot3_world_qlearn', anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot3/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_turtlebot3_openai_example')
    outdir = pkg_path + '/training_results'
    # env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    gamma = rospy.get_param("/turtlebot3/gamma")
    epsilon_start = rospy.get_param("/turtlebot3/epsilon_start")
    epsilon_end = rospy.get_param("/turtlebot3/epsilon_end")
    epsilon_decay = rospy.get_param("/turtlebot3/epsilon_decay")
    n_episodes = rospy.get_param("/turtlebot3/n_episodes")
    batch_size = rospy.get_param("/turtlebot3/batch_size")
    target_update = rospy.get_param("/turtlebot3/target_update")

    running_step = rospy.get_param("/turtlebot3/running_step")

    load_model = rospy.get_param("/turtlebot3/load_model", False)
    reset_epsilon = rospy.get_param("/turtlebot3/reset_epsilon", False)

    alpha = rospy.get_param("/turtlebot3/alpha", 0.6)
    beta_start = rospy.get_param("/turtlebot3/beta_start", 0.4)
    beta_frames = rospy.get_param("/turtlebot3/beta_frames", 100000)

    tau = rospy.get_param("/turtlebot3/tau", 0.001)
    use_soft_update = rospy.get_param("/turtlebot3/use_soft_update", True)

    # Initialises the algorithm that we are going to use for learning
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    n_observations = 10

    # initialize networks with input and output sizes
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    # potentially use a LRScheduler

    #memory = ReplayMemory(100000)
    memory = PriorityReplayMemory(100000, alpha, beta_start, beta_frames)
    episode_durations = []
    steps_done = 0

    start_time = time.time()
    highest_reward = 0
    episode_rewards = []

    latest_checkpoint_path = get_latest_checkpoint()
    start_episode = 0

    if latest_checkpoint_path and load_model:
                checkpoint = load_checkpoint(latest_checkpoint_path)
                if checkpoint:
                    start_episode = checkpoint['episode'] + 1
                    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                    target_net.load_state_dict(checkpoint['target_net_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    memory = checkpoint['memory']
                    episode_rewards = checkpoint['episode_rewards']
                    steps_done = checkpoint['steps_done']
                    highest_reward = checkpoint['highest_reward']
                    if reset_epsilon:
                        epsilon = 0.3
                        rospy.loginfo("Resetting epsilon to {epsilon}")
                    rospy.loginfo(f"Resuming training from episode {start_episode}")

    # Starts the main training loop: the one about the episodes to do
    for i_episode in range(start_episode, n_episodes):
        rospy.logdebug("############### START EPISODE=>" + str(i_episode))

        cumulated_reward = 0
        done = False

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = torch.tensor(observation, device=device, dtype=torch.float)
        #state = ''.join(map(str, observation))

        for t in count():
            #rospy.logwarn("############### Start Step=>" + str(t))
            # Select and perform an action
            action, epsilon = select_action(state, epsilon_start, epsilon_end, epsilon_decay)
            rospy.logdebug("Next action is:%d", action)

            observation, reward, done, info = env.step(action.item())
            rospy.logdebug(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            reward = torch.tensor([reward], device=device)

            #next_state = ''.join(map(str, observation))
            next_state = torch.tensor(observation, device=device, dtype=torch.float)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Perform one step of the optimization (on the policy network)
            rospy.logdebug("# state we were=>" + str(state))
            rospy.logdebug("# action that we took=>" + str(action))
            rospy.logdebug("# reward that action gave=>" + str(reward))
            rospy.logdebug("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logdebug("# State in which we will start next step=>" + str(next_state))
            optimize_model(batch_size, gamma)
            if use_soft_update:
                soft_update(target_net, policy_net, tau)
            else:
                if i_episode % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            if done:
                episode_durations.append(t + 1)
                episode_rewards.append(cumulated_reward)
                rospy.logdebug("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
                break
            else:
                rospy.logdebug("NOT DONE")
                state = next_state

        if i_episode % 100 == 0:
            plot_rewards(episode_rewards, i_episode)

        if (i_episode + 1) % 100 == 0:
            save_checkpoint(
                i_episode + 1, 
                policy_net, 
                target_net, 
                optimizer, 
                memory, 
                episode_rewards, 
                steps_done, 
                epsilon,
                highest_reward
            )

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(i_episode + 1) + " - gamma: " + str(
            round(gamma, 2)) + " - epsilon: " + str(round(epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + " - Time: %d:%02d:%02d" % (h, m, s) + " - Steps done: " + str(steps_done)))

    rospy.loginfo(("\n|" + str(n_episodes) + "|" + str(gamma) + "|" + str(epsilon_start) + "*" +
                   str(epsilon_decay) + "|" + str(highest_reward) + "| PICTURE |"))

    plot_rewards(episode_rewards, i_episode)

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
