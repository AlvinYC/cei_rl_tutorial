# **Homework 12 - Reinforcement Learning**

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from tqdm import tqdm
import gym
import random
import numpy as np
import time
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
#torch.use_deterministic_algorithms(False)

seed = 543 # Do not change this
def fix(env, seed):
  env.seed(seed)
  env.action_space.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  #torch.set_deterministic(True)
  torch.set_deterministic(False)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
class PolicyGradientNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)
class PolicyGradientAgent(): 
    def __init__(self, network):
        self.network = network.cuda()
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)
         
    def forward(self, state):
        return self.network(state)
    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum() # You don't need to revise this to pass simple baseline (but you can)

        self.optimizer.zero_grad()
        #loss.cuda()
        loss.backward()
        self.optimizer.step()
        
    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state).cuda())
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def save(self, PATH): # You should not revise this
        Agent_Dict = {
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict()
        }
        torch.save(Agent_Dict, PATH)

    def load(self, PATH): # You should not revise this
        checkpoint = torch.load(PATH)
        self.network.load_state_dict(checkpoint["network"])
        #如果要儲存過程或是中斷訓練後想繼續可以用喔 ^_^
        self.optimizer.load_state_dict(checkpoint["optimizer"])
env = gym.make('LunarLander-v2')

fix(env, seed)
start = time.time()


'''
random_action = env.action_space.sample()
print(random_action)
observation, reward, done, info = env.step(random_action)
print(reward) # after doing a random action (0), the immediate reward is stored in this 
'''
initial_state = env.reset()
env.reset()

# 最後，建立一個 network 和 agent，就可以開始進行訓練了。
network = PolicyGradientNetwork()
agent = PolicyGradientAgent(network)

#訓練 Agent
agent.network.train()  # 訓練前，先確保 network 處在 training 模式

EPISODE_PER_BATCH = 5  # 每蒐集 5 個 episodes 更新一次 agent
#NUM_BATCH = 400        # 總共更新 400 次
NUM_BATCH = 50        # 總共更新 400 次
avg_total_rewards, avg_final_rewards = [], []
prg_bar = tqdm(range(NUM_BATCH))

for batch in prg_bar:
    log_probs, rewards = [], []
    total_rewards, final_rewards = [], []

    # 蒐集訓練資料
    for episode in range(EPISODE_PER_BATCH):
        
        state = env.reset()
        total_reward, total_step = 0, 0
        seq_rewards = []
        while True:

            #action, log_prob = agent.sample(state) # at , log(at|st)
            action, log_prob = agent.sample(state) # at , log(at|st)
            next_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob) # [log(a1|s1), log(a2|s2), ...., log(at|st)]
            # seq_rewards.append(reward)
            state = next_state
            total_reward += reward
            total_step += 1
            rewards.append(reward) #改這裡

            if done:
                final_rewards.append(reward)
                total_rewards.append(total_reward)
                break

    #print(f"rewards looks like ", np.shape(rewards))  
    #print(f"log_probs looks like ", np.shape(log_probs))     
    # 紀錄訓練過程
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_final_reward = sum(final_rewards) / len(final_rewards)
    avg_total_rewards.append(avg_total_reward)
    avg_final_rewards.append(avg_final_reward)
    prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

    # 更新網路
    # rewards = np.concatenate(rewards, axis=0)
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # 將 reward 正規標準化
    #agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
    agent.learn(torch.stack(log_probs), torch.from_numpy(rewards).to(device))
    #print("logs prob looks like ", torch.stack(log_probs).size())
    #print("torch.from_numpy(rewards) looks like ", torch.from_numpy(rewards).size())

end = time.time()
plt.figure(1)
plt.plot(avg_total_rewards)
plt.title("Total Rewards")

plt.figure(2)
plt.plot(avg_final_rewards)
plt.title("Final Rewards")

# 訓練時間
print(f"total time is {end-start} sec")

plt.show()


# testing model

fix(env, seed)
agent.network.eval()  # 測試前先將 network 切換為 evaluation 模式
NUM_OF_TEST = 5 # Do not revise it !!!!!
test_total_reward = []
action_list = []
for i in range(NUM_OF_TEST):
  actions = []
  state = env.reset()

  img = plt.imshow(env.render(mode='rgb_array'))

  total_reward = 0

  done = False
  while not done:
      action, _ = agent.sample(state)
      actions.append(action)
      state, reward, done, _ = env.step(action)

      total_reward += reward

      img.set_data(env.render(mode='rgb_array'))
      display.display(plt.gcf())
      display.clear_output(wait=True)
  print(total_reward)
  test_total_reward.append(total_reward)

  action_list.append(actions) #儲存你測試的結果
  print("length of actions is ", len(actions))

PATH = "Action_List_test.npy" # 可以改成你想取的名字或路徑
np.save(PATH ,np.array(action_list)) 