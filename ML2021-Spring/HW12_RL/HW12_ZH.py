# **Homework 12 - Reinforcement Learning**
'''
若有任何問題，歡迎來信至助教信箱 ntu-ml-2021spring-ta@googlegroups.com
'''
## 前置作業
'''
首先我們需要安裝必要的系統套件及 PyPi 套件。
gym 這個套件由 OpenAI 所提供，是一套用來開發與比較 Reinforcement Learning 演算法的工具包（toolkit）。
而其餘套件則是為了在 Notebook 中繪圖所需要的套件。
接下來，設置好 virtual display，並引入所有必要的套件。
'''

#from pyvirtualdisplay import Display
#virtual_display = Display(visible=0, size=(1400, 900))
#virtual_display.start()


import matplotlib.pyplot as plt

#from IPython import display

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.notebook import tqdm

'''
# 請不要更改 random seed !!!!
# 不然在judgeboi上 你的成績不會被reproduce !!!!
'''

seed = 543 # Do not change this
def fix(env, seed):
  env.seed(seed)
  env.action_space.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.set_deterministic(True)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

# 最後，引入 OpenAI 的 gym，並建立一個 [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2/) 環境。

import gym
import random
import numpy as np

env = gym.make('LunarLander-v2')

fix(env, seed)

import time
start = time.time()

'''
Observation / State
首先，我們可以看看 environment 回傳給 agent 的 observation 究竟是長什麼樣子的資料：
'''

print(env.observation_space)
'''
Box(8,) 說明我們會拿到 8 維的向量作為 observation，其中包含：垂直及水平座標、速度、角度、加速度等等，這部分我們就不細說。

Action
而在 agent 得到 observation 和 reward 以後，能夠採取的動作有：
'''

print(env.action_space)

'''
Discrete(4) 說明 agent 可以採取四種離散的行動：

    - 0 代表不採取任何行動
    - 2 代表主引擎向下噴射
    - 1, 3 則是向左右噴射
接下來，我們嘗試讓 agent 與 environment 互動。 在進行任何操作前，建議先呼叫 reset() 函式讓整個「環境」重置。 而這個函式同時會回傳「環境」最初始的狀態。
'''

initial_state = env.reset()
print(initial_state)

# 接著，我們試著從 agent 的四種行動空間中，隨機採取一個行動

random_action = env.action_space.sample()
print(random_action)

'''
再利用 step() 函式讓 agent 根據我們隨機抽樣出來的 random_action 動作。 而這個函式會回傳四項資訊：

observation / state
reward
完成與否
其餘資訊
'''

observation, reward, done, info = env.step(random_action)

'''
第一項資訊 observation 即為 agent 採取行動之後，agent 對於環境的 observation 或者說環境的 state 為何。 而第三項資訊 done 則是 True 或 False 的布林值，當登月小艇成功著陸或是不幸墜毀時，代表這個回合（episode）也就跟著結束了，此時 step() 函式便會回傳 done = True，而在那之前，done 則保持 False。
'''

print(done)

'''
Reward
而「環境」給予的 reward 大致是這樣計算：

  - 小艇墜毀得到 -100 分
  - 小艇在黃旗幟之間成功著地則得 100~140 分
  - 噴射主引擎（向下噴火）每次 -0.3 分
  - 小艇最終完全靜止則再得 100 分
  - 小艇每隻腳碰觸地面 +10 分
Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame.
'''

print(reward) # after doing a random action (0), the immediate reward is stored in this 

'''
Random Agent
最後，在進入實做之前，我們就來看看這樣一個 random agent 能否成功登陸月球：
'''

env.reset()
'''
img = plt.imshow(env.render(mode='rgb_array'))

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)

    img.set_data(env.render(mode='rgb_array'))
    display.display(plt.gcf())
    display.clear_output(wait=True)
'''
'''
Policy Gradient
現在來搭建一個簡單的 policy network。 我們預設模型的輸入是 8-dim 的 observation，輸出則是離散的四個動作之一：
'''

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

'''
再來，搭建一個簡單的 agent，並搭配上方的 policy network 來採取行動。 這個 agent 能做到以下幾件事：

    - learn()：從記下來的 log probabilities 及 rewards 來更新 policy network。
    - sample()：從 environment 得到 observation 之後，利用 policy network 得出應該採取的行動。 而此函式除了回傳抽樣出來的action，也會回傳此次抽樣的 log probabilities。
'''

class PolicyGradientAgent(): 
    def __init__(self, network):
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)
         
    def forward(self, state):
        return self.network(state)
    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum() # You don't need to revise this to pass simple baseline (but you can)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state))
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

# 最後，建立一個 network 和 agent，就可以開始進行訓練了。
network = PolicyGradientNetwork()
agent = PolicyGradientAgent(network)
# agent = PolicyGradientAgent()

'''
訓練 Agent
現在我們開始訓練 agent。 透過讓 agent 和 environment 互動，我們記住每一組對應的 log probabilities 及 reward，並在成功登陸或者不幸墜毀後，回放這些「記憶」來訓練 policy network。
'''

agent.network.train()  # 訓練前，先確保 network 處在 training 模式
EPISODE_PER_BATCH = 5  # 每蒐集 5 個 episodes 更新一次 agent
NUM_BATCH = 400        # 總共更新 400 次

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

            action, log_prob = agent.sample(state) # at , log(at|st)
            next_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob) # [log(a1|s1), log(a2|s2), ...., log(at|st)]
            # seq_rewards.append(reward)
            state = next_state
            total_reward += reward
            total_step += 1
            rewards.append(reward) #改這裡
            # ! 重要 ！
            # 現在的reward 的implementation 為每個時刻的瞬時reward, 給定action_list : a1, a2, a3 ......
            #                                                       reward :     r1, r2 ,r3 ......
            # medium：將reward調整成accumulative decaying reward, 給定action_list : a1,                         a2,                           a3 ......
            #                                                       reward :     r1+0.99*r2+0.99^2*r3+......, r2+0.99*r3+0.99^2*r4+...... ,r3+0.99*r4+0.99^2*r5+ ......
            # boss : implement DQN
            if done:
                final_rewards.append(reward)
                total_rewards.append(total_reward)
                break

    print(f"rewards looks like ", np.shape(rewards))  
    print(f"log_probs looks like ", np.shape(log_probs))     
    # 紀錄訓練過程
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_final_reward = sum(final_rewards) / len(final_rewards)
    avg_total_rewards.append(avg_total_reward)
    avg_final_rewards.append(avg_final_reward)
    prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

    # 更新網路
    # rewards = np.concatenate(rewards, axis=0)
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # 將 reward 正規標準化
    agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
    print("logs prob looks like ", torch.stack(log_probs).size())
    print("torch.from_numpy(rewards) looks like ", torch.from_numpy(rewards).size())


'''
訓練結果
訓練過程中，我們持續記下了 avg_total_reward，這個數值代表的是：每次更新 policy network 前，我們讓 agent 玩數個回合（episodes），而這些回合的平均 total rewards 為何。 理論上，若是 agent 一直在進步，則所得到的 avg_total_reward 也會持續上升，直至 250 上下。 若將其畫出來則結果如下：
'''
end = time.time()
plt.figure(1)
plt.plot(avg_total_rewards)
plt.title("Total Rewards")
#plt.show()

'''
另外，avg_final_reward 代表的是多個回合的平均 final rewards，而 final reward 即是 agent 在單一回合中拿到的最後一個 reward。 如果同學們還記得環境給予登月小艇 reward 的方式，便會知道，不論回合的最後小艇是不幸墜毀、飛出畫面、或是靜止在地面上，都會受到額外地獎勵或處罰。 也因此，final reward 可被用來觀察 agent 的「著地」是否順利等資訊。
'''
plt.figure(2)
plt.plot(avg_final_rewards)
plt.title("Final Rewards")

#plt.show()


# 訓練時間
print(f"total time is {end-start} sec")

plt.show()
