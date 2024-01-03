import numpy as np
import random

numAgents = 30#智能体的总数量
maxSpeed = 6     # L, maximum speed，这是x或y轴方向的最大值
maxvel=6*(2**0.5)#这是速度的最大值
min_safe_dis = maxvel  # 智能体之间的安全距离，小于这个距离就认为发生了碰撞
dis_target = maxvel/2 # 当离目标的距离小于k时，认为到达目标，给予奖励
min_dis=maxvel*2 #初始时和结束时智能体之间的距离的最小值
ini_loc =160      # 初始时，所有的智能体位于一个正方形区域，该值表是正方形的边长的一半。正方形的中心是原点
map_size = [600,600]     # maximum map size，画图的画布的xy轴的范围

#关于SAC算法的相关参数
max_episode =5#30000      # define the maximum episodes
max_episode_test = 1000
max_steps = 40      # the max steps per episode
batch_size = 128 #每次更新网络参数，从回放池中抽取的样本数
replay_buffer_size =5000#回放池的容量
n_actions=2#x，y轴的动作
n_states=numAgents*4+2#42,每个agent观察到的状态的维度，所有智能体的相对位置+速度+目标相对位置
before_training = 4 #更新参数前执行的步数
hidden_dim=128 #网络隐藏层的单元数

'''bird class'''
class Agent:
    def __init__(self,x,y):
        self.pos_old = np.array([random.uniform(-ini_loc,ini_loc),random.uniform(-ini_loc,ini_loc)])  #智能体的位置
        self.vel_old = np.array([random.uniform(-maxSpeed,maxSpeed),random.uniform(-maxSpeed,maxSpeed)])  # 智能体的速度
        self.target=np.array([random.uniform(-ini_loc,ini_loc)+x,random.uniform(-ini_loc,ini_loc)+y]) # 智能体的目标位置
        self.pos_new = self.pos_old   # new position
        self.vel_new = self.vel_old   # new velocity
        self.done=0#智能体是否完成任务或者发生碰撞而结束任务
        self.collide=0#智能体是否发生了碰撞