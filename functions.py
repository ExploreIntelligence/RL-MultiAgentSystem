import networkx as nx
import pandas as pd
from params import *
from math import sqrt
# get magnitude of a vector

def get_magnitude(vector):
    return np.linalg.norm(vector)

def  position_initial(agents):
    flag=0

    for agent_idx in range(len(agents)):
        for j in range(len(agents)):
            if j != agent_idx:
                distance1= get_magnitude(agents[agent_idx].pos_new - agents[j].pos_new)
                distance2 = get_magnitude(agents[agent_idx].target - agents[j].target)
                if distance1<= min_dis or distance2<=min_dis :
                    flag=1
                    return flag
    return flag

'''define observation function'''
def get_agent_observations(agents):
    agent_observations = np.zeros([numAgents, numAgents * 4 + 2])
    for agent_idx in range(len(agents)):
        if agents[agent_idx].done==0:
            agent_observations[agent_idx][0] = agents[agent_idx].target[0] - agents[agent_idx].pos_new[0]
            agent_observations[agent_idx][1] = agents[agent_idx].target[1] - agents[agent_idx].pos_new[1]
            i=0
            for j in range(len(agents)):
                    distance = get_magnitude(agents[agent_idx].pos_new - agents[j].pos_new)
                    if distance <= maxvel*2 :#存在碰撞风险
                        agent_observations[agent_idx][4*i+2] =agents[j].pos_new[0]-agents[agent_idx].pos_new[0]
                        agent_observations[agent_idx][4*i+3] =agents[j].pos_new[1]-agents[agent_idx].pos_new[1]
                        agent_observations[agent_idx][4*i+4]= agents[j].vel_new[0]
                        agent_observations[agent_idx][4*i+5] = agents[j].vel_new[1]
                        i += 1
    return agent_observations

#计算奖励
def get_agent_rewards(agents):
    agent_rewards = np.zeros(numAgents)
    num_collide=0
    num_arrive=0
    for agent_idx in range(len(agents)):
        if agents[agent_idx].done == 0:
          a=agents[agent_idx]
          dis_tar_now =get_magnitude(a.target - a.pos_new)
          agent_rewards[agent_idx] += -dis_tar_now*0.1#以此为惩罚，让智能体不断接近目标

          #碰到边界的惩罚
          if a.pos_new[0]>=map_size[0] or a.pos_new[0]<=-map_size[0] or a.pos_new[1]>=map_size[1] or a.pos_new[1]<=-map_size[1]:
                 agent_rewards[agent_idx]-= 20
                 a.done = 1
                 a.collide=1
                 num_collide+=1


          #智能体之间发生碰撞
          for j in range(len(agents)):
              if j != agents.index(a):
                  distance = get_magnitude(a.pos_new - agents[j].pos_new)
                  # 与其他智能体碰撞：这一步的距离小于安全距离，或者两步之中路线有交叉且交点距离两者的起点相等
                  if distance <= min_safe_dis:
                      agent_rewards[agent_idx]-= 20
                      num_collide += 1
                      a.done = 1
                      a.collide = 1
                  else:
                      len1 = get_magnitude(a.pos_new - a.pos_old)
                      len2 = get_magnitude(agents[j].pos_new - agents[j].pos_old)
                      m1_x = (a.pos_new[0] + a.pos_old[0]) / 2
                      m1_y = (a.pos_new[1] + a.pos_old[1]) / 2
                      m2_x = (agents[j].pos_new[0] + agents[j].pos_old[0]) / 2
                      m2_y =  (agents[j].pos_new[1] + agents[j].pos_old[1]) / 2
                      if len1-len2<=min_safe_dis and sqrt((m1_x-m2_x)**2+(m1_y-m2_y)**2)<=min_safe_dis:
                          agent_rewards[agent_idx] -= 20
                          num_collide+=1
                          a.done = 1
                          a.collide = 1


          #到达目标
          if dis_tar_now<=dis_target and  a.collide ==0:
              agent_rewards[agent_idx]+=100
              a.done = 1
              num_arrive+=1


    return agent_rewards,num_collide,num_arrive
