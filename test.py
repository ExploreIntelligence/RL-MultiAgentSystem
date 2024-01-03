import matplotlib.pyplot as plt
from functions import *
from SAC import *
import time
from params import *


SAC = SAC(n_states, n_actions,hidden_dim)
SAC.load_models(30000)
replay_buffer = ReplayBuffer(replay_buffer_size)
reward_record =[]
Episode=[]
collide_record=[]

# plot initial locations
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

'''Episode Run'''
start_time = time.time()
episode_idx=0
while episode_idx<max_episode:
    #记录一轮中发生碰撞和到达目的地完成任务的情况
    num_collide=0
    num_arrive=0

    #为了生成目标区域，这里的x,y表示智能体所在的初始区域在x,y轴上移动的距离，由此得到目标区域的位置
    x = random.randint(10, 100) * ((-1) ** random.randint(1, 2))  # 15, 110
    y = random.randint(10, 100) * ((-1) ** random.randint(1, 2))  #

    agents = [Agent(x,y) for i in range(numAgents)]

    #判断初始位置和最终位置是否离得太近，如果太近就重新生成初始位置和目标位置
    flag =position_initial(agents)
    if flag==1:
        continue
    else:
        episode_idx+=1

    '''plot the initial positions'''
    for a in agents:
        ax.scatter(a.pos_new[0],a.pos_new[1], c='r', marker='o')
        ax.scatter(a.target[0], a.target[1], c='b', marker='o')

    step_cnt = 0
    reward=0
    done_last = [agents[i].done for i in range(numAgents)]

    agent_observations = get_agent_observations(agents)#获得每个智能体的观测信息

    # perform steps
    while not all(done_last):#如果还有智能体没有完成任务就继续执行本轮
        # get agent actions and new positions
        actions_np = SAC.select_action(agent_observations,True)  # 返回每个agent选择的动作
        for a in agents:
            if done_last[agents.index(a)]==0:
                a.vel_new = np.array([action_unnormalized(actions_np[agents.index(a)][0], maxSpeed, -maxSpeed),
                                                  action_unnormalized(actions_np[agents.index(a)][1], maxSpeed, -maxSpeed)])
            else:
                a.vel_new=np.array([0,0])
            a.pos_new = a.pos_old + a.vel_new
            if a.collide==1:
                a.pos_new=np.array([-20000,-20000])#因碰撞而坠毁

        #get observations_next of followers
        agent_observations_next = get_agent_observations(agents)#返回的是一个follower_nums*4的数组

        # get rewards of followers
        agent_rewards, collides, arrive = get_agent_rewards(agents)
        num_collide += collides
        num_arrive += arrive

        #update the historical values
        for a in agents:
            a.pos_old = a.pos_new
            a.vel_old = a.vel_new

        if step_cnt >= max_steps-1:
             for a in agents:
                 a.done = 1

        agent_observations=agent_observations_next


        # plot new positions
        ax.clear()    # clear old points
        for a in agents:
            ax.scatter(a.pos_new[0],a.pos_new[1], c='r', marker='o')
            ax.scatter(a.target[0], a.target[1], c='b', marker='o')
        ax.text(-map_size[0],map_size[1],str(int(numAgents))+' agents', color='r', fontsize=12)
        ax.text(-map_size[0]+150,map_size[1],str(int(step_cnt))+' steps', color='k', fontsize=12)
        #set axis
        ax.set_xlim(-map_size[0], map_size[0])
        ax.set_ylim(-map_size[1], map_size[1])

        # plot
        plt.pause(0.001)
        done_last = [agents[i].done for i in range(numAgents)]
        step_cnt = step_cnt+1
    reward_record.append(reward)
    Episode.append(episode_idx)
    collide_record.append(num_collide)
    print('Episode,agent_reward,num_collide,num_arrive' ,episode_idx, reward,num_collide,num_arrive)


'''Show the run time'''
end_time = time.time()
print('The total run time is: ',end_time-start_time,'s')

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(121)
ax.plot(Episode,reward_record,linewidth=1)
ax.set_xlabel('episode',fontsize=15)
ax.set_ylabel('reward',fontsize=15)
ax.tick_params(axis='x',labelsize=15)
ax.tick_params(axis='y',labelsize=15)
plt.savefig('reward.jpg',dpi=600)

ax = fig.add_subplot(122)
ax.plot(Episode,collide_record,linewidth=1)
ax.set_xlabel('episode',fontsize=15)
ax.set_ylabel('collide_num',fontsize=15)
ax.tick_params(axis='x',labelsize=15)
ax.tick_params(axis='y',labelsize=15)
plt.savefig('collide_num.jpg',dpi=600)
plt.show()