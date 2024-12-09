"""
@File  : fusion_DQN.py
@Author: DavidLin
@Date  : 2024/12/7
@Contact : davidlin659562@gmail.com
@Description : 
This is the main program file for this article "RL-based USV Path Planning Under the Marine Multimodal Features Considerations"
1. The preprocessing steps need to be completed according to the Image processing module and Meteorological analysis module.
(Of course, the processed test data has also been prepared. )
2. Interactive environment file is Multimodal_characteristic_Marine_environment.py.
3. In all files, the paths of your own data file and parameters that can be adjusted have been marked.
If you have any questions or suggestions, please feel free to contact me.
"""

import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gym
import time
import pickle
import datetime
import Multimodal_characteristics_Marine_environment
import os
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# render: online visualization
# Double structure/Multivariate Weighted Dueling Network/Priority Sampling Mechanism : Ablation experiments

RENDER = True
LAST_RENDER = False
OLD_NET=  False
'''
For Ablation experiments
fusion DQN: All is True
No Double/No Dueling/No Priority: The corresponding module is False, others are True.
No multimodal: Dueling and Multimodal_characteristics_Marine_environment need to reduce the part of 
Multimodal information fusion and learning (containing state space, multivariate weighted Dueling network)
'''
DOUBLE=  False
DUELING=  False
PRIORITIZED=  False

#The following hyperparameters can be adjusted
MAX_EPISODES = 10000     
ZEROS_EPSILONS_NUMBER = 30   #
RENDER_NUMBER = 1  
MAX_EPSILON=1   
BATCH_SIZE = 32  
LR = 0.001                   
GAMMA = 0.9                 
TARGET_REPLACE_ITER = 100   
MEMORY_CAPACITY = 10000 
HIDDEN_NUMBER=128 
env = Multimodal_characteristics_Marine_environment.PuckWorldEnv() 
N_ACTIONS = env.action_space.n
N_OBSERVATION = env.observation_space.shape[0]
'''
puckworld_n_observation = 9,
self.observation_space = spaces.Box(self.low, self.high)
'''

ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

#record
total_reward,mean_reward,steps,episode,mean_Loss,trainingtime,total_Loss,total_collision,mean_collision = [],[],[],[],[],[],[],[],[]


class SumTree(object):#sumtree structure(s,a,r,s_)

    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1) 
        self.data = np.zeros(capacity, dtype=object)  # for all transitions  


    #update the value and priority
    def add(self, p, data): #
        tree_idx = self.data_pointer + self.capacity - 1 

        self.data[self.data_pointer] = data  
        self.update(tree_idx, p)  

        self.data_pointer += 1  
        if self.data_pointer >= self.capacity:  
            self.data_pointer = 0

    #update the priority of the whole sumtree
    def update(self, tree_idx, p):  
        change = p - self.tree[tree_idx]  
        #update the priority
        self.tree[tree_idx] = p 
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            # self.tree[tree_idx] += change
            self.tree[tree_idx] = np.add(self.tree[tree_idx], change)

    def get_leaf(self, v): 
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1        
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):       
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx 
                else: 
                    v -= self.tree[cl_idx] 
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]


    def total_p(self):#total probility
        return self.tree[0]  # the root


class Memory(object):  #memory space
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  #importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001 
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):#
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  

    def sample(self, n):
        b_idx = np.empty((n, 1), dtype=np.int32)
        b_memory = np.empty((n, self.tree.data[0].size)) 
        ISWeights = np.empty((n, 1))
        #importance sampling weights
        segment = self.tree.total_p() / n  
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p()
        
       
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            lower_bound = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(lower_bound)
            prob = p / self.tree.total_p() 
            ISWeights[i,0]=np.power(prob/min_prob,-self.beta) 
            b_idx[i],b_memory[i,:]=idx,data

        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors): 
        abs_errors += self.epsilon  
        clipped_errors = np.minimum(abs_errors.data, self.abs_err_upper) 
        ps = np.power(clipped_errors, self.alpha) #pi^α
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p) 
            


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_OBSERVATION, HIDDEN_NUMBER) #input: N_OBSERVATION
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(HIDDEN_NUMBER, N_ACTIONS) #output: N_ACTIONS
        self.out.weight.data.normal_(0, 0.1)   # initialization

        ##Dueling use
        self.fc2 = nn.Linear(N_OBSERVATION-4, HIDDEN_NUMBER)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out2 = nn.Linear(HIDDEN_NUMBER, 1)
        self.out2.weight.data.normal_(0, 0.1)
        
        self.fc3 = nn.Linear(4, HIDDEN_NUMBER)
        self.fc3.weight.data.normal_(0, 0.1)
        self.out3 = nn.Linear(HIDDEN_NUMBER, 1)
        self.out3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        if DUELING:
            x = self.fc1(x)
            x = F.relu(x)     #Relu
            #Multimodal_environmental_fusion_and_learn
            x2 = self.fc2(x[:, :5])
            x2 = F.relu(x2)     #Relu
            x3 = self.fc3(x[:, -4:])
            x3 = F.relu(x3)     #Relu

            a=0.8#weight1, geographic characteristic
            b=0.2#weight2, meterorological characteristic 
            #The weights are adjusted based on different environment information and task requirements

            A = self.out(x)
            V = self.out2(x2)
            V_new = self.out3(x3)
            V_right = V * a + V_new * b
            actions_value = V_right.expand_as(A) + (A - torch.mean(A, dim=1,keepdim=True).expand_as(A)) 
        else:
            x = self.fc1(x)
            x = F.relu(x)
            actions_value = self.out(x)
        return actions_value 


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().to(device), Net().to(device)   #evaluate network
        if OLD_NET:
            self.eval_net.load_state_dict(torch.load('eval_net_params2_DQN_path1_x9522_y14992.pkl'))
            self.target_net.load_state_dict(torch.load('eval_net_params2_DQN_path1_x9522_y14992.pkl'))

        self.learn_step_counter = 0               # for target updating
        self.memory_counter = 0                   # for storing memory
        if PRIORITIZED:
            self.memory = Memory(capacity=MEMORY_CAPACITY)
        else:
            self.memory = np.zeros((MEMORY_CAPACITY, N_OBSERVATION * 2 + 2))
            '''self.memory
                s(N_OBSERVATION)  s_(N_OBSERVATION)  a   r
            0
            1
            2
            3
            ...
            50000
            
            '''
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()#先定义个损失函数


    def choose_action(self, x, i_episode):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        '''
        torch.FloatTensor() 
        torch.unsqueeze(tensor) 
        [1,2,3] ->  [[1,2,3]]
        '''
        if i_episode >= MAX_EPISODES*0.3:
            epsilon=0
        else:
            epsilon = 0.01 + (1 - 0.01) * np.exp(-0.002 * i_episode)
        if np.random.uniform() < epsilon:   # random 
            action = np.random.randint(0, N_ACTIONS) 
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)

        else:   # greedy  
            x = x.to(device)
            actions_value = self.eval_net.forward(x)  
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        return action,epsilon

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        if PRIORITIZED:
            self.memory.store(transition)
            self.memory_counter += 1
        else:
            # replace the old memory with new memory
            index = self.memory_counter % MEMORY_CAPACITY
            self.memory[index, :] = transition
            self.memory_counter += 1


    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())#eval_net -> target_net
        self.learn_step_counter += 1

        # sample batch transitions
        if PRIORITIZED:
            tree_idx, batch_memory, ISWeights = self.memory.sample(BATCH_SIZE)
            b_memory = batch_memory
        else:
            if self.memory_counter > MEMORY_CAPACITY:
                sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)#From the experience playback area, randomly select batchsize samples
            else: 
                sample_index = np.random.choice(self.memory_counter, BATCH_SIZE)
            b_memory = self.memory[sample_index, :]#s,a,r,s_
        #s,a,r,s_ form b memeory
        b_s = torch.FloatTensor(b_memory[:, :N_OBSERVATION]).to(device)#s（32*4）
        b_a = torch.LongTensor(b_memory[:, N_OBSERVATION:N_OBSERVATION+1].astype(int)).to(device)#a（32*1）
        b_r = torch.FloatTensor(b_memory[:, N_OBSERVATION+1:N_OBSERVATION+2]).to(device)#r（32*1）
        b_s_ = torch.FloatTensor(b_memory[:, -N_OBSERVATION:]).to(device)#s_（32*4）
        q_eval = self.eval_net(b_s).gather(1, b_a)  
        
        ##Double DQN     
        if DOUBLE:
            q_eval_next=self.eval_net(b_s_)
            max_action=torch.unsqueeze(torch.max(q_eval_next,1)[1],1)
            q_next = self.target_net.forward(b_s_).gather(1, max_action)
            q_target = b_r + GAMMA * q_next#Q
        else:
            q_next = self.target_net(b_s_).detach()  
            q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  
        if PRIORITIZED:
            abs_errors = torch.sum(torch.abs(q_target - q_eval), dim=1)#abs_errors |δ|
            ISWeights = torch.Tensor(ISWeights).to(device)
            loss = torch.mean(torch.mean(ISWeights* (q_target - q_eval)**2,dim=1))
            self.memory.batch_update(tree_idx, abs_errors.cpu())                  
        else:
            loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()  
        loss.backward()    
        self.optimizer.step()       


        return loss


#Save the RL training results-------------------------------------------
def Save_results(): 
    episodeVSsteps = np.vstack((episode, steps))
    episodeVSmean_reward = np.vstack((episode, mean_reward))
    episodeVStotal_reward = np.vstack((episode, total_reward))
    episodeVSmean_Loss = np.vstack((episode, mean_Loss))
    episodeVStotal_Loss = np.vstack((episode, total_Loss))
    episodeVSmean_collision = np.vstack((episode,mean_collision))
    episodeVStotal_collision = np.vstack((episode,total_collision))

    output_folder = r'Path planning module\Experiments for Testing Generalization Ability' ## Replace it with your folder path
    os.makedirs(output_folder, exist_ok=True)

    file1_path = os.path.join(output_folder, 'episode-steps(fusion DQN).pickle')
    with open(file1_path, 'wb') as file1:
        pickle.dump(episodeVSsteps, file1)
        
    file2_path = os.path.join(output_folder, 'episode-mean_reward(fusion DQN).pickle')
    with open(file2_path, 'wb') as file2:
        pickle.dump(episodeVSmean_reward, file2)

    file3_path = os.path.join(output_folder, 'episode-total_reward(fusion DQN).pickle')
    with open(file3_path, 'wb') as file3:
        pickle.dump(episodeVStotal_reward, file3)

    file4_path = os.path.join(output_folder, 'episode-mean_Loss(fusion DQN).pickle')
    with open(file4_path, 'wb') as file4:
        pickle.dump(episodeVSmean_Loss, file4)

    file5_path = os.path.join(output_folder, 'episode-total_Loss(fusion DQN).pickle')
    with open(file5_path, 'wb') as file5:
        pickle.dump(episodeVStotal_Loss, file5)

    file6_path = os.path.join(output_folder, 'episode-mean_collision(fusion DQN).pickle')
    with open(file6_path, 'wb') as file6:
        pickle.dump(episodeVSmean_collision, file6)
        
    file7_path = os.path.join(output_folder, 'episode-total_collision(fusion DQN).pickle')
    with open(file7_path, 'wb') as file7:
        pickle.dump(episodeVStotal_collision, file7)


    file1.close()
    file2.close()
    file3.close()
    file4.close()
    file5.close()
    file6.close()
    file7.close()


    
    





def main():
    dqn = DQN()
    print('\nlearning...')
    for i_episode in range(1,MAX_EPISODES+1):
        s = env.reset()
        episode_epsilon,episode_loss,episode_step, episode_reward, episode_collision=0,0,0,0,0
        x=[]
        y=[]
        angle=[]
        starttime = time.time()
        while True:
            
            if RENDER == True:
                if LAST_RENDER:
                    if i_episode>=MAX_EPISODES-RENDER_NUMBER+1:
                        env.render(0,0) 
                else:
                    env.render(0,0)
            # take action   
            a,episode_epsilon = dqn.choose_action(s, i_episode)#Epsilon-Greedy strategy
            s_, r, done, c = env.step(a) 
            dqn.store_transition(s, a, r, s_)#store
            #record
            episode_reward += r
            episode_step += 1
            episode_collision += c
            x.append(s[0])
            y.append(s[1])
            angle.append(s[2])
            if dqn.memory_counter > MEMORY_CAPACITY:
                episode_loss += dqn.learn().item()  
            if done or episode_step >= 200:
                break
            s = s_
        endtime = time.time()


        #------------------------------trainning process(1/3--2/3--3/3)--------------------------------------------------
        if i_episode == 8000 :
            path_br_x_3 = x[:]   
            path_br_y_3 = y[:]
            path_br_angle_3 = angle[:]
            data_3 = {'X_3': path_br_x_3, 'Y_3': path_br_y_3, 'angle_3': path_br_angle_3}
            df_3 = pd.DataFrame(data_3)
            folder_path_3 = r'Path planning module\Experiments for Testing Generalization Ability'  #replace it with your file path
            excel_filename_3 = 'fusionDQN_trainning_process_third_stage.xlsx'
            excel_path_3 = os.path.join(folder_path_3, excel_filename_3)
            df_3.to_excel(excel_path_3, index=False)

        if i_episode == 5000 :
            path_br_x_2 = x[:]   
            path_br_y_2 = y[:]
            path_br_angle_2 = angle[:]
            data_2 = {'X_2': path_br_x_2, 'Y_2': path_br_y_2, 'angle_2': path_br_angle_2}
            df_2 = pd.DataFrame(data_2)
            folder_path_2 = r'Path planning module\Experiments for Testing Generalization Ability'  # Replace it with your folder path
            excel_filename_2 = 'fusionDQN_trainning_process_second_stage.xlsx'
            excel_path_2 = os.path.join(folder_path_2, excel_filename_2)
            df_2.to_excel(excel_path_2, index=False)
        
        if i_episode == 3000 :
            path_br_x_1 = x[:]   
            path_br_y_1 = y[:]
            path_br_angle_1 = angle[:]
            data_1 = {'X_1': path_br_x_1, 'Y_1': path_br_y_1, 'angle_1': path_br_angle_1}
            df_1 = pd.DataFrame(data_1)
            folder_path_1 = r'Path planning module\Experiments for Testing Generalization Ability'  # Replace it with your folder path
            excel_filename_1 = 'fusionDQN_trainning_process_first_stage.xlsx'
            excel_path_1 = os.path.join(folder_path_1, excel_filename_1)
            df_1.to_excel(excel_path_1, index=False)
        

            

        print(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
        print("episode:{0},steps:{1},m_reward:{2:3.5f},t_reward:{3:3.5f},m_loss:{4:3.5f},t_loss:{5:3.5f},epsilon:{6:3.5f},t_collision:{7:3.1f},m_collsion:{8:3.5f}".
              format(i_episode,episode_step,episode_reward/episode_step,episode_reward,episode_loss/episode_step,episode_loss,episode_epsilon,episode_collision,episode_collision/episode_step))
        dtime = endtime - starttime
        print("Run time of one epoch:%.8s s" % dtime)  
        episode.append(i_episode)
        steps.append(episode_step)
        mean_reward.append(episode_reward / episode_step)
        total_reward.append(episode_reward)
        mean_Loss.append(episode_loss/episode_step)
        total_Loss.append(episode_loss)
        total_collision.append(episode_collision)
        mean_collision.append(episode_collision/episode_step)
        max_value = max(total_reward)
        max_index = total_reward.index(max_value)
        print("The rounds with the highest awards:", max_index, "The corresponding reward value:", max_value, "The corresponding step:", steps[max_index])
        if i_episode % 500 == 0:
            Save_results()
    #torch.save(dqn.eval_net, r'eval_net.pkl')  # save the total network
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # torch.save(dqn.eval_net.state_dict(), save_path)  # only save the parameters
    
    endtime = time.time()
    dtime = endtime - starttime
    print("Run time of the program: %.8s s" % dtime)  


    return
main()



