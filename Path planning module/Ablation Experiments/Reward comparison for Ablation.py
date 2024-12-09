import matplotlib.pyplot as plt
import pickle
import numpy as np
fig = plt.figure(figsize=(12, 6))


#No Mutimodal
# replace your file path
with open(r'Path planning module\Ablation Experiments\episode-mean_total_reward(No Multimodal)_x9522_y14992.pickle', 'rb') as f1:
    info1 = pickle.load(f1)
x1 = np.array(info1[0])
y1 = np.array(info1[1])


#No Priority
# replace your file path
with open(r'Path planning module\Ablation Experiments\episode-mean_total_reward(No Priority)_x9522_y14992.pickle', 'rb') as f2:
    info2 = pickle.load(f2)
x2 = np.array(info2[0])
y2 = np.array(info2[1])




#No Dueling
# replace your file path
with open(r'Path planning module\Ablation Experiments\episode-mean_total_reward(No Dueling)_x9522_y14992.pickle', 'rb') as f3:
    info3 = pickle.load(f3)
x3 = np.array(info3[0])
y3 = np.array(info3[1])



#No Double
# replace your file path
with open(r'Path planning module\Ablation Experiments\episode-mean_total_reward(No Double)_x9522_y14992.pickle', 'rb') as f4:
    info4 = pickle.load(f4)
x4 = np.array(info4[0])
y4 = np.array(info4[1])


#fusion DQN
# replace your file path
with open(r'Path planning module\Ablation Experiments\episode-mean_total_reward(fusion DQN)_x9522_y14992.pickle', 'rb') as f5:
    info5 = pickle.load(f5)
x5 = np.array(info5[0])
y5 = np.array(info5[1])




plt.plot(x1, y1, c=(0/255,0/255,255/255), label='No Mutimodal', linewidth=3)
plt.plot(x2, y2, c=(0/255,127/255,0/255), label='No Priority', linewidth=3)
plt.plot(x3, y3, c=(146/255,38/255,146/255), label='No Dueling', linewidth=3)
plt.plot(x4, y4, c=(255/255,165/255,0/255),label='No Double', linewidth=3)
plt.plot(x5, y5, c=(255/255,0/255,0/255), label='fusion DQN', linewidth=3)

plt.ylim(84, 96)


plt.legend(loc='best', fontsize='16')
plt.ylabel('reward', fontsize=16)
plt.xlabel('episode', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)



plt.savefig(r'Path planning module\Ablation Experiments\figure_ablation.png')# replace your file path
plt.savefig(r'Path planning module\Ablation Experiments\figure_ablation.eps')
plt.savefig(r'Path planning module\Ablation Experiments\figure_ablation.pdf')
plt.show()



