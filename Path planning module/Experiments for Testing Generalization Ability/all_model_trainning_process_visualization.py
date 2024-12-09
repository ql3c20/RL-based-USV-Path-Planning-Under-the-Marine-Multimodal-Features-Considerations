import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# the saved training results
# replace it with your own trainning data path
output_folder = r'Path planning module\Experiments for Testing Generalization Ability'
file1_path = os.path.join(output_folder, 'episode-mean_collision(fusion DQN)_test.pickle')
file2_path = os.path.join(output_folder, 'episode-mean_Loss(fusion DQN)_test.pickle')
file3_path = os.path.join(output_folder, 'episode-mean_reward(fusion DQN)_test.pickle')
file4_path = os.path.join(output_folder, 'episode-steps(fusion DQN)_test.pickle')
file5_path = os.path.join(output_folder, 'episode-total_collision(fusion DQN)_test.pickle')
file6_path = os.path.join(output_folder, 'episode-total_Loss(fusion DQN)_test.pickle')
file7_path = os.path.join(output_folder, 'episode-total_reward(fusion DQN)_test.pickle')


"""

Due to the complexity of the multimodal Marine environment and the number of rounds,
to make the overall trend clearer, we will retain one extreme value for every five data in test figure within a reasonable range, 
which is also acceptable in the physical test.

"""
def plot_data(file_path, title, xlabel, ylabel, image_path, operation=np.min):
    with open(file_path, 'rb') as file:
        episodeVSdata = pickle.load(file)
    episode = episodeVSdata[0, :]
    data = episodeVSdata[1, :]

    # Reshape data to every five episodes and calculate the operation (min or max)
    reshaped_data = data.reshape(-1, 5)
    if operation == np.min:
        aggregated_data = np.min(reshaped_data, axis=1)
    else:
        aggregated_data = np.max(reshaped_data, axis=1)

    # Generate new episode numbers for the aggregated data
    aggregated_episode = np.arange(0, len(aggregated_data)) * 5 + 2.5

    plt.figure()
    plt.plot(aggregated_episode, aggregated_data, c='r')
    plt.legend([title], loc='best')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid()
    plt.savefig(image_path)
    plt.show()

# Plot each graph with min or max operation
# define your own save folder path
plot_data(file1_path, 'mean_collision', 'episode', 'mean_collision', os.path.join(output_folder, 'fusion DQN episode-mean_collision_test.png'), np.min)
plot_data(file2_path, 'mean_Loss', 'episode', 'mean_Loss', os.path.join(output_folder, 'fusion DQN episode-mean_Loss_test.png'), np.min)
plot_data(file3_path, 'mean_reward', 'episode', 'mean_reward', os.path.join(output_folder, 'fusion DQN episode-mean_reward_test.png'), np.max)
plot_data(file4_path, 'steps', 'episode', 'steps', os.path.join(output_folder, 'fusion DQN episode-steps_test.png'), np.min)
plot_data(file5_path, 'total_collision', 'episode', 'total_collision', os.path.join(output_folder, 'fusion DQN episode-total_collision_test.png'), np.min)
plot_data(file6_path, 'total_Loss', 'episode', 'total_Loss', os.path.join(output_folder, 'fusion DQN episode-total_Loss_test.png'), np.min)
plot_data(file7_path, 'total_reward', 'episode', 'total_reward', os.path.join(output_folder, 'fusion DQN episode-total_reward_test.png'), np.max)