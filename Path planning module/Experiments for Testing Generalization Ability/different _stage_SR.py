import numpy as np
import pickle

# replace it with your trainning file path
file_path = r'Path planning module\Experiments for Testing Generalization Ability\episode-steps(fusion DQN)_x9522_y14992.pickle'  # 替换为你的文件路径
with open(file_path, 'rb') as file:
    episodeVSsteps = pickle.load(file)
    

# 0 epoch ~ 1667 epoch
start_episode_part1 = 0
end_episode_part1 = 1667
num_episodes_to_average = 1667
steps_data_part1 = episodeVSsteps[1][start_episode_part1:end_episode_part1]
steps_less_than_15_part1 = steps_data_part1[steps_data_part1 < 15]
steps_less_than_15_ratio_part1 = len(steps_less_than_15_part1) / num_episodes_to_average
print(f'{start_episode_part1}epoch to {end_episode_part1}epoch: {steps_less_than_15_ratio_part1 * 100}%')


# first stage
start_episode_1 = 0
end_episode_1 = 3333
num_episodes_to_average = 3333
steps_data_1 = episodeVSsteps[1][start_episode_1:end_episode_1]
steps_less_than_15_1 = steps_data_1[steps_data_1 < 15]
steps_less_than_15_ratio_1 = len(steps_less_than_15_1) / num_episodes_to_average
print(f'{start_episode_1}epoch to {end_episode_1}epoch: {steps_less_than_15_ratio_1 * 100}%')

# 1667 epoch ~ 5001 epoch
start_episode_part2 = 1667
end_episode_part2 = 5001
num_episodes_to_average = 3333
steps_data_part2 = episodeVSsteps[1][start_episode_part2:end_episode_part2]
steps_less_than_15_part2 = steps_data_part2[steps_data_part2 < 15]
steps_less_than_15_ratio_part2 = len(steps_less_than_15_part2) / num_episodes_to_average
print(f'{start_episode_part2}epoch to {end_episode_part2}epoch: {steps_less_than_15_ratio_part2 * 100}%')

# second stage
start_episode_2 = 3334
end_episode_2 = 6666
num_episodes_to_average = 3333
steps_data_2 = episodeVSsteps[1][start_episode_2:end_episode_2]
steps_less_than_15_2 = steps_data_2[steps_data_2 < 15]
steps_less_than_15_ratio_2 = len(steps_less_than_15_2) / num_episodes_to_average
print(f'{start_episode_2}epoch to {end_episode_2}epoch: {steps_less_than_15_ratio_2 * 100}%')

# 5001 epoch ~ 8335 epoch
start_episode_part3 = 5001
end_episode_part3 = 8335
num_episodes_to_average = 3334
steps_data_part3 = episodeVSsteps[1][start_episode_part3:end_episode_part3]
steps_less_than_15_part3 = steps_data_part3[steps_data_part3 < 15]
steps_less_than_15_ratio_part3 = len(steps_less_than_15_part3) / num_episodes_to_average
print(f'{start_episode_part3}epoch to {end_episode_part3}epoch: {steps_less_than_15_ratio_part3 * 100}%')

# third stage
start_episode_3 = 6667
end_episode_3 = 10000
num_episodes_to_average = 3334
steps_data_3 = episodeVSsteps[1][start_episode_3:end_episode_3]
steps_less_than_15_3 = steps_data_3[steps_data_3 < 15]
steps_less_than_15_ratio_3 = len(steps_less_than_15_3) / num_episodes_to_average
print(f'{start_episode_3}epoch to {end_episode_3}epoch: {steps_less_than_15_ratio_3 * 100}%')

