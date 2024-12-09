import numpy as np
import pickle

# replace it with your trainning file path
file_path = r'Path planning module\Experiments for Testing Generalization Ability\episode-total_collision(fusion DQN)_x9522_y14992.pickle'  # 替换为你的文件路径
with open(file_path, 'rb') as file:
    episodeVStotal_collision = pickle.load(file)


# 0 epoch ~ 1667 epoch
start_episode_part1 = 0
end_episode_part1 = 1667
collision_counts_part1 = episodeVStotal_collision[1][start_episode_part1:end_episode_part1]
non_zero_collision_counts_part1 = collision_counts_part1[collision_counts_part1 != 0]
non_zero_collision_ratio_part1 = len(non_zero_collision_counts_part1) / 1667
print(f'0~1667:{non_zero_collision_ratio_part1 * 100}%')

# 0 epoch ~ 3334 epoch
start_episode_1 = 0
end_episode_1 = 3334
collision_counts_1 = episodeVStotal_collision[1][start_episode_1:end_episode_1]
non_zero_collision_counts_1 = collision_counts_1[collision_counts_1 != 0]
non_zero_collision_ratio_1 = len(non_zero_collision_counts_1) / 3334
print(f'first stage:{non_zero_collision_ratio_1 * 100}%')

# 1667 epoch ~ 5001 epoch
start_episode_part2 = 1667
end_episode_part2 = 5001
collision_counts_part2 = episodeVStotal_collision[1][start_episode_part2:end_episode_part2]
non_zero_collision_counts_part2 = collision_counts_part2[collision_counts_part2 != 0]
non_zero_collision_ratio_part2 = len(non_zero_collision_counts_part2) / 3334
print(f'1667~5001:{non_zero_collision_ratio_part2 * 100}%')

# 3334 epoch ~ 6667 epoch
start_episode_2 = 3334
end_episode_2 = 6667
collision_counts_2 = episodeVStotal_collision[1][start_episode_2:end_episode_2]
non_zero_collision_counts_2 = collision_counts_2[collision_counts_2 != 0]
non_zero_collision_ratio_2 = len(non_zero_collision_counts_2) / 3334
print(f'second stage:{non_zero_collision_ratio_2 * 100}%')

# 5001 epoch ~ 8335 epoch
start_episode_part3 = 5001
end_episode_part3 = 8335
collision_counts_part3 = episodeVStotal_collision[1][start_episode_part3:end_episode_part3]
non_zero_collision_counts_part3 = collision_counts_part3[collision_counts_part3 != 0]
non_zero_collision_ratio_part3 = len(non_zero_collision_counts_part3) / 3334
print(f'5001~8335:{non_zero_collision_ratio_part3 * 100}%')

# 6667 epoch ~ 10000 epoch
start_episode_3 = 6667
end_episode_3 = 10000
collision_counts_3 = episodeVStotal_collision[1][start_episode_3:end_episode_3]
non_zero_collision_counts_3 = collision_counts_3[collision_counts_3 != 0]
non_zero_collision_ratio_3 = len(non_zero_collision_counts_3) / 3334
print(f'third:{non_zero_collision_ratio_3 * 100}%')




