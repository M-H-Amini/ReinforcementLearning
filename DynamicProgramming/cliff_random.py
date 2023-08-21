#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein  Amini                              ##
##                                 Title: Cliff - Random                               ##
##                                   Date: 2023/08/21                                  ##
##                                                                                     ##
#########################################################################################

##  Description: Random movements in the Cliff Walking environment




import gymnasium as gym
env = gym.make('CliffWalking-v0', render_mode='human')
observation, info = env.reset()

action_dict = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}

for _ in range(10):
    action = env.action_space.sample()
    print(f'Observation: {observation}, Action: {action_dict[action]}')
    observation, reward, terminated, truncated, info = env.step(action)
    print(f'Observation: {observation}')
    print(f'Reward: {reward}')
    print(f'Terminated: {terminated}')
    print(f'Truncated: {truncated}')
    print(f'Info: {info}')
    print()
    if terminated or truncated:
        observation, info = env.reset()

env.close()
