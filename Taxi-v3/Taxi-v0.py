import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output


MAX_EPISODE_STEPS = 200  
env = gym.make('Taxi-v3')

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"timestep: {i + 1} ")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        time.sleep(0.05)


class Agent():

    def __init__(self):
        pass

    def Random(self):
        env.s = 328

        epoches = 0
        penalties, reward = 0, 0

        frames = []

        done = False

        while not done:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
            })
            epoches += 1

        print("Timesteps to solve{}".format(epoches))
        print("Penalties incurred {}".format(penalties))
        print(frames[-1]['frame'])

    def Train(self):
        alpha = 0.85
        gamma = 0.70
        q_table = np.zeros([env.observation_space.n, env.action_space.n])

        for i in range(1, 100001):

            state = env.reset()
            epsilon = 0.11 - ((i) / 1000000)
            epochs, penalties, reward, = 0, 0, 0
            done = False

            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()  # Explore action space
                else:
                    action = np.argmax(q_table[state])  # Exploit learned values

                next_state, reward, done, info = env.step(action)

                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])

                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state, action] = new_value

                if reward == -10:
                    penalties += 1

                state = next_state
                epochs += 1

            if i % 1000 == 0:
                clear_output(wait=True)
                time.sleep(0.001)
                print(f"Episode: {i}")

        print("Training finished.\n")
        np.save('q_table', q_table)



    def Play(self):
        q_table = np.load('q_table.npy')
        total_epochs, total_penalties = 0, 0
        episodes = 100

        frames = []

        for i in range(episodes):
            state = env.reset()
            epochs, penalties, reward = 0, 0, 0

            done = False

            while not done:
                action = np.argmax(q_table[state])
                state, reward, done, info = env.step(action)
                frames.append(
                    {
                        'frame': env.render(mode='ansi'),
                        'action': action,
                        'state': state,
                        'reward': reward

                    }
                )
                if reward == -10:
                    penalties += 1

                epochs += 1
            if i % 10 == 0:
                print_frames(frames)
                print('episode:', i)
                time.sleep(1)
            frames = []
            total_penalties += penalties
            total_epochs += epochs
        return total_epochs, total_penalties

    def Test(self):

        START_TIME = time.time()
        self.Train()
        TOTAL_TIME = time.time() - START_TIME
        return self.Play(), TOTAL_TIME


if __name__ == '__main__':
    env.reset()
    env.render()
    print("Action space {}".format(env.action_space))
    print("State space {}".format(env.observation_space))

    agent = Agent()
    a, t = agent.Test()
    print(t)
