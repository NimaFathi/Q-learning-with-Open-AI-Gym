import time
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Pendulum-v0')
MAX_EPISODE_STEPS = 200  ### Don't change this for testing your code and Play() method. ###


class Agent():

    def __init__(self):
        pass

    def Train(self):
        alpha = 0.2
        max_alpha = 0.68
        min_alpha = 0.25
        gamma = 0.9
        min_epsilon = 0.10
        epsilon = 0.7
        s_time = time.time()
        arr = [(env.observation_space.high[0] - env.observation_space.low[0]), 1,
               env.observation_space.high[2] - env.observation_space.low[2]]

        num_states = (arr) * np.array([10, 1, 1])
        num_states = np.round(num_states, 0).astype(int) + 1
        print(num_states)
        Q = np.random.uniform(low=-1, high=1, size=(num_states[0], num_states[1], num_states[2], 30))
        reward_list = []
        ave_reward_list = []

        i = 0
        fav = False
        while time.time() - s_time < 900:
            done = False
            tot_reward, reward = 0, 0
            increase = (max_alpha - alpha) / 15000
            decrease = (min_alpha - alpha) / 10000
            if alpha >= 0.55:
                fav = True
            reduction = (epsilon - min_epsilon) / 20000
            state = env.reset()
            i += 1
            sign = 0
            if state[1] > 0:
                sign = 1

            s = [state[0] - env.observation_space.low[0], sign, state[2] - env.observation_space.low[2]]
            state_adj = (s) * np.array([10, 1, 1])
            state_adj = np.round(state_adj, 0).astype(int)

            while not done:
                epsilon = 0.97 * epsilon
                if np.random.random() < 1 - epsilon:
                    action_arg = np.argmax(Q[state_adj[0], state_adj[1], state_adj[2]])
                else:
                    action_arg = np.random.randint(0, 30)
                a = round(-2 + (2 / 15) * action_arg, 2)
                next_state, reward, done, _ = env.step([a])
                next_sign = 0
                if next_state[1] > 0:
                    next_sign = 1
                ns = [next_state[0] - env.observation_space.low[0], next_sign,
                      next_state[2] - env.observation_space.low[2]]
                next_state_adj = (ns) * np.array([10, 1, 1])
                next_state_adj = np.round(next_state_adj, 0).astype(int)
                if done:
                    Q[state_adj[0], state_adj[1], state_adj[2], action_arg] = reward
                else:
                    delta = alpha * (
                                reward + gamma * np.max(Q[next_state_adj[0], next_state_adj[1], next_state_adj[2]]) - Q[
                            state_adj[0], state_adj[1], state_adj[2], action_arg])
                    Q[state_adj[0], state_adj[1], state_adj[2], action_arg] += delta
                tot_reward += reward
                state_adj = next_state_adj

                if epsilon > min_epsilon:
                    epsilon -= reduction
                if fav == False:
                    alpha += increase
                else:
                    alpha += decrease
            reward_list.append(tot_reward)

            if i % 500 == 0:
                print("Episode: {}".format(i))
                ave_reward = np.mean(reward_list)
                ave_reward_list.append(ave_reward)
                reward_list = []
                print("Average Reward:", ave_reward)

        print("Training finished")
        env.close()
        np.save('q_table', Q)

    def Play(self, render=False):
        q = np.load('q_table.npy')
        step_count = 0
        scores = []
        done = None
        avg_x_list = []
        avg_score_list = []

        for episode_count in range(1000):
            episode_count += 1
            print('******Episode ', episode_count)
            state = env.reset()
            score = 0
            done = False
            print('step_count:', step_count)
            step_count = 0
            while not done and step_count < MAX_EPISODE_STEPS:
                sign = 0
                if state[1] > 0:
                    sign = 1
                s = [state[0] - env.observation_space.low[0], sign, state[2] - env.observation_space.low[2]]
                state_adj = (s) * np.array([10, 1, 1])
                state_adj = np.round(state_adj, 0).astype(int)
                action_arg = np.argmax(q[state_adj[0], state_adj[1], state_adj[2]])

                a = round(-2 + (2 / 15) * action_arg, 2)
                state, reward, done, _ = env.step([a])
                step_count += 1
                score += reward
                if render:
                    env.render()
                    time.sleep(0.04)  
            scores.append(score)
            print('Score:', score)
        print("Average score over 1000 run : ", np.array(scores).mean())
        for i in range(1, 11):
            avg_score_list.append(np.mean(scores[100 * (i - 1):100 * i]))
            avg_x_list.append(100 * i)
        plt.figure()
        plt.plot(avg_x_list, avg_score_list)
        plt.title("Average score per each 100 episodes")
        plt.show()
        print(avg_score_list)

        return scores, np.array(scores).mean()

    def Test(self):
        
        START_TIME = time.time()  
        self.Train()
        TOTAL_TIME = time.time() - START_TIME
        return self.Play(), TOTAL_TIME


agent = Agent()
a, t = agent.Test()
print(t)
