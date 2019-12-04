# Q-learning-with-Open-AI-Gym
Q-learning on Taxi-v2 and Pendulum-v0 environments.
In this Repo i'm my purpose was to dive into Reinforcment Learning.<br>
One of the easiest and first algorithms in this profession is Q-learning.<br>
Q-learning is a algorihtm that works best for problems with discrete State and action.<br>
To work with Reinforcment Learning I use [Open AI Gym](https://gym.openai.com) that is a library with a lot of perfect game environments that you can use to see how your code works.<br>
you can see all possible environments in <b>[Here](https://gym.openai.com/envs/#classic_control)<b>.
# 1- Taxi-v2:
Taxi-v2 is a problem with discrete state and action space so that is perfect match for Q-learning.
The point in here is we want to pick up passenger in one of 4 marked cells and deliver him/her to one of 4 marked destinations.<br>
State space is roughly 500 and action space is 6.
# 2- Pendulum-v0:
this environments is a little different with both state and action space continuous.<br>
To use Q-learning one of ideas is to define limited number of states and actions and change or spaces into some imaginary discrete one and the use Q-learning.
