# Report Project 2: Continuous Control

### Introduction
In this project we implemented the [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf) paper (Lillicrap et al)[1], that present a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces known as Deep Deterministic Policy Gradients (DDPG).

The DDPG implementation was used to solve the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment[2], an environment that provides a double-jointed arm that can move to target locations. For this specific environment a total of 20 double-jointed arm was provided.

### Learning Algorithm

For the solution we have implemented the Deep Deterministic Policy Gradients (DDPG), DDPG is an Actor-Critic algorithm that make use of two neural networks, one for the Actor and one for the Critic. These networks compute action predictions for the current state and generate a temporal-difference (TD) error signal at each time step. The input of the Actor network is the current state, and using a policy function provides the output, a single real value representing a chosen action (Deterministic Policy). The other neural network the Critic, is used to criticize the actions made by the Actor to the temporal-difference (TD) error.

But, deterministic policy gradient might not explore the full state and action space, to mitigate this challenge a number of techniques are applied to this implementation including Soft Target Update[7] through twin local / target network, a Replay Buffer, and adding noise for action explorations.

#### Replay Buffer
Replay Buffer is where it allows the DDPG agent to learn on both the current experience and past experiences, by sampling experiences from the Replay Buffer across a set of unrelated experiences[4]. in the this implementation it is done by randomly sampling the stored tuples in the Replay Buffer for training the model for each training step.

#### OU Noise: 
Noise helps algorithms explore their environments more effectively, leading to higher scores and more elegant behaviors, action noise promotes exploration. In this solution we have used a Ornstein-Uhlenbeck Random Process as noise [6]

#### Soft Update through twin local / target network
The Deepmind team came up with the use of a target network, where we created a copy of the actor and critic networks respectively, that are used for calculating the target values. The weights of these target networks are then updated by having them slowly track the learned networks. The target values are constrained to change slowly, greatly improving the stability of learning.[8]

This implementation performs a soft update of model parameters based on the TAU hyperparameter. The local model is updated based on TAU while the target model is updated based on 1.0 – TAU. 

#### Model Architecture
Two Neural Network models are used in the DDPG algorithm. An Actor and a Critic network both with a Local and Target network each.

The Actor Architecture:

 The model has 2 fully connected layers
 The first layer takes in the state passes it through 256 nodes with Relu activation
 The second layer take the output from the first layer and passes through 256 nodes and outputs a single real value representing an action chosen from a continuous action space
 Adam optimizer is used with a weight decay of 0.0001.

The Critic Architecture:

 The model has 4 fully connected layers
 The first layer takes the state and passes through 256 nodes with Relu activation
 Then we take the output from the first layer and concatenate it with the action size
 We then forward this to a second and third layer which passes it through 256 nodes with Relu activation
 Finally the fourth layer passes it through 256 and outputs is a estimated Q-value of the current state and of the action given by the actor
 Adam optimizer is used with a weight decay of 0.0001.


#### Implementation
 Our DDPG implementation are spilt into two files, based on the [solution](https://github.com/udacity/deep-reinforcement-learning/blob/55474449a112fa72323f484c4b7a498c8dc84be1/ddpg-bipedal) of the bipedal environment made by [Udacity](https://www.udacity.com) for the [Deep Reinforcement Learning NanoDegree](https://eu.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893):

    ddpg_agent.py - contains the implementation of the agent, OU Noise, Replay Buffer and hyperparameters
    model.py - Contains the implementation of the Actor and Critic

Hyperparameters can be adjusted, by default the following has been used:

    BUFFER_SIZE = int(1e6)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR_ACTOR = 1e-4         # learning rate of the actor 
    LR_CRITIC = 3e-4        # learning rate of the critic
    WEIGHT_DECAY = 0.0001   # L2 weight decay

### Results and Plot of Rewards
By executing the instructions in `Continuous_Control.ipynb` we were able to get the agent to be able to solve the task, of getting an average score of at least +30 in last 100 episodes, in 202 episodes with an average score of 30.21.

here is graph of increase of score over time (number of episodes):

![Training Scores](plot.png)


### Ideas for future work
The linearity of the improvements of scores over time gives an idea that it should be able to improve the training of the agent. Some ideas can be:

- Hyperparameters tunning
- Channing the Actor and Critics neural network architecture

Probably those can give some smaller improvement of performance. Instead a better option should be consider an alternative algorithms 

- Implement another algorithm that is more suitable to the environment of having 20 double-jointed arm to train in parallel as in the case of like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) 
- Try to implement Parameter Noise instead of Action noise, OpenAI has experimented with this and seems to give better performance than action noise[5]

### References

[1][Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf) paper (Lillicrap et al)

[2][Unity Reacher Environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)

[3][Introduction to Various Reinforcement Learning Algorithms](https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287)

[4][Solving Continuous Control environment using Deep Deterministic Policy Gradient](https://medium.com/@kinwo/solving-continuous-control-environment-using-deep-deterministic-policy-gradient-ddpg-agent-5e94f82f366d)

[5][Better Exploration with Parameter Noise](https://blog.openai.com/better-exploration-with-parameter-noise/)

[6][Ornstein–Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)

[7][Taming the Noise in Reinforcement Learning via Soft Updates](https://arxiv.org/pdf/1512.08562.pdf)

[8][Using Keras and Deep Deterministic Policy Gradient to play TORCS] (https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html)