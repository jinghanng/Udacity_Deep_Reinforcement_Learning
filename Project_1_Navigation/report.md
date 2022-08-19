# Deep Q-Networks

The [Deep Q-Network](http://files.davidqiu.com//research/nature14236.pdf) algorithm was proposed by Deepmind.

The Deep Q-Network algorithm has a model consisting of two fully connected (FC) hidden layers. The first FC hidden layer consist of 512 nodes connected to the input layers. The second FC hidden layer consist of 64 nodes connected to the output layer. Both of the FC layers have RELU activation function.The input layer consists of 37 nodes while the output layer consists of 4 nodes. The two hidden layers and the RELU activation function are chosen to achieve non-linearity.

Different number of nodes were tested for the hidden layers. The algorithm learns better when the number of nodes are higher.

The epsilon-greedy parameter chosen starts with 1.0, decays at a rate of 0.995 and ends at 0.01.

The discount factor was chosen as 0.99.

The learning rate was chosen as 0.0005.

The reward function is based on rewarding agent with a score of +1 for a yellow banana and -1 for a blue banana.

### Experience Replay

When an agent interacts with the environment, the sequence of experience tuples can be highly correlated. The naive Q-learning algorithm that learns from each of these experience tuples in sequential order runs the risk of getting swayed by the effects of this correlation. By instead keeping track of a replay buffer and using experience replay to sample from the buffer at random, we can prevent action values from oscillating or diverging catastrophically.

The replay buffer contains a collection of experience tuples (S, A, R, S′). The tuples are gradually added to the buffer as we are interacting with the environment. The tuple stored consists of state, action, reward, next_state and done.

The act of sampling a small batch of tuples from the replay buffer in order to learn is known as experience replay. In addition to breaking harmful correlations, experience replay allows us to learn more from individual tuples multiple times, recall rare occurrences, and in general make better use of our experience.

### Performance

![navigation_results](./images/navigation_results.png)

The plot shows that the agent is able to receive an average reward of at least +13 (over 100 consecutive episodes). The agent manages to solve the environment with an average score of 13.01 in 516 episodes.

### Future Work

To improve the performance of the agent, several improvements can be made.
For example, Double DQN (DDQN) algorithm can be implemented to reduce the problem of overestimation of action values inherent in DQN.

Furthermore, prioritized experience replay can be implemented. Deep Q-Learning samples experience transitions uniformly from a replay memory. Prioritized experienced replay is based on the idea that the agent can learn more effectively from some transitions than from others, and the more important transitions should be sampled with higher probability.

Next, Dueling DQN can be implemented to further improve the agent. Currently, in order to determine which states are (or are not) valuable, we have to estimate the corresponding action values for each action. However, by replacing the traditional Deep Q-Network (DQN) architecture with a dueling architecture, we can assess the value of each state, without having to learn the effect of each action.

Finally, a new algorithm called Rainbow developed by Google Deepmind can be used to incorporate all six modifications listed below:

- Double DQN (DDQN)
- Prioritized experience replay
- Dueling DQN
- Learning from multi-step bootstrap targets
- Distributional DQN
- Noisy DQN
