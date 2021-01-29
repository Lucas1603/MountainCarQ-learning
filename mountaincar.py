import gym
import numpy as np

# initial values to the constants that the algorithm need
#   learning rate
alpha = 0.6
#   evaluation that we give to the future rewards (discount factor)
sigma = 0.9
#   probability of explore over take the best action
epsilon = 0.5

# start the environment
env = gym.make('MountainCar-v0')
env.reset()

# initialize the q-table with zeros
####q_table = np.zeros((100,100,3))
q_table = np.random.uniform(low=0, high = 2, size=(100,100,3))

def get_discrete_state(state):
    # defines the dimension of the array that will save the states data
    dimension = np.array([100, 100])
    # get the interval of the continuos values
    interval = env.observation_space.high - env.observation_space.low
    print('MAIOR: ', env.observation_space.high)
    # converts all the numbers to positive.
    # e.g. -0.6 - (-0.6) = 0, and -0.6 in fact the lowest value
    state -= env.observation_space.low

    # rule of 3 to find the value and then turn it down
    discrete_state = (state * dimension) / interval
    return np.array([int(discrete_state[0]), int(discrete_state[1]) ])

def create_Egreedy_police(state, num_actions):
    def policy(state):
        '''
            returns the probabilty of each actions being taken.
            it depends on the value of epsilon.
        '''
        action_probabilities = np.ones(num_actions ,dtype='float') * epsilon / num_actions

        pos, vel = get_discrete_state(state)

        best_action = np.argmax(q_table[pos, vel])
        action_probabilities[best_action] += (1.0 - epsilon)
        return action_probabilities
    return policy
