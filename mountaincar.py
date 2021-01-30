import gym
import numpy as np

# initial values to the constants that the algorithm need
#   learning rate
alpha = 0.8
#   evaluation that we give to the future rewards (discount factor)
gamma = 1
#   probability of explore over take the best action
epsilon = 0.6

# start the environment
env = gym.make('MountainCar-v0')
env.reset()

# initialize the q-table with zeros
q_table = np.zeros((100,100,3))
#q_table = np.random.uniform(low=0, high = 2, size=(100,100,3))

def get_discrete_state(state):
    # makes a copy of the state to don't overwrite the data
    copy_state = state.copy()

    # defines the dimension of the array that will save the states data
    dimension = np.array([100, 100])
    # get the interval of the continuous values
    interval = env.observation_space.high - env.observation_space.low

    # converts all the numbers to positive.
    # e.g. -0.6 - (-0.6) = 0, and -0.6 in fact the lowest value
    copy_state -= env.observation_space.low

    # rule of 3 to find the value and then turn it down
    discrete_state = (copy_state * dimension) / interval
    return np.array([int(discrete_state[0]), int(discrete_state[1]) ])


def create_Egreedy_police(num_actions):
    def policy(state):
        '''
            returns the probabilty of each actions being taken.
            it depends on the value of epsilon.
        '''
        action_probabilities = np.ones(num_actions ,dtype='float') * epsilon / num_actions

        x, y = get_discrete_state(state)

        best_action = np.argmax(q_table[x, y])
        
        action_probabilities[best_action] += (1.0 - epsilon)
        return action_probabilities

    return policy


policy = create_Egreedy_police(env.action_space.n)

# i will be the episodes counter
i=0
#num_episodes = 600
while True:
    # starts the episode
    state = env.reset()
    pos,vel = get_discrete_state( state )

    while True:
        # calculate the probabilities of taking each action
        probabilities = policy(state)

        # determine what action the agent will take, following a epsilon-greedy policy
        action = np.random.choice( np.arange( len(probabilities) ), p=probabilities )

        # take the action
        next_state, reward, done, _ = env.step(action)

        next_pos, next_vel = get_discrete_state(next_state)

        # figure out what the best action is
        best_next_action = np.argmax( q_table[next_pos][next_vel] ) 
        # calculate td target, td delta and make the update following the q-learning algorithm
        td_target = reward + gamma * q_table[next_pos][next_vel][best_next_action]
        td_delta = td_target - q_table[pos][vel][action]
        q_table[pos][vel][action] += alpha * td_delta

        # if the agent reaches the goal, it can start another episode
        if done:
            break
        
        # if the agent trained enough, you can start to see what he learned
        if i>30000:
            env.render()

        #update the state
        state = next_state
        pos, vel = get_discrete_state(state)

    # update the value of epsilon, every episode it will decay, untill reach the minimum of 0.01
    epsilon = max(0.001, epsilon*0.9995)
    
    i+=1
