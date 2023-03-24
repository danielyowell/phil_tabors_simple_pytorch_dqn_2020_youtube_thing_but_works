import gym
from dqn import Agent
from utils import plotLearning
from utils import make_env
import numpy as np

if __name__ == '__main__':
    # original code, using v1 api
    #env = gym.make('LunarLander-v2')
    #    env = gym.make(env_name,new_step_api=True) ##new api
    # new cod using v2 api
    #env = make_env("LunarLander-v2")
    #env = gym.make("LunarLander-v2")
    #agent = Agent( gamma=0.99,
    #               epsilon=1.0,
    #               batch_size=64,
    #               n_actions=4,
    #               eps_end=0.01,
    #               input_dims=[8],
    #               lr=0.003)


    env = gym.make("LunarLander-v2")
    #env = gym.make("CartPole-v1")

    # The following should extract the action space and
    # state size from any gym
    state, info = env.reset()   
    agent = Agent( gamma=0.99,
                   epsilon=1.0,
                   batch_size=64,
                   n_actions=env.action_space.n,
                   eps_end=0.01,
                   input_dims=[len(state)],
                   lr=0.003)

    # initialize some lists
    scores, eps_history = [],[]
    # number of games to play
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation, info = env.reset()
        while not done:
            # take the view of the current state
            # and select a new action
            action = agent.choose_action(observation)

            # apply the new action
            #observation_, reward, done, info = env.step(action)
            # new api
            observation_, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated

            # increment the score
            score += reward

            # store the new transition in the replay buffer
            agent.store_transition(observation,
                                   action,
                                   reward,
                                   observation_,
                                   done)

            # now we learn
            agent.learn()

            # set current state to new state
            observation = observation_

        # now we are at the end of the episode
        # we want to take some metrics so we can
        # see if our agent is learning
        scores.append(score)
        eps_history.append(agent.epsilon)

        # running average of the previous 100 games
        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

    x = [i+1 for i in range(n_games)]
    filename = 'lunar_lander.png'
    plotLearning(x,scores,eps_history,filename)
