import cv2
import numpy as np

from .game.flappy_bird import Game
from .network.dqn import DQN


REPLAY_MEMORY = 50000  # number of previous transitions to remember
OBSERVE = 100000.0  # timesteps to observe before training
EXPLORE = 2000000.0  # frames over which to anneal epsilon

def preprocessing(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 84)), cv2.COLOR_BGR2GRAY)
    _, observation = cv2.treshold(observation, 1, 255, cv2.TRESH_BINARY)
    return np.reshape(observation, (84, 84, 1))

def play():
    # Init flappy bird and DQN
    flappy_bird = Game()
    legal_actions = 2
    dqn = DQN(actions=legal_actions, replay_memory=REPLAY_MEMORY,
              observe=OBSERVE, explore=EXPLORE,
              game_name="flappy_bird")

    # Init state - do nothing
    start_action = np.zeros(legal_actions)
    start_action[0] = 1
    start_observation, start_reward, terminal = flappy_bird.next_state(start_action)
    start_observation = preprocessing(start_observation)
    dqn.set_init_state(start_observation)

    # run the game and train
    while True:
        action = dqn.get_action()
        next_observation, reward, terminal = flappy_bird.next_state(action)
        next_observation = preprocessing(next_observation)
        dqn.run(next_observation, action, reward)


def main():
    play()


if __name__ == "main":
    main()
