import cv2
import numpy as np
import sys
sys.path.append("game/")

from flappy_bird import Game
from dqn import DQN


REPLAY_MEMORY = 50000  # number of previous transitions to remember
OBSERVE = 100000.0  # timesteps to observe before training
EXPLORE = 2000000.0  # f
STATUSES = {"state": 0, "action": 1, "reward": 2, "next_state": 3, "terminate": 4}


def preprocessing(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 84)), cv2.COLOR_BGR2GRAY)
    _, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (84, 84, 1))


def play():
    # Init flappy bird and DQN
    flappy_bird = Game()
    legal_actions = 2
    dqn = DQN(actions=legal_actions, max_replay_memory=REPLAY_MEMORY, observe=OBSERVE,
              explore=EXPLORE, statuses=STATUSES, game_name="flappy_bird")

    # Init state - do nothing
    start_action = np.zeros(legal_actions, dtype=int)
    start_action[0] = 1
    start_observation, start_reward, terminal = flappy_bird.next_state(start_action)
    start_observation = cv2.cvtColor(cv2.resize(start_observation, (84, 84)),
                                     cv2.COLOR_BGR2GRAY)
    _, start_observation = cv2.threshold(start_observation, 1, 255, cv2.THRESH_BINARY)
    dqn.set_init_state(start_observation)

    # run the game and train
    while True:
        action = dqn.get_action()
        next_state, reward, terminal = flappy_bird.next_state(action)
        next_state = preprocessing(next_state)
        dqn.fit(next_state, action, reward, terminal)


if __name__ == "__main__":
    play()
