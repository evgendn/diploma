import cv2
import os
import numpy as np
import sys
sys.path.append("game/")

from flappy_bird import Game
from dqn import DQN


STATUSES = {"state": 0, "action": 1, "reward": 2,
            "next_state": 3, "terminate": 4
            }

# preprocess raw image to 84x84 gray image
def preprocessing(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 84)),
                               cv2.COLOR_BGR2GRAY)
    _, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (84, 84, 1))


def play():
    # init flappy bird and DQN
    flappy_bird = Game()
    legal_actions = 2
    dqn = DQN(actions=legal_actions, max_replay_memory=50000, observe=100000,
              explore=200000, initial_epsilon=0.2, final_epsilon=0.01,
              learning_rate=1e-6, batch_size=32, gamma=0.99,
              statuses=STATUSES, game_name="flappy_bird")

    # init state - do nothing
    start_action = np.zeros(legal_actions, dtype=int)
    start_action[0] = 1
    start_observation, _, _ = flappy_bird.next_frame(start_action)
    start_observation = cv2.cvtColor(cv2.resize(start_observation, (84, 84)),
                                     cv2.COLOR_BGR2GRAY)
    _, start_observation = cv2.threshold(start_observation, 1,
                                         255, cv2.THRESH_BINARY)
    dqn.set_init_state(start_observation)

    try:
        # run the game and train
        while True:
            action, q_value = dqn.get_action()
            next_state, reward, terminal = flappy_bird.next_frame(action)

            # check preprocessed images
            if dqn.time_step % 10 == 0:
                if not os.path.exists("logs_png"):
                    os.mkdir("logs_png")
                cv2.imwrite("logs_png/frame" + str(dqn.time_step) + ".png",
                            next_state)
            next_state = preprocessing(next_state)

            # check preprocessed images
            if dqn.time_step % 10 == 0:
                if not os.path.exists("logs_png"):
                    os.mkdir("logs_png")
                cv2.imwrite("logs_png/framepr" + str(dqn.time_step) + ".png",
                            next_state)

            dqn.fit(next_state, action, reward, terminal)

            # print logs
            state = ""
            if dqn.time_step <= dqn.observe:
                state = "observe"
            elif dqn.observe < dqn.time_step <= dqn.observe + dqn.explore:
                state = "explore"
            else:
                state = "train"

            message = "timestamp: {0}, state: {1}, epsilon: {2},\
                       action: {3}, reward: {4}, q_value: {5}"
            message = message.format(dqn.time_step, state, dqn.epsilon,
                                    action, reward, np.max(q_value))
            print(message)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt")


def main():
    play()


if __name__ == "__main__":
    main()
