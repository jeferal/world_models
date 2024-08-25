"""
  This script loads a .npz file containing a rollout and visualizes it. The script is useful for debugging purposes.
"""

import argparse
import numpy as np
import cv2

import matplotlib.pyplot as plt


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument('--file', type=str, help="Path to the .npz file containing the rollout")

  args = parser.parse_args()

  data = np.load(args.file)

  # The observations are images in the form of (96, 96, 3) arrays
  observations = data['observations']
  rewards = data['rewards']
  actions = data['actions']

  # Generate a plot that displays the actions
  # The plot consists of 3 subplots, one for each action dimension
  fig, axs = plt.subplots(3, 1)
  fig.suptitle('Actions')
  ACTIONS_NAME = ['Steering', 'Gas', 'Breaking']
  for i in range(3):
    axs[i].plot(actions[:, i])
    axs[i].set_ylabel(f"Action {ACTIONS_NAME[i]}")
  plt.savefig('actions.png')

  # Iterate over the frames and display them using cv2
  for i, obs in enumerate(observations):
    cv2.imshow('frame', obs)
    print(f"Frame {i}, Reward: {rewards[i]}, Action: {actions[i]}")
    cv2.waitKey(30)

  cv2.destroyAllWindows()
