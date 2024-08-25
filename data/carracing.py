"""
Generating data from the CarRacing gym environment.
!!! DOES NOT WORK ON TITANIC, DO IT AT HOME, THEN SCP !!!
"""
import argparse
from os.path import join, exists
import gymnasium as gym
import numpy as np
import math


def sample_continuous_policy(action_space, seq_len, dt):
    """ Sample a continuous policy.

    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).

    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization

    :returns: sequence of seq_len actions
    """
    actions = [action_space.sample()]
    for _ in range(seq_len):
      daction_dt = np.random.randn(*actions[-1].shape)
      actions.append(
        np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                action_space.low, action_space.high))
    return actions

def generate_data(rollouts, data_dir, noise_type): # pylint: disable=R0914
  """ Generates data """
  assert exists(data_dir), "The data directory does not exist..."

  env = gym.make("CarRacing-v2")
  seq_len = 1000

  for i in range(rollouts):
    env.reset()
    # env.env.viewer.window.dispatch_events()
    if noise_type == 'white':
      a_rollout = [env.action_space.sample() for _ in range(seq_len)]
    elif noise_type == 'brown':
      a_rollout = sample_continuous_policy(env.action_space, seq_len, 1. / 50)

    s_rollout = []
    r_rollout = []
    d_rollout = []

    t = 0
    for t in range(seq_len):
      action = a_rollout[t]
      t += 1
      print(f"Rollout {i}, frame {t}")
      s, r, done, _, _ = env.step(action.tolist())
      # env.env.viewer.window.dispatch_events()
      s_rollout += [s]
      r_rollout += [r]
      d_rollout += [done]

      if done:
        break

    print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
    np.savez(join(data_dir, 'rollout_{}'.format(i)),
              observations=np.array(s_rollout),
              rewards=np.array(r_rollout),
              actions=np.array(a_rollout),
              terminals=np.array(d_rollout))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--rollouts', type=int, help="Number of rollouts")
  parser.add_argument('--dir', type=str, help="Where to place rollouts")
  parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                      help='Noise type used for action sampling.',
                      default='brown')
  args = parser.parse_args()
  generate_data(args.rollouts, args.dir, args.policy)
