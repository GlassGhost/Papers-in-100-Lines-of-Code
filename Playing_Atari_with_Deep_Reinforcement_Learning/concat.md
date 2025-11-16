#####
occurences of ``` will be escaped for markdown


#####
File: dqn.py
#####

```
# https://arxiv.org/pdf/1312.5602
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.buffers import ReplayBuffer
import os

"""
- `replay_mem_D` -> \(\mathcal{D}\)
- `replay_capacity_N` -> \(N\)
- `Q_val_of_action` -> \(Q\)
- `episode_iter_m` -> loop index for episodes
- `episode_count_M` -> total episodes
- `step_iter_t` -> loop index for steps
- `step_count_T` -> total steps
- `explore_prob_epsilon` -> \(\epsilon\)
- `action_at_t` -> \(a_t\)
- `best_action_from_Q` -> \(\max_{a} Q^*(\phi(s_t), a; \theta)\)
- `raw_obs_xt` -> \(x_t\)
- `raw_obs_xtp1` -> \(x_{t+1}\)
- `state_seq_st` -> \(s_t\)
- `state_seq_stp1` -> \(s_{t+1}\)
- `preproc_state_phit` -> \(\phi_t = \phi(s_t)\)
- `preproc_state_phitp1` -> \(\phi_{t+1} = \phi(s_{t+1})\)
- `reward_rt` -> \(r_t\)
- `transition_tuple` -> \((\phi_t, a_t, r_t, \phi_{t+1})\)
- `batch_phij` -> \(\phi_j\)
- `batch_aj` -> \(a_j\)
- `batch_rj` -> \(r_j\)
- `batch_phijp1` -> \(\phi_{j+1}\)
- `target_yj` -> \(y_j\)
- `loss_j` -> \( \left(y_j - Q(\phi_j, a_j; \theta) \right)^2\) squared TD error for sample \(j\)
- `grad_update_theta` -> \( \nabla_{\theta_i} L_i(\theta_i) \)gradient of loss w.r.t. parameters
"""

class DQN(nn.Module):
    def __init__(self, nb_actions):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(4, 16, 8, stride=4), nn.ReLU(),
                                     nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
                                     nn.Flatten(), nn.Linear(2592, 256), nn.ReLU(),
                                     nn.Linear(256, nb_actions), )

    def forward(self, x):
        return self.network(x / 255.)

def RMSProp(q_network, lr=2.5e-4, alpha=0.95, eps=0.01):
    """RMSProp optimizer (classic DQN setup)."""
    return torch.optim.RMSprop(q_network.parameters(), lr=lr, alpha=alpha, eps=eps)


def AdamConstantLR(q_network, lr=1e-4):
    """Adam optimizer with constant learning rate."""
    return torch.optim.Adam(q_network.parameters(), lr=lr)


def AdamLinearDecay(q_network, initial_lr=1e-4, final_lr=1e-5, nb_epochs=300_000):
    """Adam optimizer with linearly decaying learning rate."""
    # Scheduler will be attached later in training loop
    optimizer = torch.optim.Adam(q_network.parameters(), lr=initial_lr)
    lr_lambda = lambda epoch: final_lr + (initial_lr - final_lr) * (1 - epoch / nb_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler

def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
    else:
        obs = result
        info = {}
    return obs, info

def step_env(env, action):
    # Ensure env.step gets a scalar integer to satisfy ALE / gymnasium wrappers
    result = env.step(int(action))
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
    elif isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, info = result
    else:
        raise RuntimeError(f"Unexpected env.step() return shape: {type(result)} / {len(result) if isinstance(result, tuple) else 'N/A'}")
    return obs, reward, done, info


# <<< MODIFIED Function Signature >>>
# Renamed:
# - replay_memory_size -> replay_capacity_N
# - initial_exploration -> initial_epsilon
# - final_exploration -> final_epsilon

# original schedule without saving file
# def Deep_Q_Learning(env, replay_capacity_N=1_000_000, nb_epochs=30_000_000, update_frequency=4, batch_size=32,
#                     discount_factor=0.99, replay_start_size=80_000, initial_epsilon=1, final_epsilon=0.01,
#                     exploration_steps=1_000_000, device='cuda', model_path="dqn_breakout.pth"):

# new schedule constant epsilon and file save
def Deep_Q_Learning(env, replay_capacity_N=160_000, nb_epochs=160_000, update_frequency=4, batch_size=32,
# def Deep_Q_Learning(env, replay_capacity_N=160_000, nb_epochs=6_400_000, update_frequency=4, batch_size=32,
                    discount_factor=0.99, replay_start_size=80_000, initial_epsilon=0.03888, final_epsilon=0.03888,
                    exploration_steps=160_000, device='cuda', model_path="dqn_breakout.pth"):

    # Algorithm Line: Initialize replay memory D to capacity N
    # <<< RENAMED >>> rb -> replay_mem_D (maps to D)
    replay_mem_D = ReplayBuffer(replay_capacity_N, env.observation_space, env.action_space, device,
                                optimize_memory_usage=True, handle_timeout_termination=False)

    # Algorithm Line: Initialize action-value function Q with random weights
    # <<< RENAMED >>> q_network -> Q_val_of_action (maps to Q)
    Q_val_of_action = DQN(env.action_space.n).to(device)
    
    # Load existing model if it exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        Q_val_of_action.load_state_dict(torch.load(model_path, map_location=device)) # <<< RENAMED >>>
        print("Model loaded successfully.")
    else:
        print("No saved model found, starting from scratch.")

    # original
    # optimizer = torch.optim.Adam(Q_val_of_action.parameters(), lr=1.25e-4) # <<< RENAMED >>>

    #hot-swappable
    optimizer = AdamConstantLR(Q_val_of_action, lr=1e-4) # <<< RENAMED >>>

    # <<< RENAMED >>> epoch -> step_iter_t (maps to t)
    # Note: The algorithm's outer loop is episodes (M), inner is steps (T).
    # This code uses one main loop for total steps (nb_epochs), which is a
    # common implementation variant. We map 'epoch' to 'step_iter_t'.
    step_iter_t = 0
    smoothed_rewards = []
    rewards = []

    # Algorithm Line: For episode = 1, M
    # (This implementation loops for total steps, episodes are implicit)
    progress_bar = tqdm(total=nb_epochs, desc="Training Steps")
    while step_iter_t <= nb_epochs: # <<< RENAMED >>>

        dead = False
        total_rewards = 0

        # Algorithm Line: Initialise sequence s1 = {x1} and preprocessed sequenced φ1 = φ(s1)
        # <<< RENAMED >>> obs -> preproc_state_phit (maps to φ_t)
        preproc_state_phit, info = reset_env(env)

        for _ in range(random.randint(1, 30)):  # Noop and fire to reset environment
            preproc_state_phit, _, _, info = step_env(env, 1) # <<< RENAMED >>>

        # Algorithm Line: For t = 1, T
        # (This inner loop runs until an episode ends)
        while not dead:
            current_life = info['lives']

            # Calculate exploration probability ε
            # <<< RENAMED >>> epsilon -> explore_prob_epsilon (maps to ε)
            # <<< RENAMED >>> initial_epsilon, final_epsilon, step_iter_t
            explore_prob_epsilon = max((final_epsilon - initial_epsilon) / exploration_steps * step_iter_t + initial_epsilon,
                                       final_epsilon)
            
            # Algorithm Line: With probability ε select a random action a_t
            # <<< RENAMED >>> action_int -> action_at_t (maps to a_t)
            if random.random() < explore_prob_epsilon:
                action_at_t = int(env.action_space.sample())
            else:
                # Algorithm Line: otherwise select a_t = max_a Q*(φ(s_t), a; θ)
                with torch.no_grad(): # Use no_grad for inference
                    # <<< RENAMED >>> Q_val_of_action, preproc_state_phit
                    q_values = Q_val_of_action(torch.Tensor(preproc_state_phit).unsqueeze(0).to(device))
                action_at_t = int(torch.argmax(q_values, dim=1).item())

            # Algorithm Line: Execute action a_t in emulator and observe reward r_t and image x_{t+1}
            # <<< RENAMED >>> next_obs -> preproc_state_phitp1 (maps to φ_{t+1})
            # <<< RENAMED >>> reward -> reward_rt (maps to r_t)
            preproc_state_phitp1, reward_rt, dead, info = step_env(env, action_at_t)
            
            done = True if (info['lives'] < current_life) else False

            # Algorithm Line: Set s_{t+1} = ... and preprocess φ_{t+1} = φ(s_{t+1})
            # (The env wrapper handles preprocessing; we just copy the result)
            real_next_obs = preproc_state_phitp1.copy() # <<< RENAMED >>>

            total_rewards += reward_rt # Use original reward for episode stats
            reward_rt_clipped = np.sign(reward_rt)  # Reward clipping (maps to r_t)

            # Algorithm Line: Store transition (φ_t, a_t, r_t, φ_{t+1}) in D
            # <<< RENAMED >>> action_at_t
            action_for_buffer = np.array([action_at_t], dtype=np.int64).reshape(1, -1)
            # <<< RENAMED >>> replay_mem_D, preproc_state_phit, reward_rt_clipped
            replay_mem_D.add(preproc_state_phit, real_next_obs, action_for_buffer, reward_rt_clipped, done, info)

            # Update state: φ_t = φ_{t+1}
            preproc_state_phit = preproc_state_phitp1 # <<< RENAMED >>>

            # <<< RENAMED >>> step_iter_t
            if step_iter_t > replay_start_size and step_iter_t % update_frequency == 0:
                
                # Algorithm Line: Sample random minibatch of transitions (φ_j, a_j, r_j, φ_{j+1}) from D
                # <<< RENAMED >>> replay_mem_D
                data = replay_mem_D.sample(batch_size)
                
                # <<< ADDED >>> Use mapped variable names for clarity
                batch_phij = data.observations      # maps to φ_j
                batch_aj = data.actions            # maps to a_j
                batch_rj = data.rewards            # maps to r_j
                batch_phijp1 = data.next_observations # maps to φ_{j+1}
                batch_dones = data.dones           # Used for terminal check

                with torch.no_grad():
                    # Algorithm Line: Set y_j = ...
                    # y_j = r_j (for terminal φ_{j+1})
                    # y_j = r_j + γ * max_a' Q(φ_{j+1}, a'; θ) (for non-terminal)
                    
                    # <<< RENAMED >>> Q_val_of_action
                    max_q_value, _ = Q_val_of_action(batch_phijp1).max(dim=1)
                    
                    # <<< RENAMED >>> y -> target_yj (maps to y_j)
                    target_yj = batch_rj.flatten() + discount_factor * max_q_value * (1 - batch_dones.flatten())
                
                # Get Q(φ_j, a_j; θ)
                # <<< RENAMED >>> batch_aj, Q_val_of_action
                actions_for_gather = torch.as_tensor(batch_aj, device= Q_val_of_action.device if hasattr(Q_val_of_action, "device") else Q_val_of_action.parameters().__next__().device)
                actions_for_gather = actions_for_gather.long()
                # <<< RENAMED >>> Q_val_of_action, batch_phij
                current_q_value = Q_val_of_action(batch_phij).gather(1, actions_for_gather).squeeze()
                
                # Algorithm Line: Perform a gradient descent step on (y_j - Q(φ_j, a_j; θ))^2
                # <<< RENAMED >>> loss -> loss_j (maps to (y_j - Q(...))^2)
                loss_j = F.huber_loss(target_yj, current_q_value)

                optimizer.zero_grad()
                loss_j.backward()
                optimizer.step()

            step_iter_t += 1 # <<< RENAMED >>>
            
            # <<< RENAMED >>> step_iter_t
            if (step_iter_t % 50_000 == 0) and step_iter_t > 0:
                smoothed_rewards.append(np.mean(rewards))
                rewards = []
                plt.plot(smoothed_rewards)
                plt.title("Average Reward on Breakout")
                plt.xlabel("Training Steps (x50,000)") # <<< MODIFIED Label >>>
                plt.ylabel("Average Reward per Episode")
                plt.savefig('Imgs/average_reward_on_breakout.png')
                plt.close()

            progress_bar.update(1)
            
            # Check if the episode ended (agent died)
            if dead:
                break # Exit inner 'while not dead' loop
                
        rewards.append(total_rewards)

    progress_bar.close()
    # <<< MODIFIED >>> Clarified print statement (was "epochs", now "steps")
    print(f"\nTraining completed after {step_iter_t-1} steps. Saving final model...")
    # <<< RENAMED >>> Q_val_of_action
    torch.save(Q_val_of_action.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    MODEL_PATH = "dqn_breakout.pth"
    os.makedirs("Imgs", exist_ok=True) 
    
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0.0)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = MaxAndSkipEnv(env, skip=4)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available(): # First check for CUDA
        device = torch.device("cuda")
        print("Using CUDA backend for PyTorch.")
    # If CUDA is not available, check for Vulkan
    elif hasattr(torch.backends, "vulkan") and torch.backends.vulkan.is_available():
        device = torch.device("vulkan")
        print("Using Vulkan backend for PyTorch.")
    else: # Fallback to CPU
        device = torch.device("cpu")
        print("Neither CUDA nor Vulkan available, using CPU.")

    # Pass the model_path to the function
    Deep_Q_Learning(env, device=device, model_path=MODEL_PATH)
    
    env.close()
    
```

#####
File: requirements.txt
#####

```
gymnasium[atari]==1.1.1
matplotlib==3.10.1
numpy==2.2.5
stable-baselines3==2.6.0
torch==2.7.0
tqdm==4.67.1
ale-py==0.11.0
opencv-python==4.11.0
#ExecuTorch>=0.5.0

```

#####
File: test_output.log
#####

```
+ echo 'Running test at 2025.11.16T04.02.00Z'
Running test at 2025.11.16T04.02.00Z
+ cd /home/owner/Papers-in-100-Lines-of-Code/Playing_Atari_with_Deep_Reinforcement_Learning
+ nix-shell ./devShells.nix --run 'bash ./test.sh'
++ date -u +%Y.%m.%dT%H.%M.%SZ
+ utc=2025.11.16T04.02.10Z
+++ dirname ./test.sh
++ cd .
++ pwd
+ owd=/home/owner/Papers-in-100-Lines-of-Code/Playing_Atari_with_Deep_Reinforcement_Learning
+ cd /home/owner/Papers-in-100-Lines-of-Code/Playing_Atari_with_Deep_Reinforcement_Learning
+ python ./dqn.py
/home/owner/Papers-in-100-Lines-of-Code/Playing_Atari_with_Deep_Reinforcement_Learning/./dqn.py:17: SyntaxWarning: invalid escape sequence '\('
  - `replay_mem_D` -> \(\mathcal{D}\)
A.L.E: Arcade Learning Environment (version 0.11.0+unknown)
[Powered by Stella]
Traceback (most recent call last):
  File "/home/owner/Papers-in-100-Lines-of-Code/Playing_Atari_with_Deep_Reinforcement_Learning/./dqn.py", line 276, in <module>
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0.0)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nix/store/lqgr276wnv40z4cds98r5wirfny05mr2-python3.12-gymnasium-1.1.1/lib/python3.12/site-packages/gymnasium/envs/registration.py", line 742, in make
    env = env_creator(**env_spec_kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nix/store/ar8v4wvf13b3zjpakg5c8x9pwf6a9bjh-python3.12-ale-py-0.11.0/lib/python3.12/site-packages/ale_py/env.py", line 165, in __init__
    self.load_game()
  File "/nix/store/ar8v4wvf13b3zjpakg5c8x9pwf6a9bjh-python3.12-ale-py-0.11.0/lib/python3.12/site-packages/ale_py/env.py", line 228, in load_game
    self.ale.loadROM(roms.get_rom_path(self._game))
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nix/store/ar8v4wvf13b3zjpakg5c8x9pwf6a9bjh-python3.12-ale-py-0.11.0/lib/python3.12/site-packages/ale_py/roms/__init__.py", line 54, in get_rom_path
    with open(bin_path, "rb") as bin_fp:
         ^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/nix/store/ar8v4wvf13b3zjpakg5c8x9pwf6a9bjh-python3.12-ale-py-0.11.0/lib/python3.12/site-packages/ale_py/roms/breakout.bin'

```
