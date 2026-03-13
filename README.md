# AISE 4030 – Super Mario D3QN (Assignment 1)

This repository contains our implementation of a **Double Dueling Deep Q-Network (D3QN)** agent trained to play **Super Mario Bros World 1-1** using the `gym-super-mario-bros` environment.

At this point the repository includes implementations for:

- **Task 4.1**: Online learning without experience replay
- **Task 4.2**: Uniform experience replay
- **Task 4.3**: Prioritized experience replay

---

# What Has Been Completed (Task 4.1)

Implemented a **Double Dueling DQN agent that learns online without experience replay**.

Main components implemented:

- Dueling DQN network architecture
- Double DQN target computation
- Online learning (update after every step)
- Target network synchronization
- Epsilon-greedy exploration
- Training logging and result saving

Training was run for **5000 episodes**, and the outputs are stored in the `d3qn_results` folder.

This baseline is expected to be **unstable**, since replay buffers are not used.

---

# Repository Structure

Super Mario D3QN/

config.yaml  
Training parameters and environment configuration

training_script.py  
Main training loop

d3qn_agent.py  
D3QN agent implementation

d3qn_network.py  
Neural network architecture (dueling network)

environment.py  
Mario environment setup and preprocessing wrappers

replay_buffer.py  
Replay buffer implementation (used in later tasks)

utils.py  
Helper functions used during training

d3qn_results/  
Contains training outputs

d3qn_online_model.pth  
Saved trained model

episode_rewards.npy  
Reward history

episode_losses.npy  
Loss history

training_summary.json  
Training statistics

---

# Environment Setup

The project was developed using **Python 3.10** inside a Conda environment.

Create environment:

conda create -n AISE4030 python=3.10  
conda activate AISE4030  

Install dependencies:

pip install torch torchvision  
pip install gym gym-super-mario-bros nes-py  
pip install numpy pyyaml  

---

# Environment Configuration

Environment used:

SuperMarioBros-1-1-v3

The agent uses a simplified **two-action space**:

0 → move right  
1 → move right + jump  

Reducing the action space helps exploration and speeds up learning.

---

# Observation Preprocessing

Raw observation:

240 × 256 × 3 (RGB)

Wrappers applied:

1. Frame Skip = 4  
   Repeats the same action for 4 frames and sums the rewards.

2. Grayscale conversion

3. Resize to 84 × 84

4. Frame stacking (4 frames)

Final observation shape:

(4, 84, 84)

Frame stacking allows the agent to observe motion.

---

# Network Architecture

The network uses a **Dueling DQN architecture**.

The convolutional layers extract features from the input frames, then the network splits into two streams:

Value stream

V(s)

Advantage stream

A(s,a)

They are combined as:

Q(s,a) = V(s) + A(s,a) − mean(A(s,*))

This helps the network better estimate state values and action advantages.

---

# Double DQN Target Computation

To reduce overestimation bias:

Action selection:

a' = argmax Q_policy(s',a)

Action evaluation:

Q_target(s',a')

TD target:

target = r + γ * Q_target(s', a') * (1 − done)

---

# Running Training

To start training:

python training_script.py

The script will:

1. Initialize the Mario environment  
2. Create the D3QN agent  
3. Train for the configured number of episodes  
4. Save the trained model and training statistics  

Training progress is printed periodically in the terminal.

---

# Training Outputs

After training, the following files are produced in `d3qn_results`:

d3qn_online_model.pth  
Saved trained model weights.

episode_rewards.npy  
Reward history for plotting training performance.

episode_losses.npy  
Loss history.

training_summary.json  
Summary statistics of the training run.

---

# Expected Behavior (Task 4.1)

Because experience replay is **not used**, the agent typically shows:

- unstable reward curves
- occasional high reward episodes
- frequent regressions in performance

This is expected and serves as the **baseline comparison for later tasks**.

---

# Next Tasks

Task 4.2  
Add **Experience Replay Buffer**

Expected improvements:

- more stable learning  
- reduced correlation between samples  
- better performance  

Task 4.3  
Add **Prioritized Experience Replay**

Expected improvements:

- faster learning  
- more efficient updates  

To run Task 4.3, set `agent_type: "d3qn_per"` in `config.yaml` and run:

python training_script.py

