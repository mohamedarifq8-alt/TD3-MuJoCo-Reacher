# 🚀 Mastering TD3: Target-Driven Robotic Control (Reacher-v4)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL_Environment-112C3E)](https://gymnasium.farama.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-Physics_Engine-1A1A1A?logo=google&logoColor=white)](https://mujoco.org/)
[![Stable Baselines3](https://img.shields.io/badge/Stable_Baselines3-Production_RL-4B8BBE)](https://stable-baselines3.readthedocs.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Mathematics-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Model_Training-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)

This repository documents the engineering journey of training a 2-joint robotic arm to precisely reach a randomized target using the **Twin Delayed DDPG (TD3)** algorithm[cite: 18]. This project shifts the focus from simple balancing to target-driven industrial robotic control[cite: 18].

---

## 🛤️ The Journey (Project Phases)

This project was built to deeply understand deterministic policies and negative reward shaping in Reinforcement Learning:

### Phase 1: Environment Analysis (MDP Formulation)
- **Description:** Formulating the Markov Decision Process (MDP) for the MuJoCo Reacher environment[cite: 18].
- **Action Space:** Continuous 2D space controlling the torque of 2 joints (shoulder and elbow) with bounds of [-1, 1][cite: 18].
- **Observation Space:** 11-dimensional vector including angles (using Sine/Cosine), angular velocities, target coordinates, and the relative distance vector to the target[cite: 19].
- **Result:** Established a solid mathematical understanding of the agent's sensory inputs and mechanical constraints.

### Phase 2: Reward Engineering & Cost Minimization
- **Description:** Mastering the concept of "Continuous Negative Rewards" (Cost Minimization)[cite: 19, 20]. 
- **Techniques Used:**
  * Implementing a reward function where the maximum possible reward is `0.0`[cite: 20]. 
  * Applying a **Distance Penalty** to encourage moving towards the target and a **Control Penalty** to prevent excessive mechanical force[cite: 19].
  * Utilizing **Fast Truncation**: The environment ends exactly after 50 steps (Sprint), forcing the agent to learn rapidly without a termination/failure state[cite: 19].

### Phase 3: Production-Grade Training (TD3 Algorithm)
- **Description:** Training the agent using Stable Baselines3 and managing exploration mathematically.
- **Techniques Used:**
  * Choosing **TD3 (Deterministic Policy)** as it excels in precise, continuous industrial robotics tasks[cite: 18].
  * Injecting **Gaussian Action Noise** (`NormalActionNoise`) into the environment to force the deterministic agent to explore during early training phases[cite: 22].
  * Utilizing a large **Replay Buffer** to sample past experiences effectively (Off-Policy)[cite: 22, 23].
- **Result:** The robotic arm rapidly reaches the red target and stabilizes its end-effector on it with minimal energy consumption[cite: 21, 24].

---

## 🧠 Core Concepts Explored

- **Deterministic Policies (TD3):** Outputting exact action values rather than probability distributions[cite: 18].
- **Cost Minimization:** Designing environments where the agent experiences continuous negative rewards and strives to reach `0.0`[cite: 20].
- **Action Noise:** Manually adding mathematical noise to continuous actions to encourage exploration[cite: 22].
- **Truncation vs. Termination:** Differentiating between time-limit cutoffs (50 steps) and actual task failure[cite: 19].

---

### Repository Structure


```text
📦 TD3-MuJoCo-Reacher
 ┣ 📜 train_td3_reacher.py            # Main script containing the training and execution logic
 ┣ 📂 models                          
 ┃ ┗ 📜 td3_reacher.zip               # Final SB3 Production Model (Saved Weights & Config)
 ┣ 📂 videos                          
 ┃ ┗ 🎬 mujoco_reacher_td3.mp4        # Rendered video of the agent hitting targets
 ┗ 📜 README.md
```

---


## 🛠️ Installation & Repository Structure

### Prerequisites

```bash
pip install torch numpy gymnasium[mujoco] stable-baselines3[extra] imageioّ
```

---


---

## 🏋️ Training and Evaluation

To run the project (which includes building, training, and testing the TD3 model):

```bash
python train_td3_reacher.py
```
