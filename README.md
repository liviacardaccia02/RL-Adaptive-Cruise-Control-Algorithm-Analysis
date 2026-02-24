# RL-Based Adaptive Cruise Control Algorithm Analysis

**Final Project of Reinforcement Learning Class | Robotics & Mechatronics Engineering Master's | Polytech Nice-Sophia | Université Côte d’Azur**

This repository presents a comparative analysis of various Reinforcement Learning (RL) algorithms applied to the **Adaptive Cruise Control (ACC)** problem. The goal is to train an autonomous agent to maintain a safe distance and optimal speed relative to a leading vehicle in a stochastic environment.

**Project Participants:**

* Livia Cardaccia
* Tytus Felbor
* Alessandro Negri

---

### 📂 Repository Structure

The project is modularized to separate the physical environment from the learning logic:

* **`environment/acc_env.py`**: A custom Gymnasium-compatible environment that simulates the longitudinal dynamics of a vehicle, including distance headway and relative velocity.
* **`agents/`**: Contains the implementations of various RL strategies:
* **Q-Learning**: A classic off-policy temporal difference (TD) control algorithm.
* **SARSA**: An on-policy TD control algorithm focusing on the current policy's actions.
* **SARSA($\lambda$)**: An eligibility trace-based approach that combines the advantages of TD and Monte Carlo methods.


* **`utils/`**: Helper scripts for state-space discretization and performance visualization.
* **`main.py`**: The central entry point for training agents, running simulations, and comparing convergence metrics.

---

### 🚀 Technical Implementation

#### 1. The ACC Environment

The environment models a "following" vehicle and a "leading" vehicle. The state space includes:

* Relative distance (Headway).
* Velocity of both the follower and the leader.
* Acceleration/Deceleration actions that impact the follower's kinetic state.

#### 2. Discretization Strategy

Since the agents implemented are tabular-based (Q-Learning, SARSA), the continuous state space of the ACC environment is mapped into discrete bins using the `Discretizer` utility to ensure efficient convergence of the Q-tables.

#### 3. Comparative Benchmarking

The project compares the learning efficiency of **Q-Learning** vs. **SARSA** variants. By analyzing cumulative rewards and collision rates, the project evaluates how on-policy vs. off-policy updates affect the safety-critical nature of adaptive cruise control.

---

### 🛠 Installation & Usage

1. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


2. **Run Training & Analysis:**
Execute the main script to initiate the training loop for the selected agents:
```bash
python main.py

```



### 📜 License

This project is licensed under the **MIT License**.
