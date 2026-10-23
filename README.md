# RL-Based Adaptive Cruise Control Algorithm Analysis

A tabular reinforcement-learning comparison for a simplified **Adaptive Cruise Control (ACC)** simulation. The project compares Q-Learning, SARSA, and SARSA(λ) using a custom stochastic vehicle-following environment.

**Course project:** Reinforcement Learning, Robotics & Mechatronics Engineering Master's, Polytech Nice-Sophia, Université Côte d’Azur.

## Algorithms and design

- **Q-Learning** — off-policy temporal-difference control.
- **SARSA** — on-policy temporal-difference control.
- **SARSA(λ)** — SARSA with replacing eligibility traces.
- **ACC environment** — simulates ego and leader speeds, relative distance, discrete acceleration actions, a distance-based reward, and terminal crash/lost-leader events.
- **State discretization** — maps distance and relative velocity into a finite tabular state space.
- **Evaluation outputs** — training reward curves, policy/value heatmaps, and a CSV KPI log.

The environment is a project-specific Python class with a simple four-value `step()` return (`state, reward, done, info`). It is **not currently a Gymnasium API implementation**.

## Repository structure

```text
agents/
  q_learning.py
  sarsa.py
  sarsa_lambda.py
environment/
  acc_env.py
utils/
  discretizer.py
  plotting.py
main.py
requirements.txt
```

## Setup and run

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
python main.py
```

The default configuration in `main.py` trains each algorithm for 10,000 episodes with up to 400 steps per episode. This can take a while; edit `CONFIG['global_training']` to run a smaller experiment while developing.

Results are written under `results/`:
- `logs/kpi_log.csv` — aggregate KPI rows appended for each run.
- `comparison_results_tuned/` — learning curves, policy heatmaps, and value heatmaps.

## Configuration and interpretation

The environment and training parameters are centralized in `CONFIG` in `main.py`. The environment includes stochastic leader acceleration, so outcomes vary between runs. The current script does not set and record a random seed, and the main training routine evaluates each algorithm in a single sequential run rather than reporting multi-seed confidence intervals. Treat comparisons as exploratory, not as statistically conclusive evidence of general algorithm superiority.

The current KPI counters count terminal crash/lost-leader events. Episodes that reach the configured step limit are truncated by the training loop and are not counted as terminal failures.

## Tests

Run the lightweight core tests with:

```bash
python -m unittest discover -s tests -v
```

These tests cover state-index bounds/coverage and terminal-update behavior. They do not validate convergence or establish that one algorithm is safer or more effective than another.

## License

MIT. See [LICENSE](LICENSE).
