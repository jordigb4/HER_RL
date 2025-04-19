# Hindsight Experience Replay (HER) Implementation

## Project Description
This project implements Hindsight Experience Replay (HER), a technique designed to significantly improve sample efficiency in reinforcement learning, particularly for tasks with sparse rewards and goal-based objectives. HER works by augmenting the standard experience replay buffer: after a trajectory is completed, HER saves additional transitions where the achieved state is treated as the intended goal. This allows the agent to learn even from "failed" attempts.

## Papers Replicated
- Hindsight Experience Replay (Andrychowicz et al., 2017)

## Algorithms Overview

### HER (Hindsight Experience Replay)
- **Type**: Experience Replay Modification, Sample Efficiency Technique
- **Key Features**:
  - Specifically designed for goal-conditioned reinforcement learning tasks.
  - Addresses sparse reward problems by creating artificial "success" signals.
  - Relabels transitions in the replay buffer using achieved states as substitute goals.
  - Algorithm-agnostic: Can be combined with any off-policy RL algorithm that uses a replay buffer (e.g., DDPG, SAC, TD3).
  - Improves learning speed and final performance in environments where reaching the desired goal is initially rare.

## Contributions
Contributions are welcome! Please read the contributing guidelines before submitting a pull request.

## License
MIT License

## References
1. Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Abbeel, P., & Zaremba, W. (2017). Hindsight Experience Replay.
