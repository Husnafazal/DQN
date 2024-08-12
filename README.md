# DQN

This repository contains the implementation of a reinforcement learning agent trained to navigate a 5x5 hospital grid using Deep Q-Learning. The agent aims to reach the medicine cabinet while avoiding obstacles such as a doctor and nurse.

## Files

- `hospital_env.py`: Custom Gym environment for the hospital grid.
- `test_env.py`: Script to test the custom Gym environment to ensure it works as expected.
- `train.py`: Script to train the DQN agent.
- `play.py`: Script to simulate the environment using the trained policy.

## How to Run

1. Clone the repository:
    ```sh
    git clone git@github.com:Husnafazal/DQN.git
    cd DQN
    ```

2. Activate a virtual environment (optional):

3. Install the necessary libraries:
    ```sh
    pip install -r requirements.txt
    ```

3. Train the agent:
    ```sh
    python train.py
    ```

4. Simulate the environment using the trained policy:
    ```sh
    python play.py
    ```

## Simulation Video

Watch the simulation video [here](https://drive.google.com/file/d/17LRIwZoynY3EXXs5T5DCuzWt3Bq_zMqO/view?usp=sharing).

