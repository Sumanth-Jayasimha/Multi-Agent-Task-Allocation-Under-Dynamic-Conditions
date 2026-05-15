# Multi-Agent-Task-Allocation-Under-Dynamic-Conditions
Multi-agent task allocation simulation comparing greedy, auction-based, and Q-learning approaches under dynamic conditions.

##  Overview

In real-world systems such as autonomous fleets, drones, or robotic swarms, tasks arrive dynamically and agents must decide:

- Which agent should take which task?
- How to balance urgency, distance, and energy?
- How to operate efficiently under uncertainty?

This project simulates such an environment and evaluates three approaches:

- Greedy Allocation
- Auction-Based Allocation
- Q-Learning (Reinforcement Learning)

---

##  System Design

The simulation models:

- A fleet of vehicles with:
  - Position
  - Battery level
  - Speed
  - Availability (busy/idle)

- Dynamically arriving tasks with:
  - Location
  - Urgency (increasing over time)
  - Duration

- Time-stepped execution with:
  - Task queues
  - Vehicle state updates
  - Continuous allocation decisions

---

##  Methods Implemented

### 1. Greedy Allocation
- Assigns the closest feasible vehicle
- Fast and simple baseline

### 2. Auction-Based Allocation
- Vehicles bid based on cost (distance, energy, etc.)
- Task assigned to lowest bidder

### 3. Q-Learning Allocation
- Learns task assignment policy over time
- State includes:
  - Task urgency
  - Distance (binned)
  - Battery level
- Uses ε-greedy exploration strategy
- Q-table persisted across runs

---

##  Evaluation Metrics

The following metrics were used:

- Number of tasks completed
- Throughput (total task duration handled)
- Average idle time
- Energy consumption per unit work

---

##  Results Summary

Across multiple simulation runs:

- Greedy performed consistently strong as a baseline
- Auction improved fairness but slightly increased overhead
- Q-learning showed adaptive behavior but required more tuning

The results highlight that:
- Simple heuristics can outperform learning in structured environments
- Reinforcement learning benefits from richer state representation and longer training

---
