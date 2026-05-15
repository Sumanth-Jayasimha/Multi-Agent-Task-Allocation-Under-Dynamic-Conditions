import pandas as pd
import numpy as np
import random
import os

import src.fleet_greedy_allocationDynamic as ga
import src.fleet_auction_allocation as aa
import src.fleet_q_learning_persistent as qlp
import src.DataGenerationDynamic as dgd


# -----------------------------
# CONFIG
# -----------------------------
TIME_STEPS = 100
NEW_TASK_PROB = 0.5
URGENCY_INCREMENT = 1
NUM_RUNS = 10

RESET_Q_TABLE = False   # Set False if you want learning across runs


# -----------------------------
# OPTIONAL: RESET Q-TABLE
# -----------------------------
if RESET_Q_TABLE and os.path.exists("q_table.pkl"):
    os.remove("q_table.pkl")
    print("Q-table reset")


# -----------------------------
# METRICS FUNCTION
# -----------------------------
def compute_metrics(engagement_metrics, vehicles_df):
    if not engagement_metrics:
        return {}

    num_tasks = len(engagement_metrics)

    total_engagement_time = sum(item["engagement_time"] for item in engagement_metrics)
    avg_engagement_time = total_engagement_time / num_tasks

    avg_normalized_engagement = sum(
        item["normalized_engagement_time"] for item in engagement_metrics
    ) / num_tasks

    total_task_duration = sum(item["task_duration"] for item in engagement_metrics)

    total_energy_consumed = sum(item["energy_consumed"] for item in engagement_metrics)

    energy_per_unit = (
        total_energy_consumed / total_task_duration
        if total_task_duration > 0 else None
    )

    throughput = total_task_duration

    total_idle_time = vehicles_df["Idle Time"].sum()
    avg_idle_time = total_idle_time / len(vehicles_df)

    return {
        "num_tasks": num_tasks,
        "avg_engagement_time": avg_engagement_time,
        "avg_normalized_engagement": avg_normalized_engagement,
        "energy_per_unit": energy_per_unit,
        "throughput": throughput,
        "total_idle_time": total_idle_time,
        "avg_idle_time": avg_idle_time
    }


# -----------------------------
# SINGLE RUN
# -----------------------------
def run_simulation():

    vehicles_df = dgd.generate_vehicle_data(10)

    vehicles_greedy = vehicles_df.copy()
    vehicles_auction = vehicles_df.copy()
    vehicles_q = vehicles_df.copy()

    for vehicles in [vehicles_greedy, vehicles_auction, vehicles_q]:
        vehicles['Busy'] = False
        vehicles['Remaining Duration'] = 0
        vehicles["Battery Level (%)"] = vehicles["Battery Level (%)"].astype(float)
        vehicles["Remaining Duration"] = vehicles["Remaining Duration"].astype(float)
        vehicles["Idle Time"] = 0.0

    initial_tasks = dgd.generate_task_data(num_tasks=10)

    tasks_waiting_greedy = initial_tasks.to_dict('records')
    tasks_waiting_auction = initial_tasks.to_dict('records')
    tasks_waiting_q = initial_tasks.to_dict('records')

    allocations_greedy = {}
    allocations_auction = {}
    allocations_q = {}

    engagement_metrics_greedy = []
    engagement_metrics_auction = []
    engagement_metrics_q = []

    task_counter = 1

    # -----------------------------
    # SIMULATION LOOP
    # -----------------------------
    for t in range(TIME_STEPS):

        # Update vehicles
        for vehicles in [vehicles_greedy, vehicles_auction, vehicles_q]:
            for idx, vehicle in vehicles.iterrows():
                if vehicle['Busy']:
                    new_duration = vehicle['Remaining Duration'] - 1
                    vehicles.at[idx, 'Remaining Duration'] = new_duration

                    if new_duration <= 0:
                        vehicles.at[idx, 'Busy'] = False
                        vehicles.at[idx, 'Remaining Duration'] = 0
                else:
                    vehicles.at[idx, 'Idle Time'] += 1

        # Increase urgency
        for task_list in [tasks_waiting_greedy, tasks_waiting_auction, tasks_waiting_q]:
            for task in task_list:
                task['Urgency'] += URGENCY_INCREMENT

        # New task arrival
        if random.random() < NEW_TASK_PROB:
            new_task = {
                'Task ID': f"D{task_counter}",
                'Task Position (x, y)': (random.randint(0, 100), random.randint(0, 100)),
                'Urgency': random.randint(0, 9),
                'Duration (min)': random.randint(10, 30)
            }
            task_counter += 1

            tasks_waiting_greedy.append(new_task.copy())
            tasks_waiting_auction.append(new_task.copy())
            tasks_waiting_q.append(new_task.copy())

        # -----------------------------
        # GREEDY
        # -----------------------------
        if tasks_waiting_greedy:
            df = pd.DataFrame(tasks_waiting_greedy)
            alloc, details = ga.greedy_allocation(vehicles_greedy, df)

            allocations_greedy.update(alloc)
            engagement_metrics_greedy.extend(details)

            allocated_ids = set(alloc.keys())
            tasks_waiting_greedy = [
                t for t in tasks_waiting_greedy if t['Task ID'] not in allocated_ids
            ]

        # -----------------------------
        # AUCTION
        # -----------------------------
        if tasks_waiting_auction:
            df = pd.DataFrame(tasks_waiting_auction)
            alloc, details = aa.auction_allocation(vehicles_auction, df)

            allocations_auction.update(alloc)
            engagement_metrics_auction.extend(details)

            allocated_ids = set(alloc.keys())
            tasks_waiting_auction = [
                t for t in tasks_waiting_auction if t['Task ID'] not in allocated_ids
            ]

        # -----------------------------
        # Q-LEARNING
        # -----------------------------
        if tasks_waiting_q:
            df = pd.DataFrame(tasks_waiting_q)
            alloc, details = qlp.q_learning_allocation(vehicles_q, df)

            allocations_q.update(alloc)
            engagement_metrics_q.extend(details)

            allocated_ids = set(alloc.keys())
            tasks_waiting_q = [
                t for t in tasks_waiting_q if t['Task ID'] not in allocated_ids
            ]

    # -----------------------------
    # METRICS
    # -----------------------------
    mg = compute_metrics(engagement_metrics_greedy, vehicles_greedy)
    ma = compute_metrics(engagement_metrics_auction, vehicles_auction)
    mq = compute_metrics(engagement_metrics_q, vehicles_q)

    return mg, ma, mq


# -----------------------------
# MULTIPLE RUNS
# -----------------------------
results = []

for i in range(NUM_RUNS):
    print(f"\nRun {i+1}")
    mg, ma, mq = run_simulation()
    results.append((mg, ma, mq))

    # Save Q-table after each run (learning)
    qlp.save_q_table()


# -----------------------------
# AVERAGE RESULTS
# -----------------------------
def avg(results, key, idx):
    vals = [r[idx][key] for r in results if r[idx]]
    return sum(vals) / len(vals)


print("\n AVERAGE RESULTS ")

for key in ["num_tasks", "throughput", "avg_idle_time", "energy_per_unit"]:
    print(f"\n{key}:")
    print("Greedy:", avg(results, key, 0))
    print("Auction:", avg(results, key, 1))
    print("Q-learning:", avg(results, key, 2))