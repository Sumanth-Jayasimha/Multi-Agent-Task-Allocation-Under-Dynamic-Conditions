import numpy as np
import os
import pickle
import random

Q_TABLE_FILE = "q_table.pkl"

# Load or initialize Q-table
if os.path.exists(Q_TABLE_FILE):
    with open(Q_TABLE_FILE, "rb") as f:
        Q = pickle.load(f)
    print("Loaded existing Q-table")
else:
    Q = {}
    print("Initialized new Q-table")


def calculate_distance(vehicle_pos, task_pos):
    return np.sqrt(
        (vehicle_pos[0] - task_pos[0])**2 +
        (vehicle_pos[1] - task_pos[1])**2
    )


# -------------------------
# FIXED STATE FUNCTION
# -------------------------
def get_state(vehicle, task, num_waiting_tasks):

    distance = calculate_distance(
        vehicle['Vehicle Position (x, y)'],
        task['Task Position (x, y)']
    )

    distance_bin = 0 if distance < 30 else (1 if distance < 70 else 2)
    battery_bin = 0 if vehicle['Battery Level (%)'] < 30 else (1 if vehicle['Battery Level (%)'] < 70 else 2)
    load_bin = 0 if num_waiting_tasks < 10 else (1 if num_waiting_tasks < 25 else 2)

    return (task['Urgency'], distance_bin, battery_bin, load_bin)


def get_action(vehicle):
    return vehicle['Vehicle ID']


# -------------------------
# REWARD
# -------------------------
def compute_reward(vehicle, task):

    distance = calculate_distance(
        vehicle['Vehicle Position (x, y)'],
        task['Task Position (x, y)']
    )

    travel_time = distance / vehicle['Speed']
    engagement_time = task['Duration (min)'] + travel_time

    if vehicle['Battery Level (%)'] < engagement_time:
        return -100, travel_time, engagement_time

    remaining_battery = vehicle['Battery Level (%)'] - engagement_time

    reward = (
        15 * task['Urgency']
        - 0.8 * distance
        - 0.6 * travel_time
        - 1.0 * engagement_time
        + 0.2 * remaining_battery
    )

    if remaining_battery < 20:
        reward -= 15

    return reward, travel_time, engagement_time


# -------------------------
# MAIN FUNCTION
# -------------------------
def q_learning_allocation(vehicles_df, tasks_df,
                          alpha=0.1, gamma=0.9, epsilon=0.3, epsilon_decay = 0.05):

    allocations = {}
    engagement_details = []

    tasks_df = tasks_df.sort_values('Urgency', ascending=False)

    for _, task in tasks_df.iterrows():

        available_vehicles = [
            v for _, v in vehicles_df.iterrows() if not v['Busy']
        ]

        if not available_vehicles:
            continue

        # -------------------------
        # SELECT ACTION
        # -------------------------
        if np.random.rand() < epsilon:
            chosen_vehicle = random.choice(available_vehicles)
        else:
            q_values = []

            for v in available_vehicles:
                state = get_state(v, task, len(tasks_df))
                action = get_action(v)
                q_values.append(Q.get((state, action), 0))

            chosen_vehicle = available_vehicles[np.argmax(q_values)]

        # -------------------------
        # GET STATE + ACTION
        # -------------------------
        state = get_state(chosen_vehicle, task, len(tasks_df))
        action = get_action(chosen_vehicle)

        # -------------------------
        # REWARD
        # -------------------------
        reward, travel_time, engagement_time = compute_reward(chosen_vehicle, task)

        # -------------------------
        # Q UPDATE (PROPER)
        # -------------------------
        old_q = Q.get((state, action), 0)

        future_q = max([
            Q.get((get_state(v, task, len(tasks_df)), get_action(v)), 0)
            for v in available_vehicles
        ])

        Q[(state, action)] = old_q + alpha * (reward + gamma * future_q - old_q)

        # -------------------------
        # ALLOCATION
        # -------------------------
        if reward > -20:

            allocations[task['Task ID']] = chosen_vehicle['Vehicle ID']

            idx = vehicles_df.index[
                vehicles_df['Vehicle ID'] == chosen_vehicle['Vehicle ID']
            ][0]

            vehicles_df.at[idx, 'Battery Level (%)'] -= engagement_time
            vehicles_df.at[idx, 'Busy'] = True
            vehicles_df.at[idx, 'Remaining Duration'] = engagement_time
            vehicles_df.at[idx, 'Vehicle Position (x, y)'] = task['Task Position (x, y)']

            engagement_details.append({
                "task_id": task['Task ID'],
                "task_duration": task['Duration (min)'],
                "travel_time": travel_time,
                "engagement_time": engagement_time,
                "normalized_engagement_time": engagement_time / task['Duration (min)'],
                "energy_consumed": engagement_time
            })

    return allocations, engagement_details


def save_q_table():
    with open(Q_TABLE_FILE, "wb") as f:
        pickle.dump(Q, f)
    print("Q-table saved")