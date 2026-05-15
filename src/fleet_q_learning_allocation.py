# fleet q-learning allocation

import numpy as np
import random

# q-table
Q = {}

# hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.3
epsilon_decay = 0.995
epsilon_min = 0.05


# distance function
def calculate_distance(vehicle_pos, task_pos):
    return np.sqrt(
        (vehicle_pos[0] - task_pos[0])**2 + (vehicle_pos[1] - task_pos[1])**2
    )

#--------------------
# state representation
#---------------------

def get_state(vehicle, task):
    distance = calculate_distance(
        vehicle['Vehicle Position (x, y)'],
        task['Task Position (x, y)']
    )

    distance_bin = 0 if distance < 30 else (1 if distance < 70 else 2)
    battery_bin = 0 if vehicle['Battery Level (%)'] < 30 else (1 if vehicle['Battery Level (%)'] < 70 else 2)
    load_bin = 0 if num_waiting_tasks < 10 else (1 if num_waiting_tasks < 25 else 2)

    return (task['Urgency'], distance_bin, battery_bin, load_bin)

#---------------------
# q-table helpers
#---------------------

def get_q(state, action):
    return Q.get((state, action), 0.0)

def chose_action(task, vehicles):
    #exploration
    if np.random.rand() < epsilon:
        return random.choice(vehicles)
    
    q_values = []
    for v in vehicles:
        state = get_state(v, task)
        q_values.append(get_q(state, v['Vehicle ID']))
    
    return vehicles[np.argmax(q_values)]


def update_q(state, action, reward, next_state, vehicles):
    max_future_q = max([get_q(next_state, v['Vehicle ID']) for v in vehicles])
    current_q = get_q(state, action)

    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
    Q[(state, action)] = new_q


#-------------------
# reward function
#-------------------

def compute_reward(vehicle, task, travel_time, engagement_time):
    if vehicle['Battery Level (%)'] < engagement_time:
        return -100
    
    return(
        15 * task['Urgency']
        - 0.8 * travel_time
        - 0.5 * engagement_time
        + 0.3 * vehicle['Battery Level (%)']
        )

#--------------------------
# main allocation function
#---------------------------

def q_learning_allocation(vehicles_df, tasks_df):

    allocations = {}
    engagement_details = []

    tasks_df = tasks_df.sort_values('Urgency', ascending=False)

    for _,task in tasks_df.iterrows():

        available_vehicles = [
            v for _, v in vehicles_df.iterrows() if not v['Busy']
        ]

        if not available_vehicles:
            continue
        
        # build states for all vehicles
        states = [get_state(v, task) for v in available_vehicles]

        # chose action vehicle
        chosen_vehicle = chose_action(task, available_vehicles)

        # compute metrics
        distance = calculate_distance(
            chosen_vehicle['Vehicle Position (x, y)'], task['Task Position (x, y)']
        )

        travel_time = distance / chosen_vehicle['Speed']
        engagement_time = task['Duration (min)'] + travel_time

        
        #compute reward
        reward = compute_reward(chosen_vehicle, task, travel_time, engagement_time)


        #update Q
        state = get_state(v, task, len(tasks_df))
        next_state = state #simple assumption
        update_q(state, chosen_vehicle['Vehicle ID'], reward, next_state, available_vehicles)


        # apply allocation iff feasible
        if reward > -100:

            allocations[task['Task ID']] = chosen_vehicle['Vehicle ID']

            vehicle_idx = vehicles_df.index[
                vehicles_df['Vehicle ID'] == chosen_vehicle['Vehicle ID']
            ][0]

            vehicles_df.at[vehicle_idx, 'Battery Level (%)'] -= engagement_time
            vehicles_df.at[vehicle_idx, 'Busy'] = True
            vehicles_df.at[vehicle_idx, 'Remaining Duration'] = engagement_time
            vehicles_df.at[vehicle_idx, 'Vehicle Position (x, y)'] = task['Task Position (x, y)']

            engagement_details.append({
                "task_id": task['Task ID'],
                "task_duration": task['Duration (min)'],
                "energy_consumed": engagement_time,
                "engagement_time": engagement_time,
                "normalized_engagement_time": engagement_time/task['Duration (min)'],
                "travel_time": travel_time,
                "reward": reward
            })
    global epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    
    return allocations, engagement_details


