# fleet_aution_allocation
import numpy as np

def calculate_distance(vehicle_pos, task_pos):
    return np.sqrt(
        (vehicle_pos[0] - task_pos[0])**2 + (vehicle_pos[1] - task_pos[1])**2
    )

def compute_bid(vehicle, task):
    """
    compute auction bid score for a vehicle task pair
    Higher bid = better suitability
    """

    # calculate distance to task
    distance = calculate_distance(
        vehicle['Vehicle Position (x, y)'],
        task['Task Position (x, y)']
    )

    # travel time dependes on vehicle speed
    travel_time = distance / vehicle['Speed']

    # total engagement time
    engagement_time = task['Duration (min)'] + travel_time

    # check battey feasibility
    if vehicle['Battery Level (%)'] < engagement_time:
        return None
    
    if vehicle['Battery Level (%)'] <= 0:
        return -np.inf
    
    if vehicle['Battery Level (%)'] < 20:
        return None
    
    #remaining battery after completing task
    remaining_battery = (
        vehicle['Battery Level (%)'] - engagement_time
    )

    # extract task urgency
    urgency = task['Urgency']

    # extract vehicle speed
    speed = vehicle['Speed']

    # auction bid function
    bid = (
        10 * urgency + 0.2 * remaining_battery + 3.5 * vehicle['Idle Time'] + 0.5 * speed - 0.3 * travel_time
    )

    return bid


def auction_allocation(vehicles_df, tasks_df):
    """
    Auction based task allocation

    For each task:
    available vehicles bid and highest one wuns

    Returns:
    allocation: dictionary mapping task Ids to vehicle ids
    engagement details: lists of peer task metrics
    """

    allocations = {}

    engagement_details = []

    #sort tasks by urgency
    tasks_df = tasks_df.sort_values(
        'Urgency',
        ascending = False
    )

    #process each task
    for _, task in tasks_df.iterrows():

        best_vehicle = None
        best_bid = -np.inf
        best_distance = None
        best_travel_time = None

        #evaluate all vehicles
        for idx, vehicle in vehicles_df.iterrows():

            if vehicle['Busy']:
                continue
            
            bid = compute_bid(vehicle, task)

            if bid is None:
                continue


            #keep highest bid
            if bid > best_bid:
                
                best_bid = bid
                best_vehicle = vehicle

                #storing metrics for later
                distance = calculate_distance(
                    vehicle['Vehicle Position (x, y)'],
                    task['Task Position (x, y)']
                )

                best_distance = distance

                best_travel_time = (
                    distance/vehicle['Speed']
                )

        #allocate task to suitable vehicle
        if best_vehicle is not None:
            engagement_time = (
                task['Duration (min)'] + best_travel_time
                )

            #store allocation
            allocations[task['Task ID']] = best_vehicle['Vehicle ID']

            #finding vehicle index
            vehicle_idx = vehicles_df.index[vehicles_df['Vehicle ID'] == best_vehicle['Vehicle ID']][0]

            #update battery
            vehicles_df.at[vehicle_idx, 'Battery Level (%)'] = (float(vehicles_df.at[vehicle_idx, 'Battery Level (%)']) - engagement_time)


            #marking vehicle busy
            vehicles_df.at[ vehicle_idx, 'Busy'] = True

            #set remaining duration
            vehicles_df.at[ vehicle_idx, 'Remaining Duration'] = float(engagement_time)

            #update vehicle position
            vehicles_df.at[vehicle_idx, 'Vehicle Position (x, y)'] = task['Task Position (x, y)']

            #save matrics
            engagement_details.append({
                "task_id": task['Task ID'],

                "task_duration": task['Duration (min)'],

                "travel_time": best_travel_time,

                "engagement_time": engagement_time,

                "normalized_engagement_time": engagement_time/task['Duration (min)'],

                "energy_consumed": engagement_time
            })

    return allocations, engagement_details






