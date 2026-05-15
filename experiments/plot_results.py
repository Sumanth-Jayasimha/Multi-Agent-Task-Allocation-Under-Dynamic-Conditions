import matplotlib.pyplot as plt

# ===== YOUR AVERAGE RESULTS =====
greedy = {
    "num_tasks": 26.5,
    "throughput": 497.2,
    "avg_idle_time": 31.93,
    "energy_per_unit": 1.342
}

auction = {
    "num_tasks": 24.1,
    "throughput": 459.2,
    "avg_idle_time": 35.1,
    "energy_per_unit": 1.387
}

q_learning = {
    "num_tasks": 25.7,
    "throughput": 475.7,
    "avg_idle_time": 32.32,
    "energy_per_unit": 1.399
}


# ===== FUNCTION TO PLOT =====
def plot_metric(metric_name, ylabel):

    labels = ["Greedy", "Auction", "Q-learning"]

    values = [
        greedy[metric_name],
        auction[metric_name],
        q_learning[metric_name]
    ]

    plt.figure()
    plt.bar(labels, values)

    plt.title(metric_name.replace("_", " ").title())
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.show()


# ===== PLOTS =====
plot_metric("num_tasks", "Number of Tasks Completed")
plot_metric("throughput", "Total Task Duration")
plot_metric("avg_idle_time", "Average Idle Time")
plot_metric("energy_per_unit", "Energy per Unit Task")