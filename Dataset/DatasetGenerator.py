import time
import random
import heapq  # Python's built-in heap implementation
import matplotlib.pyplot as plt
import numpy as np

# --- Dataset Generation and Storage ---

def generate_random_dataset(size, filename="random_data.txt"):
    """Generates a list of random integers and stores it in a text file."""
    data = [random.randint(0, size * 10) for _ in range(size)]  # Wider range
    with open(filename, "w") as f:
        for item in data:
            f.write(str(item) + "\n")
    return data

def generate_sorted_dataset(size, filename="sorted_data.txt"):
    """Generates a list of sorted integers and stores it in a text file."""
    data = list(range(size))
    with open(filename, "w") as f:
        for item in data:
            f.write(str(item) + "\n")
    return data

def generate_nearly_sorted_dataset(size, inversions=0.05, filename="nearly_sorted_data.txt"):
    """Generates a nearly sorted list and stores it in a text file."""
    data = list(range(size))
    num_inversions = int(size * inversions)
    for _ in range(num_inversions):
        i, j = random.sample(range(size), 2)
        data[i], data[j] = data[j], data[i]
    with open(filename, "w") as f:
        for item in data:
            f.write(str(item) + "\n")
    return data

# --- Main Execution & Plotting ---

if __name__ == "__main__":
    data_sizes = [100, 1000, 10000, 100000, 250000, 500000, 750000, 1000000]  # Adjust as needed
    #data_sizes = [100, 200, 300]

    # --- Generate and Store Datasets ---
    random_data = generate_random_dataset(max(data_sizes), "random_data.txt")
    sorted_data = generate_sorted_dataset(max(data_sizes), "sorted_data.txt")
    nearly_sorted_data = generate_nearly_sorted_dataset(max(data_sizes), filename="nearly_sorted_data.txt")
