import csv
import json
from datetime import datetime

# Load passenger and driver data from CSV
passengers = []
drivers = []
with open('passengers.csv', 'r') as passenger_file:
    passenger_reader = csv.reader(passenger_file)
    for row in passenger_reader:
        passengers.append(row)
with open('drivers.csv', 'r') as driver_file:
    driver_reader = csv.reader(driver_file)
    for row in driver_reader:
        drivers.append(row)
# Load adjacency data from JSON
with open('adjacency.json', 'r') as adjacency_file:
    adjacency_data = json.load(adjacency_file)
# Load node data from JSON
with open('node_data.json', 'r') as node_data_file:
    node_data = json.load(node_data_file)

unmatched_passengers = []  # List of unmatched passengers
available_drivers = []  # List of available drivers
matched_pairs = []  # List of matched driver-passenger pairs


# Define the current time (you should replace this with your actual current time)
current_time_str = "04/25/2014 01:00:00"
current_time = datetime.strptime(current_time_str, "%m/%d/%Y %H:%M:%S")
# Sample passenger data (replace this with your actual passenger data)
passenger_data = {
    "appearance_time_str": "04/25/2014 00:30:00"
}

def calculate_waiting_time(passenger):
    # Convert the appearance time from string to datetime
    appearance_time = datetime.strptime(passenger["appearance_time_str"], "%m/%d/%Y %H:%M:%S")
    # Calculate the waiting time in minutes
    waiting_time = (current_time - appearance_time).total_seconds() / 60
    return waiting_time

# Calculate waiting time for the sample passenger
waiting_time_for_sample_passenger = calculate_waiting_time(passenger_data)
print(f"Waiting time for the passenger: {waiting_time_for_sample_passenger} minutes")
def match_passenger_to_driver():
    if unmatched_passengers and available_drivers:
        # Calculate waiting time for all unmatched passengers
        waiting_times = [calculate_waiting_time(passenger) for passenger in unmatched_passengers]
        # Find the passenger with the longest waiting time
        index_of_longest_waiting_passenger = waiting_times.index(max(waiting_times))
        longest_waiting_passenger = unmatched_passengers[index_of_longest_waiting_passenger]
        # Assign the first available driver
        driver = available_drivers.pop(0)
        # Update the matched pairs and remove the passenger from unmatched passengers
        matched_pairs.append((driver, longest_waiting_passenger))
        print(matched_pairs)
        unmatched_passengers.remove(longest_waiting_passenger)

        # You can also update other metrics and perform any necessary actions
# Example usage:
# You need to define unmatched_passengers, available_drivers, and calculate_waiting_time functions
# based on your data and application logic.
# Call the match_passenger_to_driver function
match_passenger_to_driver()


class Passenger:
    def __init__(self, appearance_time_str, source_lat, source_lon, destination_lat, destination_lon):
        self.appearance_time_str = appearance_time_str
        self.source_lat = source_lat
        self.source_lon = source_lon
        self.destination_lat = destination_lat
        self.destination_lon = destination_lon


class Driver:
    def __init__(self, appearance_time_str, current_lat, current_lon):
        self.appearance_time_str = appearance_time_str
        self.current_lat = current_lat
        self.current_lon = current_lon
        


# Initialize data structures
unmatched_passengers = []
available_drivers = []

# Simulate passenger appearances
for passenger_data in passengers:
    appearance_time_str, source_lat, source_lon, destination_lat, destination_lon = passenger_data
    passenger = Passenger(appearance_time_str, source_lat, source_lon, destination_lat, destination_lon)
    unmatched_passengers.append(passenger)

# Simulate driver appearances
for driver_data in drivers:
    appearance_time_str, current_lat, current_lon = driver_data
    driver = Driver(appearance_time_str, current_lat, current_lon)
    available_drivers.append(driver)


def handle_driver_exit(driver, passenger):
    # Implement the logic for driver exit
    # You should define your exit logic here, which can include:
    # - Calculating whether the driver wants to continue or exit based on certain criteria.
    # - Adding the driver back to available drivers if they choose to continue.
    # - Handling other relevant actions or updates based on your application's rules.

    # For demonstration purposes, let's assume the driver decides to continue working.
    # In this case, we add the driver back to the available drivers list.

    available_drivers.append(driver)  # Assuming this is the list of available drivers

    # You can also perform other necessary actions or updates here.



# Function to calculate D1: Average time for passengers to be dropped off
def calculate_D1(matched_pairs):
    total_dropoff_time = 0
    for pair in matched_pairs:
        passenger = pair["passenger"]
        driver = pair["driver"]
        # Calculate the time it takes for the driver to reach the passenger and then to the drop-off location
        dropoff_time = calculate_dropoff_time(passenger, driver)
        total_dropoff_time += dropoff_time
    return total_dropoff_time / len(matched_pairs) if matched_pairs else 0

# Function to calculate D2: Ride profit for drivers
def calculate_D2(matched_pairs):
    total_profit = 0
    for pair in matched_pairs:
        passenger = pair["passenger"]
        driver = pair["driver"]
        # Calculate ride profit based on time spent driving passengers and time spent driving to pickups
        profit = calculate_ride_profit(passenger, driver)
        total_profit += profit
    return total_profit

# Function to calculate D3: Empirical efficiency and scalability
def calculate_D3(execution_time):
    # Calculate and return D3 based on the execution time of the algorithm
    return execution_time

# Sample functions to calculate drop-off time and ride profit (replace with actual logic)
def calculate_dropoff_time(passenger, driver):
    # Calculate drop-off time based on passenger and driver information
    pass

def calculate_ride_profit(passenger, driver):
    # Calculate ride profit based on passenger and driver information
    pass

# Example usage:
D1 = calculate_D1(matched_pairs)
D2 = calculate_D2(matched_pairs)
D3 = calculate_D3(execution_time)

# Print or log the calculated metrics
print(f"D1: {D1}")
print(f"D2: {D2}")
print(f"D3: {D3}")

# Define a function to run experiments
def run_experiments():
    # Initialize a list to store the performance metrics for each experiment
    experiment_results = []

    # Loop through different scenarios (you can customize the scenarios)
    for scenario in scenarios:
        # Simulate events based on the scenario, create matched_pairs, and calculate execution time
        matched_pairs = simulate_events(scenario)
        execution_time = measure_execution_time()

        # Calculate performance metrics for this experiment
        D1 = calculate_D1(matched_pairs)
        D2 = calculate_D2(matched_pairs)
        D3 = calculate_D3(execution_time)

        # Record the metrics for this experiment
        experiment_result = {
            "scenario": scenario,
            "D1": D1,
            "D2": D2,
            "D3": D3
        }

        experiment_results.append(experiment_result)

    return experiment_results

# Sample scenarios (replace with your actual scenarios)
scenarios = [
    {"scenario_name": "Scenario 1", "passengers": [], "drivers": []},
    {"scenario_name": "Scenario 2", "passengers": [], "drivers": []},
    # ... add more scenarios with different data
]

# Call the run_experiments function
results = run_experiments()

# Print or log the results for analysis
for result in results:
    print(f"Scenario: {result['scenario_name']}")
    print(f"D1: {result['D1']}")
    print(f"D2: {result['D2']}")
    print(f"D3: {result['D3']}")