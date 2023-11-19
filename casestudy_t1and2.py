import csv
import json
from datetime import datetime, timedelta
from queue import Queue, PriorityQueue
import time
import math
import pandas as pd
from collections import deque

'''
Questions
1. Can we use Python's given implementations of a queue?

'''

'''
Gameplan
1. Use queues to keep track of recently added drivers, riders
2. We need to keep track of time somehow - I think the passengers, drivers arrays are great to start, just need a good way to introduce new drivers,
new passengers, etc. as time progresses
3. I like the classes for drivers, passengers - maybe make one for a pairing, make it easy to track all the stats we need?
4. 
'''

class Passenger:
    def __init__(self, appear_time, s_lat, s_lon, d_lat, d_lon):
        self.appear_time = appear_time
        self.s_lat = s_lat
        self.s_lon = s_lon
        self.d_lat = d_lat
        self.d_lon = d_lon

class Driver:
    def __init__(self, appear_time, cur_lat, cur_lon, avail):
        self.appear_time = appear_time
        self.cur_lat = cur_lat
        self.cur_lon = cur_lon
        self.avail = avail


class Node:
    def __init__(self, node_id, lat, lon):
        self.node_id = node_id
        self.lat = lat
        self.lon = lon

class Edge:
    def __init__(self, start_id, end_id, length, speeds):
        self.start_id = start_id
        self.end_id = end_id
        self.length = length
        self.speeds = speeds  # Dictionary of speeds for each hour

class Graph:
    def __init__(self):
        self.nodes = {}  # Dictionary: Node ID -> Node Object
        self.edges = {}  # Dictionary: Node ID -> List of Edges

    def load_nodes(self, json_file):
        with open(json_file, 'r') as file:
            nodes = json.load(file)
        for node_id, coords in nodes.items():
            #print(type(node_id))
            self.nodes[node_id] = Node(node_id, coords['lat'], coords['lon'])

    def load_edges(self, csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            start_id_str = str(int(row['start_id']))
            end_id_str = str(int(row['end_id']))

            speeds = row[3:].to_dict()
            #print(speeds)
            #print(type(str(row['start_id'])))
            edge = Edge(start_id_str, end_id_str, row['length'], speeds)
            self.edges.setdefault(start_id_str, []).append(edge)
            
    def dijkstra(self, start, goal, hour):
        assert(type(start) == str)
        assert(type(goal) == str)
        open_set = PriorityQueue()
        open_set.put((0, start))

        gScore = {node: float('inf') for node in self.nodes}
        gScore[start] = 0

        while not open_set.empty():
            current_cost, current_node = open_set.get()

            if current_node == goal:
                #print(60*gScore[current_node])
                return gScore[current_node]  # Return the cost to reach the goal
            #print("hi")
            for edge in self.edges.get(current_node, []):
                #print(edge.end_id)
                neighbor = edge.end_id

                speed = edge.speeds["weekday_" + str(hour)]
                #print("SPEED:" + str(speed))
                #print("DIST: " + str(edge.length))
                time_to_traverse = edge.length / speed
                tentative_gScore = gScore[current_node] + time_to_traverse

                if tentative_gScore < gScore[neighbor]:
                    gScore[neighbor] = tentative_gScore
                    open_set.put((tentative_gScore, neighbor))

        # If the goal is not reachable, return None or an indicative value
        return None

    def a_star(self, start, goal):
        # Implement A* Algorithm
        pass

# count=0
with open("node_data.json", 'r') as file:
    node_data = json.load(file)

graph = Graph()

graph.load_nodes("node_data.json")
graph.load_edges("edges.csv")

#print(graph.edges)

def main():
    # Load passenger and driver data from CSV
    passengers = read_csv("passengers.csv")
    drivers = read_csv("drivers.csv")

    p_idx = 1
    d_idx = 1

    trips_completed = 0             # Number of trips completed by driver and passenger combos
    
    # Total number of minutes drivers spend driving passengers from pickup to drop-off locations minus 
    # the total number of minutes they spend driving to pickup passengers
    driver_ride_profit = 0      

    # Total time between when passengers appear as an unmatched passenger, and when they are dropped off at their destination    
    passenger_time_total = 0


    p_q = Queue(maxsize=5000)
    d_q = Queue(maxsize=500)
    ud_q = deque()

    while(p_idx < len(passengers)):
        print(p_idx, d_idx)
        if(d_idx >= len(drivers)):
            decide = 0
        else:
            decide = compare_time(passengers[p_idx][0],drivers[d_idx][0])
        if(decide == 0):
            # Add the new passenger into its queue
            appear_time, s_lat, s_lon, d_lat, d_lon = passengers[p_idx]
            new_passenger = Passenger(appear_time, s_lat, s_lon, d_lat, d_lon)
            p_q.put(new_passenger)
            p_idx += 1
        else:
            # Add the new driver into its queue
            appear_time, cur_lat, cur_lon = drivers[d_idx]
            new_driver = Driver(appear_time, cur_lat, cur_lon, True)
            d_q.put(new_driver)
            d_idx += 1
        
        cur_time = appear_time

        # Now, we've grabbed the newest passenger/driver - let's see if we can match a driver passenger pair together
        if(not p_q.empty() and not d_q.empty()):
            # There is a pair!
            # For now, we just match the driver and passenger who have been waiting the longest together
            # Soon, we're going to have to update this criteria

            #################################################################################################################################
            # We can get all of the info we need from this point -- when the passenger first requested a ride, how long it's going to take  #
            # the driver to get from the driver's location to the passenger's location, and how long the trip will take                     #
            #################################################################################################################################

            # Passenger and Driver pair
            passenger = p_q.get()
            driver = d_q.get()

            # One more trip done
            trips_completed += 1
            
            # Parse the string into a datetime object
            datetime_cur = datetime.strptime(cur_time, "%m/%d/%Y %H:%M:%S")
            hour = datetime_cur.hour

            # Calculate the time it'll take for the driver to get to the passenger
            d_time_to_pass = pos_to_time(driver.cur_lat, driver.cur_lon, passenger.s_lat, passenger.s_lon, hour)
            print("DRIVER TO PASSENGER: " + str(d_time_to_pass))

            # Calculate the time it'll take for the pair to reach their destination
            ride_time = pos_to_time(passenger.s_lat, passenger.s_lon, passenger.d_lat, passenger.d_lon, hour)

            print("RIDE TIME: " + str(ride_time))
            # Increment total driver ride profit
            driver_ride_profit = driver_ride_profit + ride_time - d_time_to_pass
            
            # Increment passenger's total time from ordering the NUber to arriving at destination
            passenger_time_total = passenger_time_total + get_passenger_total(passenger.appear_time, cur_time, ride_time)

            # Still need to...
            # 1. Sort out how we get drivers back onto the queue
            # 2. Implement A* instead of Dijkstra's
            # 3. Right now I have HOURS coming out of djikstra's -- need to convert to mins
            # 4. driver.appear_time needs to be cur_time plus d_time_to_pass plus ride_time
            # (1, 3 and 4 might be done)
            # 5. Drivers hop offline after some amount of time -- figure out how to do that (random number between 30 mins and 3 hours?)
            #    This will also improve RT for T2, T3
            # 6. Implement "sectors" of the graph? (like Q1, Q2... on x,y axis)

            #print("CURRENT TIME: " + str(cur_time))
            #print("RIDE_TIME: " + str(ride_time))
            #print("DRIVER TO PASSENGER: " + str(d_time_to_pass))

            date_format = "%m/%d/%Y %H:%M:%S"
            dt = datetime.strptime(cur_time, date_format)

            # Add the minutes
            total_minutes = ride_time + d_time_to_pass
            new_dt = dt + timedelta(minutes=total_minutes)

            # Format the new datetime back into a string
            new_driver_str = new_dt.strftime(date_format)

            #print("NEW TIME: " + new_driver_str)

            driver.appear_time = new_driver_str
            ud_q.append(driver)
            n = 0
            while(n < len(ud_q)):
                if(compare_time(ud_q[n].appear_time, cur_time) == 1):
                    ready_driver = ud_q.popleft()
                    d_q.put(ready_driver)
                    n += 1
                else:
                    break

    print("Driver Ride Profit: ", driver_ride_profit/trips_completed)
    print("Passenger Total Time: ", passenger_time_total/trips_completed)

def get_passenger_total(p_at, cur_time, ride_time):
    # LIGHTLY TESTED but pretty sure it works

    format_str = "%m/%d/%Y %H:%M:%S"
    p_at_datetime = datetime.strptime(p_at, format_str)
    cur_time_datetime = datetime.strptime(cur_time, format_str)

    # Add the ride time to the current time using timedelta
    total_time = cur_time_datetime + timedelta(minutes=ride_time)

    # Calculate the time difference
    time_difference = total_time - p_at_datetime
    
    # Return the total difference in minutes
    return int(time_difference.total_seconds()/60)


def pos_to_time(src_lat, src_lon, dst_lat, dst_lon, hour):
    # Given two sets of coordinates, return the estimated time to get from src to dst

    # global count
    # count += 1

    src_node, src_dist = get_node(src_lat, src_lon)
    dst_node, dst_dist = get_node(dst_lat, dst_lon)

    # if(count % 100 == 0):
    #     print("Distance from src to node: " + str(src_dist))
    #     print("Distance from dst to node: " + str(dst_dist))
    #print(src_node)
    #print(dst_node)
    #print(src_node)
    #print(dst_node)
    global graph
    return 60*graph.dijkstra(src_node, dst_node, hour)

def get_node(lat, lon):
    # Given a latitude and longitude, find the closest node to those coords
    # Maybe we implement a binary search of sorts: sort the node_data (by lat? lon?) so that we get O(log(N)) RT, not O(N) RT
    max_distance = .001
    
    for node, coords in node_data.items():
        distance = euclidean_distance(float(lon), float(lat), coords['lon'], coords['lat'])
        if distance < max_distance:
            return node, distance
    #print(count)

    closest_node = None
    min_distance = float('inf')

    for node, coords in node_data.items():
        distance = euclidean_distance(float(lon), float(lat), coords['lon'], coords['lat'])
        if distance < min_distance:
            min_distance = distance
            closest_node = node
    print("MINIMUM: " + str(min_distance))
    return closest_node, min_distance

def euclidean_distance(lon1, lat1, lon2, lat2):
    return math.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)


def compare_time(p_time, d_time):
    # Decides whether we should add a new passenger or driver to their respective queues
    format_str = "%m/%d/%Y %H:%M:%S"
    p_dt = datetime.strptime(p_time, format_str)
    d_dt = datetime.strptime(d_time, format_str)

    # Compare the datetime objects
    if p_dt < d_dt:
        return 0
    else:
        return 1

def read_csv(csv_name):
    ret = []
    with open(csv_name, 'r') as file:
        file_reader = csv.reader(file)
        for row in file_reader:
            ret.append(row)
    return ret
    
if __name__ == "__main__":
    print("START")
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The function took {elapsed_time} seconds to execute.")
