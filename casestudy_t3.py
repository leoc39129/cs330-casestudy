import csv
import json
from datetime import datetime, timedelta
from queue import Queue, PriorityQueue
import time
import math
import heapq
import random
from collections import deque

'''
Questions

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
        with open(csv_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                start_id_str = str(int(row['start_id']))
                end_id_str = str(int(row['end_id']))

                # Extract the speeds starting from the fourth column
                speeds = {key: float(value) for key, value in row.items() if key not in ['start_id', 'end_id', 'length']}
                
                edge = Edge(start_id_str, end_id_str, float(row['length']), speeds)
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

    def heuristic(self, node1, node2):
        # Scaled Euclidean distance as a simple heuristic
        scale_factor = 1.5  # Adjust based on road network properties
        lat1, lon1 = self.nodes[node1].lat, self.nodes[node1].lon
        lat2, lon2 = self.nodes[node2].lat, self.nodes[node2].lon
        return scale_factor * math.sqrt((lon2 - lon1) ** 2 + (lat2 - lat1) ** 2)

    def a_star(self, start, goal, hour):
        open_set = []
        heapq.heappush(open_set, (0, start))

        gScore = {node: float('inf') for node in self.nodes}
        gScore[start] = 0

        fScore = {node: float('inf') for node in self.nodes}
        fScore[start] = self.heuristic(start, goal)

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                return 60*gScore[current]

            for edge in self.edges.get(current, []):
                neighbor = edge.end_id
                speed = edge.speeds["weekday_" + str(hour)]
                time_to_traverse = edge.length / speed
                tentative_gScore = gScore[current] + time_to_traverse

                if tentative_gScore < gScore[neighbor]:
                    gScore[neighbor] = tentative_gScore
                    fScore[neighbor] = tentative_gScore + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (fScore[neighbor], neighbor))

        return None  # Goal not reachable

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
    #d_q = Queue(maxsize=500)        # THIS LINE
    d_q = []
    ud_q = deque()

    while(p_idx <= len(passengers) or not p_q.empty()):
        print(p_idx, d_idx)
        # print("Passenger Queue Size: " + str(p_q.qsize()))
        # print("Driver Queue Size: " + str(d_q.qsize()))
        # print("Unavailable Driver Queue Size: " + str(len(ud_q)))
        # print("Trips Completed: " + str(trips_completed))
        # print()
        # print("p_q:")
        # print(len(p_q))
        # print("d_q:")
        # print(len(d_q))
        # print("ud_q:")
        # print(len(ud_q))

        #print("# OF UNAVAILABLE DRIVERS:")
        #print(len(ud_q))
        #print("\n")

        if(p_idx >= len(passengers)):
            #print("Decide=2")
            decide = 2
        elif(d_idx >= len(drivers)):
            decide = 0
        else:
            decide = compare_time(passengers[p_idx][0],drivers[d_idx][0])
        if(decide == 0):
            # Add the new passenger into its queue
            appear_time, s_lat, s_lon, d_lat, d_lon = passengers[p_idx]
            new_passenger = Passenger(appear_time, s_lat, s_lon, d_lat, d_lon)
            p_q.put(new_passenger)
            p_idx += 1
        elif(decide == 1):
            # Add the new driver into its queue
            appear_time, cur_lat, cur_lon = drivers[d_idx]
            new_driver = Driver(appear_time, cur_lat, cur_lon, True)
            #d_q.put(new_driver)     # THIS LINE
            d_q.append(new_driver)
            d_idx += 1
        
        cur_time = appear_time
        if(decide == 2):
            p_idx += 1
            incr = p_idx - len(passengers)

            # Once we're done with new passengers/drivers, we have to increment cur_time slowly to advance
            date_format = "%m/%d/%Y %H:%M:%S"
            cur_time_datetime = datetime.strptime(cur_time, date_format)

            # Add the minutes
            cur_time_new_dt = cur_time_datetime + timedelta(minutes=5*incr)

            # Format the new datetime back into a string
            cur_time = cur_time_new_dt.strftime(date_format)

        # Now, we've grabbed the newest passenger/driver - let's see if we can match a driver passenger pair together
        while(not p_q.empty() and len(d_q) > 0):
            # There is a pair!
            # For now, we just match the driver and passenger who have been waiting the longest together
            # Soon, we're going to have to update this criteria

            #################################################################################################################################
            # We can get all of the info we need from this point -- when the passenger first requested a ride, how long it's going to take  #
            # the driver to get from the driver's location to the passenger's location, and how long the trip will take                     #
            #################################################################################################################################

            # Passenger and Driver pair
            passenger = p_q.get()
            
            # T2: Loop over all driver's, pull driver with minimum euclidean distance to passenger #
            # T3: Run Dijkstra's/A* on each driver to find the closest available driver, rather than pulling the longest waiting driver #
            
            datetime_cur = datetime.strptime(cur_time, "%m/%d/%Y %H:%M:%S")
            hour = datetime_cur.hour

            #Assuming we have d_list (driver list) set up...
            min_dist_driver = float('inf')
            min_dist_driver_idx = 0
            for d in range(len(d_q)):
                cur_dist_driver = pos_to_time(float(d_q[d].cur_lon), float(d_q[d].cur_lat), float(passenger.s_lon), float(passenger.s_lat), hour)
                if cur_dist_driver < min_dist_driver:
                    min_dist_driver_idx = d
                    min_dist_driver = cur_dist_driver
            driver = d_q.pop(min_dist_driver_idx)
            
            
            #driver = d_q.get()

            # One more trip done
            trips_completed += 1
        
            # Calculate the time it'll take for the driver to get to the passenger
            d_time_to_pass = pos_to_time(driver.cur_lat, driver.cur_lon, passenger.s_lat, passenger.s_lon, hour)
            #print("DRIVER TO PASSENGER: " + str(d_time_to_pass))

            # Calculate the time it'll take for the pair to reach their destination
            ride_time = pos_to_time(passenger.s_lat, passenger.s_lon, passenger.d_lat, passenger.d_lon, hour)

            #print("RIDE TIME: " + str(ride_time))
            # Increment total driver ride profit
            driver_ride_profit = driver_ride_profit + ride_time - d_time_to_pass
            
            # Increment passenger's total time from ordering the NUber to arriving at destination
            passenger_time_total = passenger_time_total + get_passenger_total(passenger.appear_time, cur_time, ride_time)

            # Still need to...
            # 1. Implement "sectors" of the graph? (like Q1, Q2... on x,y axis) ( we can use this for T4(i) )
            #    a) We'll have to update the load_nodes function that's part of the Graph class
            #    b) We'll also have to update the get_node function to only look at nodes in the correct sector
            #    c) There might be some leakage if someone is close to the edge of the quadrants -- i.e. the closest node might
            #       be in Q3 when the longitude latitude given is in Q2, but that shouldn't happen often so this should work
            #    d) We'll have to update the node class to have a sector field as part of init

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
            driver.cur_lat = passenger.d_lat
            driver.cur_lon = passenger.d_lon

            ud_q.append(driver)
            n = 0
            while(n < len(ud_q)):
                #print("DRIVER NEXT AVAIL:")
                #print(ud_q[n].appear_time)
                #print("CUR TIME:")
                #print(cur_time)
                #print("COMPAREVAL:")
                #print(compare_time(ud_q[n].appear_time, cur_time))
                if(compare_time(ud_q[n].appear_time, cur_time) == 0):
                    ready_driver = ud_q.popleft()

                    # Generate a random number between 0 and 1
                    random_number = random.random()

                    # Driver is done taking rides 8% of the time -- eaech driver will average round 12-13 rides
                    #print("Random Number: " + str(random_number))
                    if random_number < 0.92:
                        d_q.append(ready_driver)
                    n += 1
                else:
                    break
        if(decide == 2):
            #print("No more new, getting unavail drivers back on d_q")
            n = 0
            while(n < len(ud_q)):
                #print("DRIVER NEXT AVAIL:")
                #print(ud_q[n].appear_time)
                #print("CUR TIME:")
                #print(cur_time)
                #print("COMPAREVAL:")
                #print(compare_time(ud_q[n].appear_time, cur_time))
                if(compare_time(ud_q[n].appear_time, cur_time) == 0):
                    ready_driver = ud_q.popleft()    
                    d_q.append(ready_driver)
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
    return graph.a_star(src_node, dst_node, hour)

def get_node(lat, lon):
    # Given a latitude and longitude, find the closest node to those coords
    # Maybe we implement a binary search of sorts: sort the node_data (by lat? lon?) so that we get O(log(N)) RT, not O(N) RT
    max_distance = .0015
    
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
    #print("MINIMUM: " + str(min_distance))
    return closest_node, min_distance

# MAYBE: getting rid of the euclidean_distance fn overhead might save lots of time -- test it once we've gotten through more stuff
# MEANING: instead of "distance = euclidean_distance(...)" just write "distance = math.sqrt(...)"

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