import numpy as np

r0 = 0.08 # dicount rate
nyear = 20 # the value decreasing over each year
cg = 1000000 # station build price 
phi = 10000 # equipment cost coefficient (10,000 CNY/UNIT^2)
epsilon = 100000 # price of the charger
gamma = 0.1 # maintenance factor
lam = 1.79 # cost per km
n_char = 10 # number of plugs in the station (can change on the way), initializing the varibles according to the document
deg_to_km = 100 # we take it like 1 degree is 100km

def calculate_fitness(stations, demand_points):
# finding Fco(construction cost)
    capital_recovery = (r0 * (1 + r0)**nyear) / ((1 + r0)**nyear - 1) # tells us how much of the loan we pay back each year over 20 years.

    AC_one = capital_recovery * (cg + phi * (n_char**2) + epsilon * n_char) # construcation cost for one station 
    AO_one = (cg + phi * (n_char**2) + epsilon * n_char) * gamma # maintanace cost for one station
    
    num_stations = len(stations)
    F_CO = num_stations * (AC_one + AO_one) # find out how many stations we have and multiply with single cost to get total Fco

# finding Ftime(how much it cost drive to station )
    total_distance_km = 0

    diff = demand_points[:, np.newaxis, :] - stations[np.newaxis, :, :] # demand points shape (130, 2), stations shape (14, 2), by adding np.newaxis we strerch them into 3d cubes so we can subtruct them all at once so diff is (130, 14, 2)

    dists_deg = np.sqrt(np.sum(diff**2, axis = 2)) # We calculate the Euclidean distance (a^2 + b^2 = c^2) in degrees, we use axis 2 to make 130 + 4 
    
    dists_km = dists_deg * deg_to_km # we multiply degree with 100km to fşnd distance
    
    min_dists = np.min(dists_km, axis=1) # for every demand point we will find closest station
    
    F_Time = 365 * np.sum(min_dists) * lam # we add up all those distance and multipy by 365 than again multy by 1.79 to find yearly travel cost

# finding Flimit
    violation = 0 
    
    for i in range(num_stations): # setting a double loop to compare evry station to other stations as distance 
        for j in range(i + 1, num_stations):
            dist_deg = np.sqrt(np.sum((stations[i] - stations[j])**2))
            dist_km = dist_deg * deg_to_km # calcularing and converting the distance 
            
            if dist_km < 6.0: # this check for the rule, as we know if they are closer than 6 km we will give + to violation counter
                violation += 1 
                
    Cc = 0 
    F_Limit = (violation / 2) * 15000 + Cc * 10000 # and calculating the fine 

    F_total = F_CO + F_Time + F_Limit # we add all cost to get to the fitness value
    
    return F_total # this is the number the whale algorithm will try to minimize


# check for if its working or not
mock_stations = np.random.rand(12, 2) 

mock_demand_points = np.random.rand(160, 2)

if __name__ == "__main__":
    print(calculate_fitness(mock_stations, mock_demand_points))