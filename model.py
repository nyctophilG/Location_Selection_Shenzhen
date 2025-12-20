import numpy as np

r0 = 0.08 # dicount rate
nyear = 20 # the value decreasing over each year
cg = 1000000 # station build price 
phi = 10000 # equipment cost coefficient (10,000 CNY/UNIT^2)
epsilon = 100000 # price of the charger
gamma = 0.1 # maintenance factor
lam = 1.79 # cost per km
n_char = 5 # number of plugs in the station (can change on the way), initializing the varibles according to the document
deg_to_km = 100 # we take it like 1 degree is 100km

def calculate_fitness(stations, demand_points):
# finding Fco(construction cost)
    num_stations = len(stations)

    # 1. Annualization Factor (Capital Recovery)
    # Applies to investment costs
    capital_recovery = (r0 * (1 + r0)**nyear) / ((1 + r0)**nyear - 1) # This is the standard "Capital Recovery Factor" (CRF). It calculates what percentage of your total loan you need to pay back each year to pay it off in 20 years.

    # 2. Variable Cost Per Station (Chargers + Equipment)
    # Linear (n_char), not Squared (n_char^2)
    variable_cost_per_station = (phi * n_char) + (epsilon * n_char) # You buy 10 chargers, you pay for 10. You don't pay for 100.
    
    # 3. Total Investment (Construction)
    # Cg is usually network-level added ONCE, not per station
    total_investment = cg + (num_stations * variable_cost_per_station) # Adds cg only once as a "Network Setup Cost" like buying a licance of software, one time thing
    
    # Annual Construction Cost
    AC_total = total_investment * capital_recovery

    # 4. Operation Cost (AO)
    # Usually a fraction of investment
    AO_total = total_investment * gamma

    F_CO = AC_total + AO_total

    # Travel & Penalty, this calculates how far every user drives to their nearest station
    diff = demand_points[:, np.newaxis, :] - stations[np.newaxis, :, :]
    dists_deg = np.sqrt(np.sum(diff**2, axis=2))
    min_dists = np.min(dists_deg * deg_to_km, axis=1)
    F_Time = 365 * np.sum(min_dists) * lam

    # Penalty Check
    violation = 0
    for i in range(num_stations):
        for j in range(i + 1, num_stations):
            if (np.sqrt(np.sum((stations[i] - stations[j])**2)) * deg_to_km) < 6.0:
                violation += 1
    F_Limit = (violation / 2) * 15000 # f stations are too close than 6km, add a fine

    return F_CO + F_Time + F_Limit