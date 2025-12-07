import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d # these are standart tools for us, this lan is needed because we will use voroni diagrms to divide the service range of stations	

x_min, x_max = 113.8, 114.4
y_min, y_max = 22.45, 22.85 # this is the ranges given in the documents (fig 25)

np.random.seed(42) # this is for randomazing number but not typically randomization if our number change in every iteration our map will change too, so we give a seed, this makes map stand still be cause number will not cahnge every iterations
demand_points = np.column_stack(( # here we will creating cordinates for demand points

	np.random.uniform(x_min, x_max, 130),
	np.random.uniform(y_min, y_max, 130)
))

stations = np.column_stack(( # we will create stations and set it to 14 because documant says us its the optimum number of stations

	np.random.uniform(x_min, x_max, 14),
	np.random.uniform(y_min, y_max, 14)
)) 

vor = Voronoi(stations) # in here we are creating voronoi diagram

fig, ax = plt.subplots(figsize=(10, 8)) 
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_alpha=0.6)
ax.scatter(demand_points[:, 0], demand_points[:, 1], c='red', marker='*', label='Demand Point')
ax.scatter(stations[:, 0], stations[:, 1], c='blue', marker='o', label='Charging Station') # and in those lines we configure it like figure 26

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_title(f"Initial Random Map: 14 station & 130 demand point")
ax.legend()
plt.show() # lets see what we create