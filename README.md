# Location_Selection_Shenzhen
Optimization of EV Charging Station Locations using Improved Whale Optimization Algorithm (IWOA)

Course: ENS001 - Application Development for Optimization

Team: Group 6

Semester: Spring 2024
📌 Project Overview

This project addresses the facility location problem for Electric Vehicle (EV) Charging Stations in Shenzhen, China. The goal is to determine the optimal coordinates for 12 charging stations to serve 130 demand points while minimizing the Total Annual System Cost.

To solve this complex non-linear optimization problem, we implemented and deployed an Improved Whale Optimization Algorithm (IWOA). Our hybrid approach integrates chaos theory, reverse learning, and adaptive mechanisms to overcome the premature convergence issues found in the standard Whale Optimization Algorithm (WOA).

🎯 Problem Definition

The objective function minimizes the Comprehensive Cost (Ftotal​), which consists of:

    Construction & Operation Cost (FCO​): Land, equipment, and annual maintenance.

    Travel Cost (FT​): The cost of user time and energy to travel to the nearest station.

    Penalty Cost (FP​): Fines for violating constraints (e.g., stations too close together or capacity overflow).

Constraints:

    Minimum distance between stations ≥6 km.

    Service capacity limits.

    Every demand point must be assigned to the nearest station.

🧠 Methodology: Hybrid IWOA

We enhanced the standard WOA with four hybrid strategies to create IWOA:

    Circle & Tent Chaos Maps: Replaces random initialization with chaotic sequences to ensure a more homogeneous distribution of the initial population.

    Hybrid Reverse Learning (1:3:6 Rule): A layered strategy that preserves elite whales, conditionally improves average ones, and forces bad ones to explore the opposite search space.

    Adaptive Probability Threshold (adp_p): Replaces the fixed 50% coin toss with a dynamic sine-wave threshold, intelligently switching between "Hunting" and "Spiraling" behaviors.

    Nonlinear Convergence Factor (a): Uses a cosine wave to balance global exploration and local exploitation more effectively than a linear drop.

📊 Results & Performance

We validated the algorithm on 18 Benchmark Functions (F1–F18) and the real-world Shenzhen dataset.
Real-World Case Study (Shenzhen)

Our IWOA significantly outperformed the Standard WOA in the final simulation (100 iterations):

Metric,Standard WOA,Improved WOA (Ours)
Total Cost,"832,343 CNY","735,996 CNY"
Penalty Cost,"7,500 CNY (Invalid)",0.00 CNY (Valid)
Construction Cost,"823,557 CNY","734,742 CNY"
Travel Cost,"1,285 CNY","1,254 CNY"

Key Achievement: We saved approx. 100,000 CNY/year (~12%) compared to the standard algorithm and found a zero-penalty solution.
📂 File Structure

    iwoa.ipynb: Main implementation of the Improved Whale Optimization Algorithm.

    woa.ipynb: Implementation of the Standard WOA (for comparison).

    model.py: Contains the objective function, cost calculations, and constraints logic.

    benchmark.py: The suite of 18 mathematical functions (F1–F18) for unit testing the algorithm.

    visuals.py: Helper scripts for plotting convergence graphs and Voronoi maps.
