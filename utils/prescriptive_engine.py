import pulp
import pandas as pd
import numpy as np

class PrescriptiveEngine:
    """
    Prescriptive analytics engine using linear programming (PuLP)
    to optimize resource allocation and strategic decisions.
    """
    def __init__(self, story_name):
        self.story_name = story_name
        self.problem = None
        self.solution = None

    def solve_supply_chain_optimization(self, city_demand, warehouse_stock, shipping_costs):
        """
        Phase 5: Optimize allocation of stock to cities to minimize shipping costs
        under constraints of stock availability and demand satisfaction.
        """
        print(f"Running Prescriptive Optimization for {self.story_name}...")
        
        # Define the problem
        prob = pulp.LpProblem("SupplyChainOptimization", pulp.LpMinimize)
        
        # Decision Variables: X_ij = amount of stock sent from warehouse i to city j
        # For simplicity, we assume 1 warehouse for this demonstration
        cities = list(city_demand.keys())
        allocation = pulp.LpVariable.dicts("Alloc", cities, lowBound=0, cat='Continuous')
        
        # Objective Function: Minimize Total Cost
        prob += pulp.lpSum([allocation[city] * shipping_costs[city] for city in cities])
        
        # Constraints
        # 1. Total allocation <= Available stock
        prob += pulp.lpSum([allocation[city] for city in cities]) <= warehouse_stock
        
        # 2. Allocation for each city >= Demand (if possible)
        for city in cities:
            prob += allocation[city] >= city_demand[city]
            
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        self.solution = {city: pulp.value(allocation[city]) for city in cities}
        self.solution['status'] = pulp.LpStatus[prob.status]
        return self.solution

    def solve_startup_budget_optimization(self, budget, channels, expected_roi):
        """
        Phase 5: Optimize marketing/R&D budget across channels to maximize success probability
        """
        print(f"Running Startup Budget Optimization for {self.story_name}...")
        
        prob = pulp.LpProblem("StartupBudgetOptimization", pulp.LpMaximize)
        
        alloc = pulp.LpVariable.dicts("Invest", channels, lowBound=0, cat='Continuous')
        
        # Objective: Maximize Total ROI (or success probability proxy)
        prob += pulp.lpSum([alloc[ch] * expected_roi[ch] for ch in channels])
        
        # Constraint: Total investment <= Budget
        prob += pulp.lpSum([alloc[ch] for ch in channels]) <= budget
        
        # Specific constraints (e.g., at least 20% in R&D)
        if 'R&D' in channels:
            prob += alloc['R&D'] >= 0.2 * budget
            
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        self.solution = {ch: pulp.value(alloc[ch]) for ch in channels}
        self.solution['status'] = pulp.LpStatus[prob.status]
        return self.solution

if __name__ == "__main__":
    pass
