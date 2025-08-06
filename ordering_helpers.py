from ortools.linear_solver import pywraplp
from itertools import combinations, permutations
import numpy as np
import pandas as pd
import pickle
import networkx as nx
import ast
import random

# Define generic functions for ordering analysis

def get_trait_list(node, trait_name):
    if trait_name in node and node[trait_name] is not None and node[trait_name] != "" and not pd.isna(node[trait_name]):
        if isinstance(node[trait_name], str) and node[trait_name][:1] == "[" and node[trait_name][-1] == "]":
            traits = ast.literal_eval(node[trait_name])
        elif isinstance(node[trait_name], list):
            traits = node[trait_name]
        else:
            traits = [node[trait_name]]
    else:
        traits = []
    return traits

def get_trait_counter(G, trait_name, print_output=False):
    trait_counter = {}

    # Iterate over each node in the original graph G
    counter = 0
    for node, data in G.nodes(data=True):
        counter += 1
        if counter % 1000 == 0 and print_output:
            print(f"Processed {counter} nodes")
        if trait_name in data and data[trait_name] is not None:
            traits = get_trait_list(data, trait_name)
            #print(traits)
            #if isinstance(data[trait_name], str):
            #    traits = ast.literal_eval(data[trait_name])
            #elif isinstance(data[trait_name], list):
            #    traits = data[trait_name]
            #else:
            #    traits = [data[trait_name]]
            for trait in traits:
                if trait in trait_counter:
                    trait_counter[trait] += 1
                else:
                    trait_counter[trait] = 1

    return trait_counter

def get_trait_graph(G, trait_name, trait_counter=None):
    trait_graph = nx.DiGraph()

    # Log transitiosn for additions
    total_added_transitions = 0
    total_removed_transitions = 0

    # Iterate over each edge in the original graph G
    for parent, child, data in G.edges(data=True):
        if trait_name in G.nodes[parent] and G.nodes[parent][trait_name] is not None:
            parent_traits = get_trait_list(G.nodes[parent], trait_name)#ast.literal_eval(G.nodes[parent][trait_name])
        else:
            continue
        if trait_name in G.nodes[child] and G.nodes[child][trait_name] is not None:
            child_traits = get_trait_list(G.nodes[child], trait_name)#ast.literal_eval(G.nodes[child][trait_name])
        else:
            continue
        
        # Log transitions for additions
        added_traits = set(child_traits) - set(parent_traits)
        for added_trait in added_traits:
            for parent_trait in parent_traits:
                if trait_graph.has_edge(parent_trait, added_trait):
                    trait_graph[parent_trait][added_trait]['transitions'] += 1  
                    trait_graph[parent_trait][added_trait]['added_transitions'] += 1
                    total_added_transitions += 1
                else:
                    trait_graph.add_edge(parent_trait, added_trait, transitions=1, added_transitions=1, removed_transitions=0)

        # Log transitions for subtractions
        removed_traits = set(parent_traits) - set(child_traits)
        for removed_trait in removed_traits:
            for remaining_trait in child_traits:
                if trait_graph.has_edge(removed_trait, remaining_trait):
                    trait_graph[removed_trait][remaining_trait]['transitions'] += 1
                    trait_graph[removed_trait][remaining_trait]['removed_transitions'] += 1
                    total_removed_transitions += 1
                else:
                    trait_graph.add_edge(removed_trait, remaining_trait, transitions=1, added_transitions=0, removed_transitions=1)

    if trait_counter is not None:
        for node in trait_graph.nodes():
            if node in trait_counter:
                trait_graph.nodes[node]['total_appearances'] = trait_counter[node]
            else:
                print(f"WARNING: Node {node} not found in trait_counter")
                trait_graph.nodes[node]['total_appearances'] = 0
    
    return trait_graph

def append_total_appearances(trait_graph, trait_counter):
    for node in trait_graph.nodes():
        if node in trait_counter:
            trait_graph.nodes[node]['total_appearances'] = trait_counter[node]
        else:
            print(f"WARNING: Node {node} not found in trait_counter")
            trait_graph.nodes[node]['total_appearances'] = 0

    return trait_graph

def get_trait_ratios(trait_graph, print_ratios=False):
    trait_ratios = []

    # Generate all combinations of traits
    unique_traits = list(trait_graph.nodes())
    trait_combinations = list(combinations(unique_traits, 2))

    for source_trait, destination_trait in trait_combinations:
        n_transitions = trait_graph[source_trait][destination_trait]['transitions'] if trait_graph.has_edge(source_trait, destination_trait) else 0
        if source_trait == destination_trait:
            if print_ratios:
                print(f"Trait {source_trait} to {destination_trait}: {n_transitions} <-> {n_transitions}")
        else:
            reverse_transitions = trait_graph[destination_trait][source_trait]['transitions'] if trait_graph.has_edge(destination_trait, source_trait) else 0
            if n_transitions + reverse_transitions > 0:
                if print_ratios:
                    print(f"Trait {source_trait} to {destination_trait}: {n_transitions} <-> {reverse_transitions}")
                ratio = n_transitions / (n_transitions + reverse_transitions)
                trait_ratios.append(ratio)
                if print_ratios:
                    print(ratio)

    return trait_ratios

def get_oriented_trait_graph(trait_graph):
    oriented_trait_graph = nx.DiGraph()

    # Iterate over all combinations of traits
    unique_traits = list(trait_graph.nodes())
    trait_combinations = list(combinations(unique_traits, 2))

    for source_trait, destination_trait in trait_combinations:
        n_transitions = trait_graph[source_trait][destination_trait]['transitions'] if trait_graph.has_edge(source_trait, destination_trait) else 0
        reverse_transitions = trait_graph[destination_trait][source_trait]['transitions'] if trait_graph.has_edge(destination_trait, source_trait) else 0
        
        # Calculate traffic and total_n
        traffic = n_transitions - reverse_transitions
        total_n = n_transitions + reverse_transitions
        
        # Determine the direction of the edge based on traffic
        if traffic > 0:
            # More transitions from source to destination
            oriented_trait_graph.add_edge(source_trait, destination_trait, traffic=traffic, total_n=total_n)
        elif traffic < 0:
            # More transitions from destination to source
            oriented_trait_graph.add_edge(destination_trait, source_trait, traffic=-traffic, total_n=total_n)

    return oriented_trait_graph

def get_top_trait_graph(trait_graph, n, remove_other=True):
    top_trait_graph = nx.DiGraph()

    # Get the top-n traits
    top_traits = sorted(trait_graph.nodes(), key=lambda x: trait_graph.nodes[x]['total_appearances'], reverse=True)[:min(n+5, len(trait_graph.nodes()))]

    if remove_other:
        # Remove all traits equal to "other", "unknown", "None", "", or "[]". Count how many are removed
        removed_traits = [trait for trait in top_traits if trait in ["other", "unknown", "None", "", "[]"]]
        print(f"Removed {len(removed_traits)} traits: {removed_traits}")
        top_traits = [trait for trait in top_traits if trait not in removed_traits]
    top_traits = top_traits[:n]

    # Create a subgraph with only the top traits
    top_trait_graph = trait_graph.subgraph(top_traits).copy()

    return top_trait_graph
    

def solve_max_compatible_ordering(graph):
    nodes = list(graph.nodes)
    edges = list(graph.edges)

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if solver is None:
        raise RuntimeError("Solver not available.")

    # Define x[u,v] = 1 if u comes before v
    x = {}
    for u in nodes:
        for v in nodes:
            if u != v:
                x[(u, v)] = solver.IntVar(0, 1, f'x_{u}_{v}')

    # Antisymmetry: x_uv + x_vu = 1
    for u, v in combinations(nodes, 2):
        solver.Add(x[(u, v)] + x[(v, u)] == 1)

    # Transitivity: x_uv + x_vw + x_wu ≤ 2
    for u, v, w in permutations(nodes, 3):
        if u != v and v != w and w != u:
            solver.Add(x[(u, v)] + x[(v, w)] + x[(w, u)] <= 2)

    # Objective: Maximize number of compatible edges
    objective = solver.Objective()
    for u, v in edges:
        if u != v:
            objective.SetCoefficient(x[(u, v)], 1)
    objective.SetMaximization()

    # Solve the ILP
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("No optimal solution found.")

    # Recover total order: count how many nodes come before each node
    before_count = {u: 0 for u in nodes}
    for u in nodes:
        for v in nodes:
            if u != v and x[(u, v)].solution_value() == 1:
                before_count[v] += 1

    ordering = sorted(nodes, key=lambda u: before_count[u])
    compatible_count = int(objective.Value())

    return ordering, compatible_count

def get_violating_edges(graph, ordering):
    # Map each node to its index in the ordering
    position = {node: idx for idx, node in enumerate(ordering_unweighted)}

    # A violating edge goes from a later node to an earlier node in the order
    violating_edges = [
        (u, v) for u, v in graph.edges
        if position[u] > position[v]
    ]
    return violating_edges


def solve_weighted_compatible_ordering(graph):
    nodes = list(graph.nodes)
    edges = list(graph.edges(data='traffic', default=1.0))  # default weight is 1.0

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if solver is None:
        raise RuntimeError("Solver not available.")

    # Decision variables: x[u,v] = 1 if u comes before v
    x = {}
    for u in nodes:
        for v in nodes:
            if u != v:
                x[(u, v)] = solver.IntVar(0, 1, f'x_{u}_{v}')

    # Antisymmetry constraints: x_uv + x_vu = 1
    for u, v in combinations(nodes, 2):
        solver.Add(x[(u, v)] + x[(v, u)] == 1)

    # Transitivity constraints: x_uv + x_vw + x_wu ≤ 2
    for u, v, w in permutations(nodes, 3):
        if u != v and v != w and w != u:
            solver.Add(x[(u, v)] + x[(v, w)] + x[(w, u)] <= 2)

    # Objective: maximize total weight of compatible edges
    objective = solver.Objective()
    for u, v, weight in edges:
        if u != v:
            objective.SetCoefficient(x[(u, v)], weight)
    objective.SetMaximization()

    # Solve
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("No optimal solution found.")

    # Recover ordering from pairwise comparisons
    before_count = {u: 0 for u in nodes}
    for u in nodes:
        for v in nodes:
            if u != v and x[(u, v)].solution_value() == 1:
                before_count[v] += 1

    ordering = sorted(nodes, key=lambda u: before_count[u])
    total_weight = objective.Value()

    return ordering, total_weight

# Given the optimal_ordering and the trait graph, get the total traffic among all nodes in the ordering, and the total traffic among all nodes compatible with the ordering
def get_compatible_and_total_traffic(trait_graph, optimal_ordering):
    total_traffic = 0
    total_traffic_compatible_with_ordering = 0
    for i in range(len(optimal_ordering)):
        for j in range(i+1, len(optimal_ordering)):
            if trait_graph.has_edge(optimal_ordering[j], optimal_ordering[i]):
                total_traffic += trait_graph[optimal_ordering[j]][optimal_ordering[i]]['transitions']
            if trait_graph.has_edge(optimal_ordering[i], optimal_ordering[j]):
                total_traffic += trait_graph[optimal_ordering[i]][optimal_ordering[j]]['transitions']
                total_traffic_compatible_with_ordering += trait_graph[optimal_ordering[i]][optimal_ordering[j]]['transitions']
    return  total_traffic_compatible_with_ordering, total_traffic
