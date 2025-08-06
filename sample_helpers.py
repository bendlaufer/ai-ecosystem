import numpy as np
import networkx as nx
import random
import pandas as pd
import matplotlib.pyplot as plt
import math

def rando(graph, n, config=1):
    if config != 1:
        print("Invalid config")
        return []
    sampled_pairs = []
    
    # Select 2 nodes from the graph without replacement
    for _ in range(n):
        all_nodes = list(graph.nodes)
        sampled_nodes = random.sample(all_nodes, 2)
        sampled_pairs.append(list(sampled_nodes))

    return sampled_pairs, len(sampled_nodes)*(len(sampled_nodes)-1)

def duo(graph, n, config=1):
    if config != 1:
        print("Invalid config")
        return []
    sampled_pairs = []
    
    # Select n edges from the graph at random with replacement
    all_edges = list(graph.edges)
    sampled_edges = random.choices(all_edges, k=n)
    for u, v in sampled_edges:
        sampled_pairs.append(list([u, v]))

    return sampled_pairs, len(sampled_edges)

def trio_1(graph, n, config=1):
    # Dictionary to store nodes with two or more children and their sibling pairs count
    weighting_dict = {}
    
    # Iterate over each node in the graph
    for node in graph.nodes:
        # Get the children of the current node
        n_children = len(list(graph.successors(node)))
        
        # Check if the node has two or more children
        if n_children >= 2:
            # Calculate the number of sibling pairs
            sibling_pairs_count = math.comb(n_children, 2)
            weighting_dict[node] =  sibling_pairs_count   
    
    # Prepare a lookup table of cutoffs for each node defined using the weights. Then use a random number generator to sample a parent node and two children.
    # Create a lookup table for cutoffs based on sibling pairs count
    cutoff_table = {}
    total_instances = sum(sibling_pairs_count for sibling_pairs_count in weighting_dict.values())
    
    # Calculate cumulative weights for each node
    cumulative_weight = 0
    for node, sibling_pairs_count in weighting_dict.items():
        cumulative_weight += sibling_pairs_count / total_instances
        cutoff_table[node] = cumulative_weight
    
    # Use a random number generator to sample a parent node. Repeat n times.
    sampled_pairs = []
    for _ in range(n):
        random_value = random.random()
        parent_node = None
        for node, cutoff in cutoff_table.items():
            if random_value <= cutoff:
                parent_node = node
                break
        if config == 1:
            # Sample one child at random from the parent node's children, and append the parent-child pair to the sampled_pairs list.
            sampled_pairs.append(list([parent_node, random.choice(list(graph.successors(parent_node)))]))
        elif config == 2:
            # Sample two children at random without replacement from the parent node's children, and append the parent-child pair to the sampled_pairs list.
            sampled_children = random.sample(list(graph.successors(parent_node)), 2)
            # Sort the sampled children
            sampled_children.sort()
            sampled_pairs.append(sampled_children)
        else: 
            print("Invalid config")
            return []
    return sampled_pairs, total_instances


def trio_2(graph, n, config=1):
    # Define an empty dictionary called weighting_dict
    weighting_dict = {}

    # Cycle through all edges between u and v
    for u, v in graph.edges:
        # If v has predecessors, add the pairing to the weighting dict
        if list(graph.predecessors(v)):
            weighting_dict[(u, v)] = len(list(graph.successors(v)))

    # Initialize sampled_pairs
    sampled_pairs = []

    # Iterate n times
    for _ in range(n):
        # Sample from the dict using a weighted (random number) procedure
        total_weight = sum(weighting_dict.values())
        random_value = random.uniform(0, total_weight)
        cumulative_weight = 0
        selected_pair = None

        for pair, weight in weighting_dict.items():
            cumulative_weight += weight
            if random_value <= cumulative_weight:
                selected_pair = pair
                break

        u, v = selected_pair
        # Choose a child of v at random
        w = random.choice(list(graph.successors(v)))

        # Append to sampled_pairs based on config
        if config == 1:
            sampled_pairs.append([u, w])
        elif config == 2:
            sampled_pairs.append([v, w])
        elif config == 3:
            sampled_pairs.append([u, v])
        else:
            print("Invalid config")
            return []

    return sampled_pairs, total_weight
    
def quad_0(graph, n, config=1):
    # Dictionary to store nodes with two or more children and their sibling pairs count
    weighting_dict = {}
    
    # Iterate over each node in the graph
    for node in graph.nodes:
        # Get the children of the current node
        n_children = len(list(graph.successors(node)))
        
        # Check if the node has two or more children
        if n_children >= 3:
            # Calculate the number of sibling pairs
            sibling_pairs_count = math.comb(n_children, 3)
            weighting_dict[node] =  sibling_pairs_count   
    
    # Prepare a lookup table of cutoffs for each node defined using the weights. Then use a random number generator to sample a parent node and two children.
    # Create a lookup table for cutoffs based on sibling pairs count
    cutoff_table = {}
    total_instances = sum(sibling_pairs_count for sibling_pairs_count in weighting_dict.values())
    
    # Calculate cumulative weights for each node
    cumulative_weight = 0
    for node, sibling_pairs_count in weighting_dict.items():
        cumulative_weight += sibling_pairs_count / total_instances
        cutoff_table[node] = cumulative_weight
    
    # Use a random number generator to sample a parent node. Repeat n times.
    sampled_pairs = []
    for _ in range(n):
        random_value = random.random()
        parent_node = None
        for node, cutoff in cutoff_table.items():
            if random_value <= cutoff:
                parent_node = node
                break
        if config == 1:
            # Sample one child at random from the parent node's children, and append the parent-child pair to the sampled_pairs list.
            sampled_pairs.append(list([parent_node, random.choice(list(graph.successors(parent_node)))]))
        elif config == 2:
            # Sample two children at random without replacement from the parent node's children, and append the parent-child pair to the sampled_pairs list.
            sampled_children = random.sample(list(graph.successors(parent_node)), 2)
            # Sort the sampled children
            sampled_children.sort()
            sampled_pairs.append(sampled_children)
        else: 
            print("Invalid config")
            return []
    return sampled_pairs, total_instances


def quad_1(graph, n, config=1):
    # Initialize a weighting_dict
    weighting_dict = {}

    # Cycle through all edges (u, v)
    for u, v in graph.edges:
        # Check whether v has 2 or more successors
        successors = list(graph.successors(v))
        if len(successors) >= 2:
            # Add the edge to the dictionary with weight equal to number of successors of v choose 2
            weighting_dict[(u, v)] = len(successors) * (len(successors) - 1) // 2

    # Initialize the sampled_pairs list
    sampled_pairs = []

    # Looping over range(n), do a weighted sample n times using a random number generator
    for _ in range(n):
        total_weight = sum(weighting_dict.values())
        random_value = random.uniform(0, total_weight)
        cumulative_weight = 0
        selected_pair = None

        for pair, weight in weighting_dict.items():
            cumulative_weight += weight
            if random_value <= cumulative_weight:
                selected_pair = pair
                break

        u, v = selected_pair
        successors = list(graph.successors(v))

        # If config is 1, append u and one of v's successors (chosen at random)
        if config == 1:
            w = random.choice(successors)
            append_pair = [u, w]
            append_pair.sort()
            sampled_pairs.append(append_pair)
        # If config is 2, append v and one of its successors (chosen at random)
        elif config == 2:
            w = random.choice(successors)
            append_pair = [v, w]
            append_pair.sort()
            sampled_pairs.append(append_pair)
        # If config is 3, append u and v
        elif config == 3:
            append_pair = [u, v]
            append_pair.sort()
            sampled_pairs.append(append_pair)
        # If config is 4, choose two distinct children of v at random
        elif config == 4:
            w, x = random.sample(successors, 2)
            # sort the pair
            append_pair = [w, x]
            append_pair.sort()
            sampled_pairs.append(append_pair)
        else:
            print("Invalid config")
            return []

    return sampled_pairs, total_weight

def quad_2(graph, n, config=1):
    # Initialize the weighting dictionary
    weighting_dict = {}

    # Cycle through all edges (u, v)
    for u, v in graph.edges:
        # Check whether u has 2 or more successors and v has 1 or more successors
        successors_u = list(graph.successors(u))
        successors_v = list(graph.successors(v))
        if len(successors_u) >= 2 and len(successors_v) >= 1:
            # Add the edge to the dictionary with weight equal to number of successors of v times (number of successors of u minus 1)
            weighting_dict[(u, v)] = len(successors_v) * (len(successors_u) - 1)

    # Initialize the sampled_pairs list
    sampled_pairs = []

    # Looping over range(n), do a weighted sample n times using a random number generator
    for _ in range(n):
        total_weight = sum(weighting_dict.values())
        random_value = random.uniform(0, total_weight)
        cumulative_weight = 0
        selected_pair = None

        for pair, weight in weighting_dict.items():
            cumulative_weight += weight
            if random_value <= cumulative_weight:
                selected_pair = pair
                break

        u, v = selected_pair
        successors_u = list(graph.successors(u))
        successors_v = list(graph.successors(v))

        # If config is 1, append u and one of v's successors (chosen at random)
        if config == 1:
            w = random.choice(successors_v)
            append_pair = [u, w]
            append_pair.sort()
            sampled_pairs.append(append_pair)
        # If config is 2, choose a child of v and a child of u (not v) at random
        elif config == 2:
            w = random.choice(successors_v)
            x = random.choice([node for node in successors_u if node != v])
            append_pair = [w, x]
            append_pair.sort()
            sampled_pairs.append(append_pair)
        # If config is 3, append v and one of its successors (chosen at random)
        elif config == 3:
            w = random.choice(successors_v)
            append_pair = [v, w]
            append_pair.sort()
            sampled_pairs.append(append_pair)
        # If config is 4, append u and v
        elif config == 4:
            append_pair = [u, v]
            append_pair.sort()
            sampled_pairs.append(append_pair)
        # If config is 5, append v and another child of u (chosen at random)
        elif config == 5:
            x = random.choice([node for node in successors_u if node != v])
            append_pair = [v, x]
            append_pair.sort()
            sampled_pairs.append(append_pair)
        else:
            print("Invalid config")
            return []

    return sampled_pairs, total_weight

def quad_3(graph, n, config=1):
    weighting_dict = {}
    
    # Cycle through all edges (u, v)
    for u, v in graph.edges():
        predecessors_u = list(graph.predecessors(u))
        successors_v = list(graph.successors(v))
        
        # Append edge to dict keys if there are >0 predecessors of u and >0 successors of v
        if predecessors_u and successors_v:
            weighting_dict[(u, v)] = len(successors_v)
    
    sampled_pairs = []
    total_weight = sum(weighting_dict.values())
    
    for _ in range(n):
        random_value = random.uniform(0, total_weight)
        cumulative_weight = 0
        selected_pair = None
        
        # Take a weighted average of dict keys
        for pair, weight in weighting_dict.items():
            cumulative_weight += weight
            if random_value <= cumulative_weight:
                selected_pair = pair
                break
        
        u, v = selected_pair
        t = random.choice(list(graph.predecessors(u)))
        w = random.choice(list(graph.successors(v)))
        

        if config == 1:
            append_pair = [t, w]
        elif config == 2:
            append_pair = [u, w]
        elif config == 3:
            append_pair = [t, v]
        elif config == 4:
            append_pair = [v, w]
        elif config == 5:
            append_pair = [u, v]
        elif config == 6:
            append_pair = [t, u]
        else:
            print("Invalid config")
            return []
        
        append_pair.sort()
        sampled_pairs.append(append_pair)
    
    return sampled_pairs, total_weight