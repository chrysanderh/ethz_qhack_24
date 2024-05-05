# This file is adapted from
# https://github.com/ZeddTheGoat/QAOA_in_QAQA/blob/main/QAOA_square.py

import numpy as np
import json
from src.utilities import *
from src.QAOA import *
import csv
import os
import sys

def qaoa_square(G:Graph, depth:int=1, sub_size:int=10, partition_method:str='random'):
    '''
    data_path : where graph data saved

    depth : depth level of QAOA circuit

    sub_size : allowable number of qubit

    return : 
    '''

    # create a Graph
    #G = Graph(v=list(range(n_v)), edges=edges)

    const = 0
    sols = {}
    # level indicate the hierarchy of QAOA
    level = 0 
    while G.n_v > sub_size:
        
        sols[level] = {}
        H = G.graph_partition(n=sub_size,policy=partition_method)
        
        obj = []
        sol = []
        for H_sub in H:
            const_temp = 0.5 * sum([x[2] for x in H_sub.e])
            ret = qaoa(H_sub,const=const_temp,layer_count=depth)
            obj.append(ret[0])
            sol.append(ret[1])
        sols[level]['sol'] = sol
        sols[level]['v'] = [h.v for h in H]
        n_sub = len(H)
        adjoint = np.zeros((n_sub,n_sub))
        for i in range(n_sub):
            for j in range(i+1, n_sub):
                w_pos = 0
                w_neg = 0
                for x in range(H[i].n_v):
                    for y in range(H[j].n_v):
                        m, n = H[i].v[x], H[j].v[y]
                        w_pos += (sol[i][x]!=sol[j][y]) * G.adj[m][n]
                        w_neg += (sol[i][x]==sol[j][y]) * G.adj[m][n]
                # w^prime
                adjoint[i][j]= w_neg-w_pos
                adjoint[j][i]= w_neg-w_pos
                const += w_pos
            const += obj[i]
        G = Graph(v=list(range(n_sub)), adjoint=adjoint)
        level += 1

    const_temp = 0.5 * sum([x[2] for x in G.e])
    ret = qaoa(G, const = const+const_temp, layer_count=depth)

    sols[level] = {}
    sols[level]['sol'] = ret[1]
    sols[level]['v'] = G.v

    return ret[0].item(), sols


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage:", "./" + sys.argv[0], "filename", "n_nodes", "partition_size", "num_layers")
        exit(1)

    headers = ["nodes", "runtime", "partition_gurobi", "partition_dac", "cost_gurobi", "cost_dac", "dac_subgraph_size", "num_layers"]
    # Check if the file exists
    file_path = sys.argv[1]
    # If the file doesn't exist, create it and add the headers
    if not os.path.isfile(file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    n_nodes = int(sys.argv[2])
    partition_size = int(sys.argv[3])

    graph, edges = generate_regular_3_graph(n_nodes, seed=42)
    gurobi_sol = max_cut_gurobi(graph)

    G = Graph(v=list(range(n_nodes)), edges=edges)

    start = time.time()
    num_layers = int(sys.argv[4])
    value, sols = qaoa_square(G, depth=num_layers, sub_size=partition_size, partition_method='modularity')
    end = time.time()
    
    results = {'nodes': n_nodes, 'dac_subgraph_size': partition_size}

    runtime = end - start
    results['runtime'] = runtime
    if gurobi_sol is not None:
        results['cost_gurobi'] = gurobi_sol[0]
        results['partition_gurobi'] = gurobi_sol[1]

    partition_dac = contract_solution(sols, sort=True)[0]['sol']
    results['partition_dac'] = partition_dac

    cost_dac = get_cost(partition_dac, edges)
    results['cost_dac'] = cost_dac
    results['num_layers'] = num_layers

    results_list = [results[key] for key in headers]
    with open(file_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(results_list)

    print("== RESULTS ==")
    for h in headers:
        print("  " + str(h) + ": " + str(results[h]))

