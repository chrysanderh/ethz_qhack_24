# This file is adapted from
# https://github.com/ZeddTheGoat/QAOA_in_QAQA/blob/main/QAOA_square.py

import numpy as np
import json
from utilities import *
from QAOA import *


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
    graph = [(16, 20), (16, 25), (16, 13), (20, 7), (20, 10), (18, 23), (18, 8), (18, 0), (23, 9), (23, 10), (7, 24), (7, 27), (15, 27), (15, 5), (15, 22), (27, 17), (6, 24), (6, 1), (6, 17), (24, 11), (12, 19), (12, 25), (12, 26), (19, 2), (19, 28), (3, 10), (3, 13), (3, 21), (13, 4), (25, 0), (14, 22), (14, 9), (14, 21), (22, 28), (9, 5), (8, 2), (8, 29), (1, 17), (1, 4), (2, 5), (0, 26), (26, 29), (29, 28), (4, 11), (11, 21)]
    n_nodes = max(max(graph, key=lambda x: x[0])[0], max(graph, key=lambda x: x[1])[1]) + 1
    print("Running divide and conquer...")
    print("Number of nodes:", n_nodes)
    G = Graph(v=list(range(n_nodes)), edges=graph)

    start = time.time()
    value, sols = qaoa_square(G, depth=2, sub_size=15, partition_method='modularity')
    end = time.time()
    
    print("Time taken:", str(end - start))
    
    print(value)
    print(sols)
    final_partition = contract_solution(sols, sort=True)[0]['sol']
    cost = get_cost(final_partition, graph)
    
    print("Final solution:", final_partition)
    print("Cost:", cost)

