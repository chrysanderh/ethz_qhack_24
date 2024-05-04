# This file is adapted from
# https://github.com/ZeddTheGoat/QAOA_in_QAQA/blob/main/QAOA_square.py

import numpy as np
import json
from utilities import *
from QAOA import *


def qaoa_square(G:Graph, depth:int=1, sub_size:int=10):
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
        H = G.graph_partition(n=sub_size,policy='random')
        
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
    hex_graph = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,4), (2,3), (2,4), (2,5), (3,5), (4,5)]
    G = Graph(v=list(range(6)), edges=hex_graph)

    value, sols = qaoa_square(G, depth=2, sub_size=3)
    #value, sols = qaoa(G, layer_count=2)

    print(value)
    print(sols)
