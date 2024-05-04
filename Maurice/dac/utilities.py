# This file is adapted from
# https://github.com/ZeddTheGoat/QAOA_in_QAQA/blob/main/utilities.py

import numpy as np
import math
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

class Graph():
    '''
    A graph is saved in both an adjoint matrix and edge list.
    '''
    def __init__(self, v:list=None, edges:list=None,adjoint=None) -> None:

        self.v = v
        self.n_v = len(v)
        self.e = edges
        self.adj = adjoint

        if self.adj is None:
            self._edges_to_adjoint()

        if self.e is None:
            self._adjoint_to_edges()

        self.v2i = {v[i]:i for i in range(self.n_v)}

    def _edges_to_adjoint(self) -> None:
        self.adj = np.zeros((self.n_v, self.n_v))
        for edge in self.e:
            v1 = edge[0]
            v2 = edge[1]
            if len(edge) < 3:
                w = 1
            else:
                w = edge[2]
            self.adj[v1][v2] = w
            self.adj[v2][v1] = w

    def _adjoint_to_edges(self) -> None:
        self.e = []

        for i in range(self.n_v):
            for j in range(i+1, self.n_v):
                if self.adj[i][j] != 0:
                    self.e.append((i, j, self.adj[i][j].item()))

    def graph_partition(self, n:int, policy:str='random',n_sub=1) -> list:
        '''
        n : Allowable qubit number.

        policy : Partition strategy. Default is 'random'. Another is 'modularity', which partitions graph basing on greedy modularity method.

        n_sub : number of subgraphs. Only use in 'modularity'.
        ''' 
        H = []
        v = self.v

        if policy == 'modularity':
            G = nx.Graph()
            G.add_nodes_from(v)
            for x in self.e:
                w = 1 if len(x) < 3 else x[3]
                G.add_edge(x[0],x[1],weight=w)
            c = greedy_modularity_communities(G,cutoff=n_sub,best_n=n_sub)
            sub_list = [list(x) for x in c]
            for x in sub_list:
                if len(x) > n:
                    n_ssub = math.ceil(len(x) / n)
                    
                    ssub_list = [x[n*i:n*(i+1)] for i in range(n_ssub)]
                    for i in range(n_ssub):
                        A = self.adj[ssub_list[i]][:,ssub_list[i]]
                        H.append(Graph(v=ssub_list[i], adjoint=A))
                else:
                    A = self.adj[x][:,x]
                    H.append(Graph(v=x, adjoint=A))
        if policy == 'random':
            n_sub = math.ceil(self.n_v / n)
            np.random.seed(24)
            np.random.shuffle(v)
            sub_list = [v[n*i:n*(i+1)] for i in range(n_sub)]
            for i in range(n_sub):
                A = self.adj[sub_list[i]][:,sub_list[i]]
                H.append(Graph(v=sub_list[i], adjoint=A))
        return H

def flatten_2d_list(l):
    flattened = []
    for i in l:
        for j in i:
            flattened.append(j)
    return flattened

def contract_level(sols, sort = True):
    """
    Take a dictionary from a divide and conquer solution and contract the last level

    Parameters:
        sols: Dict{int: List[String]}
    """
    level = max(sols.keys())
    merged_sol = []
    for i, s in enumerate(sols[level - 1]['sol']):
        # find index corresponding to this subgraph in the next level
        idx = 0
        for j, v in enumerate(sols[level]['v']):
            if v == i:
                idx = j
                break
        
        for b in s:     
            if sols[level]['sol'][idx] == '0':
                merged_sol.append(b)
            else:
                merged_sol.append('0' if b == '1' else '1')

    merged_sol = ''.join(merged_sol)
    nodes = flatten_2d_list(sols[level - 1]['v'])
    merged_sol_sorted = [-1 for _ in range(len(merged_sol))]
    for n, p in zip(nodes, merged_sol):
        merged_sol_sorted[n] = p
    merged_sol_sorted = ''.join(merged_sol_sorted)
    sol_dict = {i: sols[i] for i in range(level - 1)}
    if sort:
        sol_dict[level - 1] = {'sol': merged_sol_sorted, 'v': sorted(list(nodes))}
    else:
        sol_dict[level - 1] = {'sol': merged_sol, 'v': list(nodes)}
    return sol_dict

def contract_solution(sols, sort = True):
    while len(sols) > 1:
        sols = contract_level(sols, sort)
    return sols

def get_cost(x_vec, graph):
    cost = 0
    for edge in graph:
        n1, n2 = edge
        cost += int(x_vec[n1])*(1-int(x_vec[n2])) + int(x_vec[n2])*(1-int(x_vec[n1]))

    return cost




