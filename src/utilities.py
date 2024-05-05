# This file is adapted from
# https://github.com/ZeddTheGoat/QAOA_in_QAQA/blob/main/utilities.py

import numpy as np
import math
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from gurobipy import Model, GRB

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
            np.random.seed(42)
            np.random.shuffle(v)
            sub_list = [v[n*i:n*(i+1)] for i in range(n_sub)]
            for i in range(n_sub):
                A = self.adj[sub_list[i]][:,sub_list[i]]
                H.append(Graph(v=sub_list[i], adjoint=A))
        return H

def flatten_2d_list(l):
    """
    Take a 2D list and flatten it along the first dimension (stack second dimension horizontally)
    
    Parameters: List[List]

    Returns:
        List: flattened list
    """
    flattened = []
    for i in l:
        for j in i:
            flattened.append(j)
    return flattened

def contract_level(sols, sort = True):
    """
    Take a dictionary from a divide and conquer solution and contract the last level

    Parameters:
        sols: Dict{int: Dict{'sol': List[String], 'v': List[List[Int]]}} Output from qaoa_square

    Returns:
        sol_dict: Dict with one less entry than sols where the last level has been used to flip the partitions for the previous level
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
    """
    Iteratively contract levels until only one remains

    Parameters:
        sols: Dict{int: Dict{'sol': List[String], 'v': List[List[Int]]}} Output from qaoa_square
    
    Returns:
        sols: Dict{0: Dict{'sol': String, 'v': List[Int]}
    """
    while len(sols) > 1:
        sols = contract_level(sols, sort)
    return sols

def get_cost(x_vec, graph):
    """
    Get the number of cut edges of a graph partition.

    Parameters:
        x_vec: List[Int], String Partition of the graph
        graph: List[Tuple[Int, Int]] Edges of the graph

    Returns:
        cost: Int Number of cut edges
    """
    cost = 0
    for edge in graph:
        n1, n2 = edge
        cost += int(x_vec[n1])*(1-int(x_vec[n2])) + int(x_vec[n2])*(1-int(x_vec[n1]))

    return cost

def max_cut_gurobi(graph):
    """
    Use Gurobi algorithm to solve the max-cut problem.
    """
    model = Model("max_cut")
    model.setParam('OutputFlag', 0)

    # Variables: x[i] is 1 if node i is in one set of the cut, 0 otherwise
    x = model.addVars(graph.nodes(), vtype=GRB.BINARY, name="x")

    # Objective: Maximize the sum of edges between the sets
    model.setObjective(sum(x[i] + x[j] - 2 * x[i] * x[j] for i, j in graph.edges()), GRB.MAXIMIZE)

    # Optimize model
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print('Max-Cut Value:', int(model.objVal))
        group1 = {i for i in graph.nodes() if x[i].X >= 0.5}
        group2 = {i for i in graph.nodes() if x[i].X < 0.5}
        return int(model.objVal), [group1, group2]
    else:
        print("No optimal solution found.")
        return None

def generate_regular_3_graph(n_nodes, seed):
    """
    Generate a random regular-3 graph with the given number of nodes and fixed seed.
    """
    G = nx.random_regular_graph(3, n_nodes, seed=seed)
    edges = list(G.edges())
    return G, edges

def cost_function_qaoa(results, edges):
 """
 This function calculates the cost of a max cut solution.
 Parameters
 ----------
 results : list
     A list of results from a quantum algorithm.
 edges : list
     A list of edges in the graph.
 Returns
 -------
 cost : int
     The cost of the max cut solution.
 """
 x_vec = results[0][0]
 cost = 0
 for edge in edges:
     n1, n2 = edge
     cost += int(x_vec[n1]) * (1-int(x_vec[n2])) + int(x_vec[n2]) * (1-int(x_vec[n1]))
 return cost


def visualize_max_cut(G, partition, title):
    """
    Visualize the graph with the max cut highlighted

    Parameters
    ----------
    G : networkx.Graph
        The input graph
    partition : tuple
        A partition of the nodes of G into two sets
    title : str
        The title of the plot

    Returns
    -------
    None
    """
    # Prepare to plot the graph
    pos = nx.spring_layout(G)  # positions for all nodes using the spring layout

    # Draw 
    nx.draw_networkx_nodes(G, pos, nodelist=partition[0], node_color='red', node_size=500, alpha=0.8) 
    nx.draw_networkx_nodes(G, pos, nodelist=partition[1], node_color='blue', node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # Draw node labels
    nx.draw_networkx_labels(G, pos)

    # Set plot title
    plt.title(title)
    plt.axis('off')  # Turn off the axis
    plt.show()
    

def partition_to_bin(partition):
    """
    This function converts a max cut partition of a graph into a binary number.

    Parameters
    ----------
    partition : tuple
        A tuple containing two sets of nodes that represent the partition of a graph.

    Returns
    -------
    binary_string : str
        A binary number that represents the partition of the graph.
    """
    # Define the subsets
    subset_zero = partition[0]
    subset_one = partition[1]

    # Determine the maximum index to define the size of the binary number
    max_index = max(max(subset_zero), max(subset_one))  # Find the highest index

    # Initialize the binary number list with zeros
    binary_number = [0] * (max_index + 1)

    # Set the positions in the binary number according to the subsets
    for index in subset_one:
        binary_number[index] = 1

    # Since the binary number needs to be expressed as an actual number, convert list to string and then to a binary number
    binary_string = ''.join(str(bit) for bit in binary_number)

    # Print results
    # print("Max cut solution, classical:", binary_string)
    return binary_string


def bin_to_partition(binary_string):
    """
    This function converts a binary number into a max cut partition of a graph.

    Parameters
    ----------
    binary_string : str
        A binary number that represents the partition of the graph.

    Returns
    -------
    partition : tuple
        A tuple containing two sets of nodes that represent the partition of a graph.
    """
    # Convert the binary number to a list of integers
    binary_list = [int(bit) for bit in binary_string]

    # Define the subsets
    subset_zero = set()
    subset_one = set()

    # Set the positions in the subsets according to the binary number
    for index, bit in enumerate(binary_list):
        if bit == 0:
            subset_zero.add(index)
        else:
            subset_one.add(index)

    # Define the partition
    partition = (subset_zero, subset_one)

    # Print results
    # print("Max cut solution, quantum:", partition)
    return partition
