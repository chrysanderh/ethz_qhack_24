# This file is adapted from
# https://github.com/ZeddTheGoat/QAOA_in_QAQA/blob/main/QAOA.py

import cudaq
from cudaq import spin

from typing import List

from utilities import *
import collections

import time
from datetime import datetime

def get_hamiltonian(edges):
    """
    Get the Hamiltonian mapping for an arbitrary graph

    Parameters
    ----------
    edges : List[Tuple[int, int]]
        List of edges in the graph
    
    Returns
    -------
    hamiltonian : cudaq.Operator
        Hamiltonian operator
    qubit_count : int
        Number of qubits required to represent the graph
    """
    # avoid 0 term in the Hamiltonian
    hamiltonian = 0.5 * spin.z(edges[0][0]) * spin.z(edges[0][1])
    if len(edges[0]) < 3:
        edges = [(e[0], e[1], 1) for e in edges]
    for u, v, w in edges[1:]:
        hamiltonian += 0.5 * spin.z(u) * spin.z(v)
    return hamiltonian

def qaoa(G:Graph, shots:int=1000, layer_count:int=1, const=0, save_file=False):
    '''
    standard qaoa for max cut
    --------------------------
    G : Graph 

    shots : number of circuit shots to obtain optimal bitstring

    layer_count : number of QAOA layers

    const : constant in max cut objective function


    Return cut value and solution
    '''
    qubit_count = G.n_v
    edges = G.e
    edges_1 = [edges[i][0] for i in range(len(edges))]
    edges_2 = [edges[i][1] for i in range(len(edges))]
    parameter_count = 2 * layer_count

    # subgraph with no edges, any partition is optimal
    if edges == []:
        return const, format(0,"0{}b".format(qubit_count))[::-1]

    hamiltonian = get_hamiltonian(edges)
    now = datetime.now()
    if save_file:
        filename = now.strftime('data/%Y%m%d-%H%M%S-') + f'Q{qubit_count}L{layer_count}N{shots}' + ".csv"
        fp = open(filename, "w")

    cudaq.set_target("nvidia") # activates the single-gpu backend
    
    @cudaq.kernel
    def kernel_qaoa(edges_src: List[int], edges_tgt: List[int], qubit_count: int, layer_count: int, thetas: List[float]):
        """
        QAOA ansatz for Max-Cut
        
        Parameters
        ----------
        edges : List[Tuple[int, int]]
            The edges of the graph.
        qubit_count : int  
            The number of qubits.
        layer_count : int
            The number of layers in the QAOA ansatz.
        thetas : List[float]
            The angles for the QAOA ansatz.
        """
        qvector = cudaq.qvector(qubit_count)

        # Create superposition
        h(qvector)

        # Loop over the layers
        for layer in range(layer_count):
            for i, u in enumerate(edges_src):
                v = edges_tgt[i]
                x.ctrl(qvector[u], qvector[v])
                rz(2.0 * thetas[layer], qvector[v])
                x.ctrl(qvector[u], qvector[v])

            # Mixer unitary
            for qubit in range(qubit_count):
                rx(2.0 * thetas[layer + layer_count], qvector[qubit])

    # Make it repeatable with fixing random seeds
    cudaq.set_random_seed(13)
    np.random.seed(13)

    # Specify the optimizer and its initial parameters for the angles in the layers
    optimizer = cudaq.optimizers.COBYLA()
    optimizer.initial_parameters = np.random.uniform(-np.pi / 8.0, np.pi / 8.0, parameter_count)
    #print("Initial parameters = ", optimizer.initial_parameters)

    def objective(parameters):
        """
        Compute the expected value of the hamiltonian with respect to the kernel.

        Parameters
        ----------
        parameters : List[float]
            The parameters to optimize. Contains the angles for the qaoa ansatz.

        Returns
        -------
        result : float
            The expectation value of the hamiltonian: `<state(params) | H | state(params)>`
        """
        return cudaq.observe(kernel_qaoa, hamiltonian, edges_1, edges_2, qubit_count, layer_count, parameters).expectation()

    optimal_expectation, optimal_parameters = optimizer.optimize(
        dimensions=parameter_count, function=objective)

    # Print the optimized value and its parameters
    #print("Optimal expectation value = ", optimal_expectation)
    #print("Optimal parameters = ", optimal_parameters)
    # print("No of iter = ", len(result_time))
        
    if save_file:
        fp.close()
    
    # Sample the circuit using the optimized parameters
    counts = cudaq.sample(kernel_qaoa, edges_1, edges_2, qubit_count, layer_count, optimal_parameters, shots_count=1000000)
    results = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    sol = results[0][0]

    obj = 0
    for edge in edges:
        obj += 0.5 * (2*(sol[edge[0]]==sol[edge[1]]) - 1)
    return const - obj, sol

if __name__ == '__main__':
    graph = [(16, 20), (16, 25), (16, 13), (20, 7), (20, 10), (18, 23), (18, 8), (18, 0), (23, 9), (23, 10), (7, 24), (7, 27), (15, 27), (15, 5), (15, 22), (27, 17), (6, 24), (6, 1), (6, 17), (24, 11), (12, 19), (12, 25), (12, 26), (19, 2), (19, 28), (3, 10), (3, 13), (3, 21), (13, 4), (25, 0), (14, 22), (14, 9), (14, 21), (22, 28), (9, 5), (8, 2), (8, 29), (1, 17), (1, 4), (2, 5), (0, 26), (26, 29), (29, 28), (4, 11), (11, 21)]
    n_nodes = max(max(graph, key=lambda x: x[0])[0], max(graph, key=lambda x: x[1])[1]) + 1
    print("Running normal QAOA...")
    print("Number of nodes:", n_nodes)
    G = Graph(v=list(range(n_nodes)), edges=graph)

    start = time.time()
    value, sol = qaoa(G, layer_count=2)
    end = time.time()
    
    print("Time taken:", str(end - start))
    
    print(value)
    cost = get_cost(sol, graph)
    
    print("Final solution:", sol)
    print("Cost:", cost)
