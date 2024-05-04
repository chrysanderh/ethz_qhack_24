import time
from datetime import datetime

import cudaq
from cudaq import spin

from typing import List

import numpy as np

#cudaq.set_target("nvidia-fp64") # activates the single-gpu backend

def run_qaoa(hamiltonian, qubit_count, layer_count, parameter_count, n_shots = 1000, save_file = False):
    start = time.time()

    now = datetime.now()
    if save_file:
        filename = now.strftime('data/%Y%m%d-%H%M%S-') + f'Q{qubit_count}L{layer_count}N{n_shots}' + ".csv"
        fp = open(filename, "w")

    @cudaq.kernel
    def kernel_qaoa(qubit_count: int, layer_count: int, thetas: List[float]):
        """QAOA ansatz for Max-Cut"""
        qvector = cudaq.qvector(qubit_count)
    
        # Create superposition
        h(qvector)
    
        # Loop over the layers
        for layer in range(layer_count):
            # Loop over the qubits
            # Problem unitary
            for qubit in range(qubit_count):
                x.ctrl(qvector[qubit], qvector[(qubit + 1) % qubit_count])
                rz(2.0 * thetas[layer], qvector[(qubit + 1) % qubit_count])
                x.ctrl(qvector[qubit], qvector[(qubit + 1) % qubit_count])
    
            # Mixer unitary
            for qubit in range(qubit_count):
                rx(2.0 * thetas[layer + layer_count], qvector[qubit])
    
    
    # Specify the optimizer and its initial parameters. Make it repeatable.
    cudaq.set_random_seed(13)
    optimizer = cudaq.optimizers.COBYLA()
    np.random.seed(13)
    optimizer.initial_parameters = np.random.uniform(-np.pi / 8.0, np.pi / 8.0,
                                                     parameter_count)    
    
    # Define the objective, return `<state(params) | H | state(params)>`
    def objective(parameters):
        result = cudaq.observe(kernel_qaoa, hamiltonian, qubit_count, layer_count,
                             parameters).expectation()
        if save_file:
            fp.write(str(time.time()) + "," + str(result) + "\n")
        return result
    
    
    # Optimize!
    optimal_expectation, optimal_parameters = optimizer.optimize(
        dimensions=parameter_count, function=objective)
        
    if save_file:
        fp.close()
    
    # Sample the circuit using the optimized parameters
    counts = cudaq.sample(kernel_qaoa, qubit_count, layer_count, optimal_parameters, shots_count=n_shots)
    end = time.time()
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    sol = sorted_counts[0][0]
    
    return sol

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
    for u, v in edges[1:]:
        hamiltonian += 0.5 * spin.z(u) * spin.z(v)
    return hamiltonian


# The Max-Cut for this problem is 010110 or 101001.

# The problem Hamiltonian
hex_graph = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,4), (2,3), (2,4), (2,5), (3,5), (4,5)]
hamiltonian = get_hamiltonian(hex_graph)
# Problem parameters.
qubit_count: int = 6
layer_count: int = 2
parameter_count: int = 2 * layer_count

run_qaoa(hamiltonian, qubit_count, layer_count, parameter_count)