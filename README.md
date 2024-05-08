# ethz_qhack_24
Repository for the 24h ETH Quantum Hackathon 2024. Challenge: NVIDIA.


## Team Members
- [Vinicius Mohr](https://github.com/vimohr)
- [Chrysander D. Hagen](https://github.com/chrysanderh)
- [Paul Uriarte Vicandi](https://github.com/PaU1570)
- [Maurice D. Hanisch](https://github.com/MauriceDHanisch)


## Structure
The repository is structured as follows:
- `Getting_started/`: Contains introductory notebooks on how to use CUDA-Q.
- `Other/`: Contains additional files and information.
- `results/`: Contains the QAOA results in a CSV format.
- `src/`: Contains the source code for the project.
- `FINAL_QHACK.pptx`: Contains the final presentation for the project.
- `Final_Submission.ipynb`: Contains the final submission for the project as a Jupyter Notebook.


## Challenge and solution overview
The given instructions can be found in the following [pdf file](.Other/ETHZ_QHack_2024_NVIDIA.pdf). We will introduce the main points here.

The Vehicle Routing Problem (VRP) is an NP-Hard combinatorial optimization problem that involves finding the best routes for a fleet of vehicles to deliver to customers ![VRP example](.Other/VRP.jpeg). In the most straightforward setting, the goal is to minimize the total distance traveled by the cars. In some sense, the VRP is a generalization of the Traveling Salesman Problem (TSP). Interestingly, the VRP is a general problem that finds applications in many critical fields, such as supply chain management, health care, and the chip design industry. Consequently, finding efficient solutions to the VRP is of paramount importance.

The VRP can be solved using a hybrid classical-quantum approach called Quantum Approximate Optimization Algorithm (QAOA). The QAOA is a variational algorithm used to solve combinatorial optimization problems. The adiabatic theorem [[1]](https://doi.org/10.1007/BF01343193) guarantees its convergence for infinite quantum circuit depth. 

As circuit depth corresponds to computation time, the theorem does not imply that the QAOA provides a polynomial-time solution. Moreover, there is no proof that QAOA efficiently finds the optimal solution. Consequently, there is `no guarantee of `quantum advantage`, and QAOA does not fall into the problems with known (exponential) quantum speedups, such as Shor's factoring or quantum simulation of quantum systems. However, finite-depth QAOA is still currently used as a heuristic algorithm to solve combinatorial optimization problems because, in some cases, it can be implemented using shallow circuits suitable for NISQ devices. 

However, current devices have finite reliability (gate errors, SPAM errors, leakage, crosstalk, etc..), compromising the algorithm's performance. For example, having noisy devices can lead to an exponential flat landscape of the cost function (barren plateaus) and make the optimization problem hard to solve. We can use simulations of noiseless quantum circuits on classical computers to address this issue and find ways to test and refine our algorithms before having access to fault-tolerant quantum computers. This project aims to address this aspect using `NVIDIA CUDA-Q` to simulate the QAOA algorithm and explore its performance on optimization problems.

To solve the VRP using a quantum system, we can map the cost function of the VRP to a Hamiltonian and then, using the parametrized quantum circuit of the QAOA, classically optimize the state's preparation that minimizes the Hamiltonian's energy expectation. The found state corresponds to a solution with hopefully extremal energy/cost and thus is a solution to the VRP. We will not detail the exact form of the QAOA circuit here and reference the reader to a review paper on the topic [[2]](https://arxiv.org/abs/2306.09198). 

The cost function of the VRP can be mapped to a Hamiltonian using a binary decision vector of length `n*(n-1)`, where `n` is the number of customers (+ one node for the depot) in the VRP. The VRP can then be formulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem. The QUBO problem can be solved using the QAOA algorithm on `n*(n-1)` qubits. The exact mapping of the VRP to a QUBO problem is detailed in [[3]](https://arxiv.org/abs/2002.01351v2). It is important to note that the number of qubits required to solve the VRP grows quadratically with the number of customers. Consequently, the VRP is a challenging problem and requires many qubits to solve efficiently.

The proposed challenge was translating the VRP to an instance of the Max-Cut problem [[4]](https://doi.org/10.1007/978-1-4684-2001-2_9). To achieve this translation, we can use the Ising QUBO formulation from Equation 21 of [[3]](https://arxiv.org/abs/2002.01351v2) and make it quadratic by adding an extra qubit with spin `s_extra = 1` and rewrite the `sum_i h_i s_i` term as `sum_i h_i_extra s_i s_extra` and integrate it in `- sum_i sum_{i<j} J_{ij} s_i s_j`. Moreover, the quadratic form of the QUBO problem is equivalent to a Max Cut problem on `n*(n-1) + 1` nodes, achieving the required mapping of the VRP to the Max-Cut problem.

The number of qubits needed to implement the VRP grows quadratically with the number of customers, and to solve relevant instances of the VRP, a large number of qubits is required. As the computational space grows exponentially with the number of qubits, the simulation of QAOA becomes increasingly tricky. Consequently, we decided to focus the project on scaling QAOA for the Max-Cut problem to as big graphs as possible using `distributed computing` and a `divide-and-conquer` approach [[5]](https://arxiv.org/pdf/2205.11762). To have consistent and comparable results to the literature, we focused on the Max-Cut of `3-regular` graphs where 2-layer QAOA is guaranteed to find a Max-Cut with a weight that is `>75%` of the optimal solution [[2]](https://arxiv.org/abs/2306.09198). During the 24 hours of "hacking," we had access to two NVIDIA A100 GPUs, allowing us to explore the scaling of the QAOA algorithm on large graphs. We focused on the 


The project was divided into the following steps:
1. Implement a pipeline to generate random unweighted 3-regular graphs of increasing size and calculate their Max-Cut classically with `Gurobi`. 
2. Implement the QAOA algorithm using CUDA-Q to solve the Max-Cut problem on the generated graphs.
3. Simulate the QAOA algorithm on our local CPUs for up to 18 qubits and compare it to the "optimal" Gurobi Max-Cut. 
4. Simulate the QAOA algorithm on the NVIDIA A100 GPUs for up to 26 qubits and compare the runtime and performance to the local CPU simulations.
5. Implement the divide-and-conquer approach to scale the QAOA algorithm to larger graphs of up to 64 qubits. 


The final results are:
- The successful implementation of the QAOA algorithm using CUDA-Q allowing to find the `exact` (100%) Max-Cut solution for up to 26 qubits.
- The successful implementation of the QAOA algorithm on GPUs allowing a speedup of up to `260x` compared to the local CPU simulations.
- The successful implementation of the divide-and-conquer approach to scale the QAOA algorithm to larger graphs of up to 64 qubits while keeping a performance of `80%`. This number of qubits is beyond the reach of "naive" classical simulations on current HPC clusters.









## Other Challenges
Our project was selected as the winner of the NVIDIA challenge. Within the hackathon, we competed against the other challenges' winners and were awarded first place overall after presenting our solutions. We summarize the other challenges below.

### Moody’s x QuEra
MIS in complex graphs and applications in finance. The challenge was investigating the quantum speed-up in computing the MIS of defective king’s lattice graphs with connectivity via unit-disks of the order of 3 times the grid lattice constant and then finding an application in risk analysis and finance. 

### Quantinuum
The challenge was to investigate the normal forms of quantum circuits and their use for T gate optimization. The challenge repo can be found [here](https://github.com/CQCL/ethz-hack-24/). 

### Qilimanjaro
The challenge was about using quantum optimization algorithms on analog quantum hardware to solve the knapsack problem. It was needed to solve the problem with the least number of qubits to make it NISQ-compatible. The last part of the challenge was to work on the exact hardware model and simulate the results. 

### Pasqal
The challenge was about implementing Differentiable Quantum Circuits (DQC) [[1]](https://arxiv.org/abs/2011.10395) using Qadence. The goal was to implement a DQC to solve a 2D Laplace equation. As a bonus, it was proposed to use Harmonic quantum neural networks [[2]](https://arxiv.org/abs/2212.07462) to improve the results. The instructions repo can be found [here](https://github.com/pasqal-io/eth_quantum_2024).



