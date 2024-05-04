import networkx as nx
from gurobipy import Model, GRB
from networkx.algorithms.approximation import maxcut



def create_graph(reg_deg = 3, nb_nodes=4, seed=42):
    G = nx.random_regular_graph(reg_deg, nb_nodes, seed=seed)



def max_cut_gurobi(graph: nx.Graph):
    """Solve the Max-Cut problem using Gurobi."""
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
        return [group1, group2]
    else:
        print("No optimal solution found.")
        return None


