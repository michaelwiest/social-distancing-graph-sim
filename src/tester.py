from graph_constructor import GraphConstructor
from simulator import Simulator
import matplotlib.pyplot as plt
import networkx as nx

# How many nodes (people) in the simulation.
N = 2000
cluster_size = 4
cluster_stdev = 0.8
in_cluster_transition = 0.08
out_cluster_transition = 0.02
out_cluster_edge_mean = 0.9
out_cluster_edge_stdev = 0.6

# How many timesteps until someone recovers.
timesteps_for_recovery = 30
num_initially_infected = 2

GC = GraphConstructor(N,
                     cluster_size,
                     cluster_stdev,
                     in_cluster_transition,
                     out_cluster_transition,
                     out_cluster_edge_mean,
                     out_cluster_edge_stdev)
GC.construct_graph()
S = Simulator(GC.graph)
S.prepare_simulation(num_initially_infected,
                     timesteps_for_recovery=timesteps_for_recovery)
S.simulate(num_steps=None)
# S.animate_infection(interval=0.1, granularity=5,
#                     keep_final=True)
S.show_stats()
plt.show()
