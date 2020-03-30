from graph_constructor import GraphConstructor
from pathogen import Pathogen
from simulator import Simulator
import matplotlib.pyplot as plt
import networkx as nx

# Parameters for the graph constructor
# How many nodes (people) in the simulation.
N = 1000
cluster_size = 10
cluster_stdev = 0.8
in_cluster_transition = 0.9
out_cluster_transition = 0.3
out_cluster_edge_mean = 1.1
out_cluster_edge_stdev = 0.6

# parameters for the pathogen object.
r_0 = 0.05
timesteps_for_recovery = 45
# What fraction of infections result in symptomatic infection.
symptomatic_prob = 0.2

# Parameters for the simulation.
num_initially_infected = 2
# If you are symptomatic, then your likelihood of interaction becomes this.
symptomatic_edge_weight = 0.1

pathogen = Pathogen(r_0,
                    timesteps_for_recovery,
                    symptomatic_prob)

GC = GraphConstructor(N,
                      cluster_size,
                      cluster_stdev,
                      in_cluster_transition,
                      out_cluster_transition,
                      out_cluster_edge_mean,
                      out_cluster_edge_stdev)
GC.construct_graph()

S = Simulator(GC.graph, pathogen)
S.prepare_simulation(num_initially_infected,
                     symptomatic_edge_weight=symptomatic_edge_weight)

# Run until completion.
S.simulate(num_steps=None)

# Uncomment this if you want to see the animation of
# the infection spreading in the graph. It can be a bit slow though.
# num_frames = 25
# granularity = max(1, int(len(S.sim_graphs) / num_frames))
# S.animate_infection(interval=0.1,
#                     granularity=granularity,
#                     keep_final=True)

S.show_stats()
plt.show()
