from graph_constructor import GraphConstructor
from simulator import Simulator
import matplotlib.pyplot as plt
import networkx as nx

GC = GraphConstructor(2000,
                     4,
                     0.8,
                     0.25,
                     0.05,
                     0.9,
                     0.6)
GC.construct_graph()
S = Simulator(GC.graph)
S.prepare_simulation(2,
                     timesteps_for_recovery=21)
S.simulate(num_steps=50)
S.animate_infection(interval=0.1, keep_final=False)
S.show_stats()
plt.show()
