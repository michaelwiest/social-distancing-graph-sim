from graph_simulator import GraphSimulator
import matplotlib.pyplot as plt
import networkx as nx

GS = GraphSimulator(100, 5, .5, .8, .2, 0.9, .6)
GS.construct_graph()
GS.prepare_simulation(2)

GS.draw()
plt.show()
