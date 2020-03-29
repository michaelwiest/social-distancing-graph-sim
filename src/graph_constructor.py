import networkx as nx
import pandas as pd
import numpy as np

class GraphConstructor(object):
    def __init__(self,
                 N: int,  # number of people in graph,
                 cluster_size: int,  # Average number of people per cluster
                 cluster_stdev: float,  # Standard Deviations of people per cluster
                 in_cluster_transition: float,  # Transmission probability in cluster
                 out_cluster_transtion: float,  # Transmission probability out cluster
                 out_cluster_edge_mean: float, # Average number of out-of cluster edges.
                 out_cluster_edge_stdev: float, # Standard Deviations of edges out of cluster.
                 ):
        '''
        Docstring move above shit here.
        '''
        self.N = N
        self.cluster_size = cluster_size
        self.cluster_stdev = cluster_stdev
        self.in_cluster_transition = in_cluster_transition
        self.out_cluster_transition = out_cluster_transtion
        self.out_cluster_edge_mean = out_cluster_edge_mean
        self.out_cluster_edge_stdev = out_cluster_edge_stdev

        self._graph = None

        self.active_graph = None

    @property
    def graph(self):
        if self._graph is None:
            raise ValueError('Please call construct_graph.')
        return self._graph

    def _add_edges_for_new_cluster(self,
                                   new_cluster):
        # Generate all of the edges to add
        edges_to_add = [[tuple((new_cluster[i], new_cluster[j]))
                         for j in range(i + 1, len(new_cluster))]
                         for i in range(len(new_cluster))]
        edges_to_add = [item for sublist in edges_to_add for item in sublist]
        # Add the edges.
        self._graph.add_edges_from(edges_to_add,
                                   type='in_cluster',
                                   weight=self.in_cluster_transition)

    def _construct_clusters(self):
        '''
        Function for generating tightly connected clusters of edges.
        '''
        clustered_nodes = set()
        unclustered_nodes = set(self._graph.nodes())
        self.clusters = []

        # Generate clusters until all nodes are clustered.
        cluster_number = 0
        while len(clustered_nodes) < self.N:
            new_cluster_size = int(np.floor(np.random.normal(self.cluster_size,
                                                             self.cluster_stdev)))
            new_cluster_size = max(0, new_cluster_size)
            # print(unclustered_nodes)
            new_cluster_size = min(new_cluster_size, len(unclustered_nodes))
            new_cluster = np.random.choice(list(unclustered_nodes),
                                           size=new_cluster_size,
                                           replace=False).tolist()

            [unclustered_nodes.remove(nc) for nc in new_cluster]
            clustered_nodes.update(set(new_cluster))
            self.clusters.append(new_cluster)

            self._add_edges_for_new_cluster(new_cluster)
            cluster_name_dict = {nn: {'cluster_number': cluster_number}
                                 for nn in new_cluster}
            # cluster_name_dict = {cluster_name_dict}
            nx.set_node_attributes(self.graph,
                                   cluster_name_dict)
            cluster_number += 1

    def _add_out_of_cluster_edges(self):

        for node in self._graph.nodes():
            num_new_edges = max(0, int(np.floor(np.random.normal(self.out_cluster_edge_mean,
                                             self.out_cluster_edge_stdev))))
            # print(num_new_edges)
            neighbors = list(self._graph.neighbors(node)) + [node]
            potential_new_neighbors = list(set(self.graph.nodes())
                                           - (set(neighbors)))
            new_neighbors = np.random.choice(potential_new_neighbors,
                                             num_new_edges,
                                             replace=False)
            self.graph.add_edges_from([tuple((node, nn))
                                       for nn in new_neighbors],
                                      type='out_cluster',
                                      weight=self.out_cluster_transition
                                      )

    def construct_graph(self):
        self._graph = nx.Graph()
        self._graph.add_nodes_from(list(range(self.N)),
                                   infected=False,
                                   symptomatic=False,
                                   recovered=False,
                                   infected_timestep=np.inf)
        self._construct_clusters()
        self._add_out_of_cluster_edges()
