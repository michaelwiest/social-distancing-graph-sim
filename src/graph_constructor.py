import networkx as nx
import pandas as pd
import numpy as np
from typing import List

class GraphConstructor(object):
    def __init__(self,
                 N: int,
                 cluster_size: float,
                 cluster_stdev: float,
                 in_cluster_transition: float,
                 out_cluster_transtion: float,
                 out_cluster_edge_mean: float,
                 out_cluster_edge_stdev: float,
                 max_number_of_edges: int=None
                 ):
        """
        Graph Constructor Class. This is used to simulate clusters of
        people undergoing "social distancing."

        During the construction clusters and edges are generated
        using normal distributions about the provided means and stdevs
        for each set of parameters.

        Args:
            N int: Total number of people in the graph.
            cluster_size float: The average size of a cluster of people who are
                                social-distancing together.
            cluster_stdev float: the stdev of size of a cluster of people who
                                are social-distancing together.
            in_cluster_transition float: The probability of interacting with
                                         a node in the shelter-group
            out_cluster_transition float: The probability of interacting with
                                          a node NOT in the shelter-group
            out_cluster_edge_mean float: The average number of edges to nodes
                                         outside of an individual node's cluster.
            out_cluster_edge_stdev float: The stdev of number of edges to nodes
                                         outside of an individual node's cluster.
            max_number_of_edges int: The maximum number of edges in the graph.
                                     This is used to normalize comparisons
                                     between graphs with different cluster sizes,
                                     because as the cluster size grows so does the
                                     number of edges.
        """
        self.N = N
        self.cluster_size = cluster_size
        self.cluster_stdev = cluster_stdev
        self.in_cluster_transition = in_cluster_transition
        self.out_cluster_transition = out_cluster_transtion
        self.out_cluster_edge_mean = out_cluster_edge_mean
        self.out_cluster_edge_stdev = out_cluster_edge_stdev
        self.max_number_of_edges = max_number_of_edges

        self._graph = None

        self.active_graph = None

    @property
    def graph(self):
        '''
        The networkx graph object itself.

        returns networkx.Graph
        '''
        if self._graph is None:
            raise ValueError('Please call construct_graph.')
        return self._graph

    def _add_edges_for_new_cluster(self,
                                   new_cluster: List[int]):
        '''
        For a given cluster, connect all of the nodes with the appropriate
        weights and label the appropriately.

        Args:
            new_cluster: List[int]: indices of nodes in self.graph.
        '''
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
        The number of nodes per-cluster is sampled from a normal distribution
        about the parameters:
            self.cluster_size
            self.cluster_stdev
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
            new_cluster_size = min(new_cluster_size, len(unclustered_nodes))
            new_cluster = np.random.choice(list(unclustered_nodes),
                                           size=new_cluster_size,
                                           replace=False).tolist()
            # Remove the new indices from the ste of unclustered nodes.
            [unclustered_nodes.remove(nc) for nc in new_cluster]
            # Add new nodes to set of clustered ones.
            clustered_nodes.update(set(new_cluster))
            # Record the cluster.
            self.clusters.append(new_cluster)

            # Connect all of the nodes in the new cluster.
            self._add_edges_for_new_cluster(new_cluster)
            cluster_name_dict = {nn: {'cluster_number': cluster_number}
                                 for nn in new_cluster}
            # Label the cluster number of each cluster.
            nx.set_node_attributes(self.graph,
                                   cluster_name_dict)
            cluster_number += 1

        if self.max_number_of_edges is not None:
            self._prune_extra_edge()

    def _prune_extra_edge(self):
        edges = self.graph.edges
        num_edges_initially = self.graph.number_of_edges()
        num_to_remove = max(0, num_edges_initially - self.max_number_of_edges)
        choices = np.random.choice(list(range(len(edges))),
                                   size=num_to_remove,
                                   replace=False)
        to_remove = np.array(edges)[choices]
        to_remove = [tuple(tr) for tr in to_remove]
        self.graph.remove_edges_from(to_remove)
        print('Went from {} to {} edges after pruning.'.format(num_edges_initially,
                                                               self.graph.number_of_edges()))


    def _add_out_of_cluster_edges(self):
        '''
        For each cluster this function adds edges to nodes not in that cluster
        using the parameters:
            self.out_cluster_edge_mean
            self.out_cluster_edge_stdev
        '''

        for node in self._graph.nodes():
            num_new_edges = max(0, int(np.floor(np.random.normal(self.out_cluster_edge_mean,
                                             self.out_cluster_edge_stdev))))
            neighbors = list(self._graph.neighbors(node)) + [node]
            # List all potential new connections not in existing cluster.
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
        '''
        Main function for constructing the graph object.
        '''
        self._graph = nx.Graph()
        self._graph.add_nodes_from(list(range(self.N)),
                                   infected=False,
                                   symptomatic=False,
                                   recovered=False,
                                   infected_timestep=np.inf)
        self._construct_clusters()
        self._add_out_of_cluster_edges()
