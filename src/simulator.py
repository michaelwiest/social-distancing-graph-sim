import networkx as nx
from pathogen import Pathogen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
cm = plt.rcParams['axes.prop_cycle'].by_key()['color']


class Simulator(object):
    def __init__(self,
                 graph: nx.Graph,
                 pathogen: Pathogen
                 ):
        '''
        Simulator object. It receives a graph as constructed by
        GraphConstructor, namely GraphConstructor.graph

        Args:
            graph networkx.Graph: A graph object.
            pathogen Pathogen: a pathogen object that holds its relevant
                               infectivity information.
            symptomatic_edge_weight: the weight to reassign edges to following
                                     a symptomatic infection.

        TODO: Make it such that transition probabilities change if you
        are or are NOT symptomatic.
        '''
        self.graph = graph
        self.pathogen = pathogen

        self.simulation_data = pd.DataFrame(columns=['num_healthy',
                                                     'num_infected',
                                                     'num_recovered',
                                                     'num_symptomatic',
                                                     'num_asymptomatic'])

        self.display_lookup = {'asymptomatic': {'color': cm[1],
                                                'size': 300,
                                                'alpha': 0.8,
                                                'name': 'Asymptomatic'},
                               'uninfected': {'color': cm[0],
                                              'size': 100,
                                              'alpha': 0.6,
                                              'name': 'Uninfected'},
                               'recovered': {'color': cm[2],
                                             'size': 50,
                                             'alpha': 0.6,
                                             'name': 'Recovered'},
                               'symptomatic': {'color': cm[3],
                                               'size': 300,
                                               'alpha': 0.8,
                                               'name': 'Symptomatic'},
                               }

    def get_node_statuses(self,
                          which_graph=None):
        '''
        For each node in the graph get a readable status that can be used
        by self.display_lookup for picture generation.
        '''
        to_return = []
        if which_graph is None:
            which_graph = self.active_graph
        for n in which_graph.nodes():
            recovered = which_graph.nodes[n]['recovered']
            infected = which_graph.nodes[n]['infected']
            symptomatic = which_graph.nodes[n]['symptomatic']
            if infected:
                if symptomatic:
                    to_return.append('symptomatic')
                else:
                    to_return.append('asymptomatic')
            elif recovered:
                to_return.append('recovered')
            else:
                to_return.append('uninfected')
        return to_return

    def get_asymptomatic_nodes(self,
                               which_graph=None):
        '''
        Returns indices of infected nodes in the specified graph.
        '''
        if which_graph is None:
            which_graph = self.active_graph

        if which_graph is None:
            return set()
        else:
            asymptomatic = set([n[0]
                                for n in which_graph.nodes.data('symptomatic')
                                if n[1] is False])
            return list(set(asymptomatic).intersection(set(self.get_infected_nodes(which_graph))))

    def get_symptomatic_nodes(self,
                              which_graph=None):
        '''
        Returns indices of infected nodes in the specified graph.
        '''
        if which_graph is None:
            which_graph = self.active_graph
        if which_graph is None:
            return set()
        else:
            symptomatic = set([n[0]
                               for n in which_graph.nodes.data('symptomatic')
                               if n[1] is True])
            return list(set(symptomatic).intersection(set(self.get_infected_nodes(which_graph))))

    def get_infected_nodes(self,
                           which_graph=None):
        '''
        Returns indices of infected nodes in the specified graph.
        '''
        if which_graph is None:
            which_graph = self.active_graph
        if which_graph is None:
            return set()
        else:
            return [n[0] for n in
                    which_graph.nodes.data('infected')
                    if n[1] is True]

    def get_recovered_nodes(self,
                            which_graph=None):
        '''
        Returns indices of recovered nodes in the specified graph.
        '''
        if which_graph is None:
            which_graph = self.active_graph
        if which_graph is None:
            return set()
        else:
            return [n[0] for n in
                    which_graph.nodes.data('recovered')
                    if n[1] is True]

    def _infect_node(self,
                     node_number,
                     which_graph=None):
        '''
        For a specified node_number (index in the graph) attempt to infect that
        node. If it is already infected or recovered then nothing is done.
        Args:
            node_number int: the index of the node to attempt to infect.
        '''
        if which_graph is None:
            which_graph = self.active_graph
        infected = self.active_graph.nodes.data()[node_number]['infected']
        recovered = self.active_graph.nodes.data()[node_number]['recovered']
        if not (infected or recovered):
            which_graph.nodes[node_number]['infected'] = True
            which_graph.nodes[node_number]['symptomatic'] = False
            which_graph.nodes[node_number]['infected_timestep'] = self.sim_timestep
            coin_flip = np.random.uniform()
            # If they are symptomatic. There should be some more logic
            # in here to update edge weights if so.
            if coin_flip < self.pathogen.symptomatic_prob:
                which_graph.nodes[node_number]['symptomatic'] = True
                to_update = {e: {'weight': self.symptomatic_edge_weight}
                             for e in which_graph.edges(node_number)}
                nx.set_edge_attributes(which_graph, to_update)

    def _spread_from_node(self,
                          node_number,
                          which_graph=None):
        '''
        For a specified node_number (index in graph) attempt to spread the
        infection to all of its neighbors following a sampling of transmission
        probabilities.
        Args:
            node_number int: the index of the node whose neighbors to infect.
        '''
        if which_graph is None:
            which_graph = self.active_graph
        # Check if this node is even infected.
        if self.active_graph.nodes.data()[node_number]['infected']:
            neighbors = list(self.active_graph.neighbors(node_number))
            neighbor_edges = self.active_graph.edges(node_number)
            neighbor_weights = np.array([self.active_graph.get_edge_data(*ne)['weight']
                                         for ne in neighbor_edges])
            neighbor_weights *= self.pathogen.r_0
            coin_flips = np.random.uniform(size=len(neighbors))
            should_infect = (coin_flips < np.array(neighbor_weights))
            to_infect = [neighbors[i]
                         for i in range(len(neighbors))
                         if should_infect[i]]
            [self._infect_node(ti) for ti in to_infect]

    def _try_to_recover_node(self,
                             node_number,
                             which_graph=None):
        '''
        For a specified node_number (index in graph) attempt to have that
        node recover from infection. This is contingent upon how long it has
        been since infected.
        Args:
            node_number int: the index of the node to recover.

        # TODO: Could make this function also sample randomly for more
                stochasticity.
        '''
        if which_graph is None:
            which_graph = self.active_graph
        infected_timestep = which_graph.nodes.data(
        )[node_number]['infected_timestep']

        if self.sim_timestep - infected_timestep >= self.pathogen.timesteps_for_recovery:
            which_graph.nodes[node_number]['infected'] = False
            which_graph.nodes[node_number]['recovered'] = True
    #

    def prepare_simulation(self,
                           num_initially_infected,
                           symptomatic_edge_weight: float = 0.01):
        '''
        Instantiate the simulation with a number of infected nodes and
        how long it takes to recover.
        Args:
            num_initially_infected int: how many nodes start infected.
            timesteps_for_recovery int: how many timesteps to recover from infection.
        '''
        self.active_graph = self.graph.copy()
        self.sim_graphs = []
        self.sim_timestep = 0
        self.drawing_positions = None
        self.symptomatic_edge_weight = symptomatic_edge_weight

        to_infect = np.random.choice(list(self.graph.nodes()),
                                     num_initially_infected,
                                     replace=False).tolist()

        [self._infect_node(ti) for ti in to_infect]

        self.sim_graphs.append(self.active_graph)

    def step_simulation(self):
        '''
        Move the simulation forward by one step.
        '''
        if self.active_graph is None:
            raise ValueError('Please call prepare_simulation() first.')
        # Make a copy of the graph.
        self.active_graph = self.active_graph.copy()
        # Actually spread the infection
        [self._spread_from_node(ci) for ci in self.get_infected_nodes()]

        [self._try_to_recover_node(ci) for ci in self.get_infected_nodes()]

        # Increment our timestep.
        self.sim_timestep += 1
        # Store the graph.
        self.sim_graphs.append(self.active_graph)

    def simulate(self, num_steps=None):
        '''
        Do a full simulation.
        Args:
            num_steps int: if not specified then the simulation runs until
                           it becomes static (no more infections/recoveries).
                           Otherwise it runs for this many timesteps.
        '''
        if num_steps is not None:
            for i in range(num_steps):
                self.step_simulation()
                if len(self.get_infected_nodes()) == 0:
                    break
        else:
            i = 0
            while len(self.get_infected_nodes()) > 0:
                self.step_simulation()
                i += 1
        print('Simulation finished after {} steps'.format(i))

        # Store the information of the simulation into a pandas df.
        for i, sg in enumerate(self.sim_graphs):
            num_infected = len(self.get_infected_nodes(sg))
            num_recovered = len(self.get_recovered_nodes(sg))
            num_healthy = len(sg.nodes()) - (num_infected + num_recovered)
            num_symptomatic = len(self.get_symptomatic_nodes(sg))
            num_asymptomatic = len(self.get_asymptomatic_nodes(sg))
            self.simulation_data.loc[i] = [num_healthy, num_infected,
                                           num_recovered,
                                           num_symptomatic, num_asymptomatic]

    def animate_infection(self,
                          interval: float = 0.1,
                          granularity: int = 1,
                          keep_final: bool = False):
        '''
        Draws a series of graphs of the infection spreading.
        Args:
            interval float: how many seconds to show each frame.
            granularity int: how many timesteps to draw. larger is fewer frames of video.
            keep_final boolean: Whether or not to leave the last frame of the video.
        '''

        for i in list(range(len(self.sim_graphs)))[::granularity]:
            sg = self.sim_graphs[i]
            fig, ax = self.draw_graph(which_graph=sg)
            percent_infected = np.round(100.0 * (self.simulation_data.loc[i, 'num_infected'] /
                                                 self.simulation_data.loc[i].sum()), 3)

            ax.set_title('Time: {}\n{}% Infected'.format(i, percent_infected),
                         fontsize=32,
                         color='gray')
            plt.show(block=False)
            plt.pause(interval)

            if (i < len(self.sim_graphs) - 1):
                plt.close(fig)
            elif not keep_final:
                plt.close(fig)
            else:
                plt.show()

    def show_stats(self,
                   num_steps=None):
        '''
        Display the number of healthy, infected, and recovered
        over time in a stacked bar chart.
        '''
        if num_steps is None:
            df_to_use = self.simulation_data
        else:
            df_to_use = self.simulation_data[:, :num_steps]
        fig, ax = plt.subplots(1,
                               figsize=(10, 7))

        bottom = [0] * df_to_use.shape[0]

        ax.bar(range(df_to_use.shape[0]),
               df_to_use['num_symptomatic'].values,
               width=1,
               color=self.display_lookup['symptomatic']['color'],
               label=self.display_lookup['symptomatic']['name'])
        bottom += df_to_use['num_symptomatic'].values

        ax.bar(range(df_to_use.shape[0]),
               df_to_use['num_asymptomatic'].values,
               width=1,
               bottom=bottom,
               color=self.display_lookup['asymptomatic']['color'],
               label=self.display_lookup['asymptomatic']['name'])
        bottom += df_to_use['num_asymptomatic'].values

        ax.bar(range(df_to_use.shape[0]),
               df_to_use['num_healthy'].values,
               width=1,
               bottom=bottom,
               color=self.display_lookup['uninfected']['color'],
               label='Uninfected')

        bottom += df_to_use['num_healthy'].values
        ax.bar(range(df_to_use.shape[0]),
               df_to_use['num_recovered'].values,
               width=1,
               bottom=bottom,
               color=self.display_lookup['recovered']['color'],
               label='Recovered')

        ax.set_xlim(0, df_to_use.shape[0])
        ax.legend(loc='upper right', fontsize=14)
        ax.set_ylabel('Number of People', fontsize=24)
        ax.set_xlabel('Timestep', fontsize=24)
        fig.tight_layout()
        return fig, ax

    def draw_graph(self,
                   which_graph=None):
        '''
        Draw a particular graph.
        '''
        if which_graph is None:
            which_graph = self.graph
        if self.drawing_positions is None:
            self.drawing_positions = nx.spring_layout(which_graph)

        fig, ax = plt.subplots(1,
                               figsize=(10, 10))
        ax.set_xticks([])
        ax.set_yticks([])

        labs = {i: which_graph.nodes()[i]['cluster_number']
                for i in range(len(which_graph.nodes()))}
        statuses = self.get_node_statuses(which_graph=which_graph)
        colors = [self.display_lookup[statuses[i]]['color']
                  for i in range(len(which_graph.nodes()))]
        # alphas = [self.display_lookup[statuses[i]]['alpha']
        #           for i in range(len(which_graph.nodes()))]

        sizes = [self.display_lookup[statuses[i]]['size']
                 for i in range(len(which_graph.nodes()))]

        nx.draw_networkx_nodes(which_graph,
                               self.drawing_positions,
                               node_color=colors,
                               node_size=sizes,
                               ax=ax,
                               alpha=0.8,
                               edgecolors='white'
                               )
        nx.draw_networkx_edges(which_graph,
                               self.drawing_positions)

        return fig, ax
