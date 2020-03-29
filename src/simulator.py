import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
cm = plt.rcParams['axes.prop_cycle'].by_key()['color']

display_lookup = {'infected': {'color': cm[1],
                               'size': 300,
                               'alpha': 0.8},
                  'uninfected': {'color': cm[0],
                               'size': 100,
                               'alpha': 0.6},
                  'recovered': {'color': cm[2],
                               'size': 100,
                               'alpha': 0.6}
                }



class Simulator(object):
    def __init__(self, graph,
                 symptomatic_prob: float=0.0,
                 symptomatic_edge_weight: float=0.01):
        self.graph = graph
        self.symptomatic_prob = symptomatic_prob
        self.symptomatic_edge_weight = symptomatic_edge_weight

        self.simulation_data = pd.DataFrame(columns=['num_healthy',
                                                     'num_infected',
                                                     'num_recovered'])

    def get_node_statuses(self,
                          which_graph=None):
        to_return = []
        if which_graph is None:
            which_graph = self.active_graph
        for n in which_graph.nodes():
            recovered = which_graph.nodes[n]['recovered']
            infected = which_graph.nodes[n]['infected']
            if infected:
                # Something else in here for symptomatic
                to_return.append('infected')
            elif recovered:
                to_return.append('recovered')
            else:
                to_return.append('uninfected')
        return to_return

    def get_infected(self,
                               which_graph=None):
        if which_graph is None:
            which_graph = self.active_graph
        if which_graph is None:
            return set()
        else:
            return [n[0] for n in
                    which_graph.nodes.data('infected')
                    if n[1] is True]

    def get_recovered(self,
                                which_graph=None):
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
            if coin_flip < self.symptomatic_prob:
                self.graph.nodes[node_number]['symptomatic'] = True

    def _spread_from_node(self,
                         node_number,
                         which_graph=None):
        if which_graph is None:
            which_graph = self.active_graph
        # Check if this node is even infected.
        if self.active_graph.nodes.data()[node_number]['infected']:
            neighbors = list(self.active_graph.neighbors(node_number))
            neighbor_edges = self.active_graph.edges(node_number)
            neighbor_weights = [self.active_graph.get_edge_data(*ne)['weight']
                                for ne in neighbor_edges]
            coin_flips = np.random.uniform(size=len(neighbors))
            should_infect = (coin_flips < np.array(neighbor_weights))
            to_infect = [neighbors[i]
                         for i in range(len(neighbors))
                         if should_infect[i]]
            [self._infect_node(ti) for ti in to_infect]

    def _try_to_recover_node(self,
                            node_number,
                            which_graph=None):
        if which_graph is None:
            which_graph = self.active_graph
        infected_timestep = which_graph.nodes.data()[node_number]['infected_timestep']

        if self.sim_timestep - infected_timestep >= self.timesteps_for_recovery:
            which_graph.nodes[node_number]['infected'] = False
            which_graph.nodes[node_number]['recovered'] = True
    #

    def prepare_simulation(self,
                           num_intially_infected,
                           timesteps_for_recovery=14):
        self.timesteps_for_recovery = timesteps_for_recovery
        self.active_graph = self.graph.copy()
        self.sim_graphs = []
        self.sim_timestep = 0
        self.drawing_positions = None

        to_infect = np.random.choice(list(self.graph.nodes()),
                                     num_intially_infected,
                                     replace=False).tolist()

        [self._infect_node(ti) for ti in to_infect]

        self.sim_graphs.append(self.active_graph)

    def step_simulation(self):
        if self.active_graph is None:
            raise ValueError('Please call prepare_simulation() first.')
        # Make a copy of the graph.
        self.active_graph = self.active_graph.copy()
        # Actually spread the infection
        [self._spread_from_node(ci) for ci in self.get_infected()]

        [self._try_to_recover_node(ci) for ci in self.get_infected()]

        # Increment our timestep.
        self.sim_timestep += 1
        # Store the graph.
        self.sim_graphs.append(self.active_graph)

    def simulate(self, num_steps=None):
        if num_steps is not None:
            for i in range(num_steps):
                self.step_simulation()
                if len(self.get_infected()) == 0:
                    break
        else:
            i = 0
            while len(self.get_infected()) > 0:
                self.step_simulation()
                i += 1
        print('Simulation finished after {} steps'.format(i))

        # Store the information of the simulation.
        for i, sg in enumerate(self.sim_graphs):
            num_infected = len(self.get_infected(sg))
            num_recovered = len(self.get_recovered(sg))
            num_healthy = len(sg.nodes()) - (num_infected + num_recovered)
            self.simulation_data.loc[i] = [num_healthy, num_infected, num_recovered]


    def animate_infection(self,
                          interval=0.1,
                          granularity=1,
                          keep_final=True):
        for i, sg in enumerate(self.sim_graphs):
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
        if num_steps is None:
            df_to_use = self.simulation_data
        else:
            df_to_use = self.simulation_data[:, :num_steps]
        fig, ax = plt.subplots(1,
                               figsize=(10, 7))
        ax.bar(range(df_to_use.shape[0]),
                df_to_use['num_infected'].values,
                # bottom=bottom,
                color=display_lookup['infected']['color'],
                label='Infected')
        bottom = df_to_use['num_infected'].values
        ax.bar(range(df_to_use.shape[0]),
                df_to_use['num_healthy'].values,
                bottom=bottom,
                color=display_lookup['uninfected']['color'],
                label='Uninfected')

        bottom += df_to_use['num_healthy'].values
        ax.bar(range(df_to_use.shape[0]),
                df_to_use['num_recovered'].values,
                bottom=bottom,
                color=display_lookup['recovered']['color'],
                label='Recovered')

        ax.legend(loc='upper right', fontsize=18)
        ax.set_xlabel('Number of People', fontsize=24)
        ax.set_ylabel('Timestep', fontsize=24)
        return fig, ax

    def draw_graph(self,
                   which_graph=None):
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
        colors = [display_lookup[statuses[i]]['color']
                  for i in range(len(which_graph.nodes()))]
        # alphas = [display_lookup[statuses[i]]['alpha']
        #           for i in range(len(which_graph.nodes()))]

        sizes = [display_lookup[statuses[i]]['size']
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
