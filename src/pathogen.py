import numpy as np

class Pathogen(object):
    def __init__(self,
                 r_0: float,
                 timesteps_for_recovery: int,
                 symptomatic_prob: float=0.0,
                 ):
        self.r_0 = r_0
        self.timesteps_for_recovery = timesteps_for_recovery
        self.symptomatic_prob = symptomatic_prob
