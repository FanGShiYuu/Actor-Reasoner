import math
import numpy as np

# Scenario name
Scenario_name = 'intersection'
# HV params
Action_space = np.array([[0, 0], [2, 0], [-2, 0], [-4, 0]])
Action_length = len(Action_space)
Target_speed = [10, 10, 8]
Weight_hv = [[1.56, 0, 8.33, 3.69], [1.72, 0, 8.2, 5.7], [2.1, 0, 7.79, 8.44]]  # agg nor con
Acceleration_list = [0, 2, -2, -4, 0, 0]
Acceleration_bds = [min(Acceleration_list), max(Acceleration_list)]
Dt = 0.1
Pi = math.pi
# CAV params
SPEED_LIMIT = 8
# Environment info
INTERSECTION_POSSIBLE_ENTRANCE = ['n2', 'n3', 'e2', 'e3', 's2', 's1', 'w2', 'w1']  # possible entrance
INTERSECTION_ENTRANCE_EXIT_RELATION = {'n2': ['w2', 's2', 's3'], 'n3': ['e1', 'e2', 's3'],
                          'e2': ['n2', 'w2', 'w3'], 'e3': ['s2', 's3', 'w3'],
                          's2': ['n1', 'n2', 'e2'], 's1': ['w2', 'w3', 'n1'],
                          'w2': ['e1', 'e2', 's2'], 'w1': ['n1', 'n2', 'e1']}
MERGE_POSSIBLE_ENTRANCE = ['s', 'm']  # possible entrance
MERGE_ENTRANCE_EXIT_RELATION = {'s': ['s'], 'm': ['s']}
ROUNDABOUT_POSSIBLE_ENTRANCE = ['w', 's']  # possible entrance
ROUNDABOUT_ENTRANCE_EXIT_RELATION = {'w': ['e', 'n'], 's': ['e', 'w', 'n']}
ROUNDABOUT_R = 20

if Scenario_name == 'intersection':
    POSSIBLE_ENTRANCE = INTERSECTION_POSSIBLE_ENTRANCE
    ENTRANCE_EXIT_RELATION = INTERSECTION_ENTRANCE_EXIT_RELATION
    STOP_LINE = {'n2':70, 'n3':70, 'e2':70, 'e3':70, 's2':70, 's1':70, 'w2':70, 'w1':70}
elif Scenario_name == 'merge':
    POSSIBLE_ENTRANCE = MERGE_POSSIBLE_ENTRANCE
    ENTRANCE_EXIT_RELATION = MERGE_ENTRANCE_EXIT_RELATION
    STOP_LINE = {'s': 75, 'm': 40}

elif Scenario_name == 'roundabout':
    POSSIBLE_ENTRANCE = ROUNDABOUT_POSSIBLE_ENTRANCE
    ENTRANCE_EXIT_RELATION = ROUNDABOUT_ENTRANCE_EXIT_RELATION
    STOP_LINE = {'w': 100, 's': 70}
else:
    raise ValueError('no such environment, check Scenario_name in params')


