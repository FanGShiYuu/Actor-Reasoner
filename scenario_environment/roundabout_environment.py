import copy
import math
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splrep, splev
from params import *

def RGB_to_Hex(rgb):
    RGB = rgb.split(',')
    color = '#'
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

def scenario_outfit(ax, color=RGB_to_Hex('202,202,202')):
    radius = ROUNDABOUT_R - 5
    theta = np.linspace(0, 2 * np.pi, 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    ax.plot(x, y, c=color)

    radius = ROUNDABOUT_R
    theta = np.linspace(0, 2 * np.pi, 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    ax.plot(x, y, c=color)

    ax.plot([20, 100], [5, 5], c=color)
    ax.plot([20, 100], [-5, -5], c=color)
    ax.plot([-100, -20], [5, 5], c=color)
    ax.plot([-100, -20], [-5, -5], c=color)
    #
    ax.plot([5, 5], [20, 100], c=color)
    ax.plot([-5, -5], [20, 100], c=color)
    ax.plot([5, 5], [-100, -20], c=color)
    ax.plot([-5, -5], [-100, -20], c=color)

    ax.plot([0, 0], [25, 100], c='black', linestyle='--', linewidth=0.5,alpha=0.5)  # 虚线
    ax.plot([0, 0], [-100, -25], c='black', linestyle='--', linewidth=0.5,alpha=0.5)
    ax.plot([-100, -25], [0, 0], c='black', linestyle='--', linewidth=0.5,alpha=0.5)
    ax.plot([25, 100], [0, 0], c='black', linestyle='--', linewidth=0.5,alpha=0.5)


def smooth_ployline(cv_init, point_num=3000):
    cv = cv_init
    list_x = cv[:, 0]
    list_y = cv[:, 1]
    if type(cv) is not np.ndarray:
        cv = np.array(cv)
    delta_cv = cv[1:, ] - cv[:-1, ]
    s_cv = np.linalg.norm(delta_cv, axis=1)

    s_cv = np.array([0] + list(s_cv))
    s_cv = np.cumsum(s_cv)
    bspl_x = splrep(s_cv, list_x, s=0.1)
    bspl_y = splrep(s_cv, list_y, s=0.1)
    # values for the x axis
    s_smooth = np.linspace(0, max(s_cv), point_num)
    # get y values from interpolated curve
    x_smooth = splev(s_smooth, bspl_x)
    y_smooth = splev(s_smooth, bspl_y)
    new_cv = np.array([x_smooth, y_smooth]).T
    delta_new_cv = new_cv[1:, ] - new_cv[:-1, ]
    s_accumulated = np.cumsum(np.linalg.norm(delta_new_cv, axis=1))
    s_accumulated = np.concatenate(([0], s_accumulated), axis=0)
    return new_cv, s_accumulated

def if_going_straight(entrance, exit):
    if entrance == 'w':
        return True
    else:
        return False


def if_right_turning(entrance, exit):
    if entrance == 's':
        return True
    else:
        return False

def if_left_turning(entrance, exit):
    return False

def enter_roundabout_ref_line(entrance, exit):
    cv_init = None
    if entrance == 's':
        cv_init = np.array([[2.5, -25], [2.5, -24], [2.5, -23], [2.5, -22], [2.9, -20], [5, -18], [7, -16], [7.7, -15.75], [8, -15.6]])
    if entrance == 'n':
        cv_init = np.array([[-2.5, 25], [-2.5, 24], [-2.5, 23], [-2.5, 22], [-2.9, 20], [-5, 18], [-7, 16], [-7.7, 15.75], [-8, 15.6]])
    if entrance == 'e':
        cv_init = np.array([[25, 2.5], [24, 2.5], [23, 2.5], [22, 2.5], [20, 2.9], [18, 5], [16, 7], [15.75, 7.7], [15.6, 8]])
    if entrance == 'w':
        cv_init = np.array([[-25, -2.5], [-24, -2.5], [-23, -2.5], [-22, -2.5], [-20, -2.9], [-18, -5], [-16, -7], [-15.75, -7.7], [-15.6, -8]])

    assert cv_init is not None
    cv_smoothed, s_accumulated = smooth_ployline(cv_init)
    return cv_smoothed  # , s_accumulated

def exit_roundabout_ref_line(entrance, exit):
    cv_init = None
    if exit == 's':
        cv_init = np.array([[-8, -15.6], [-7.7, -15.75], [-7, -16], [-5, -18], [-2.9, -20], [-2.5, -22], [-2.5, -23], [-2.5, -24], [-2.5, -25]])
    if exit == 'n':
        cv_init = np.array([[8, 15.6], [7.7, 15.75], [7, 16], [5, 18], [2.9, 20], [2.5, 22], [2.5, 23], [2.5, 24], [2.5, 25]])
    if exit == 'e':
        cv_init = np.array([[15.6, -8], [15.75, -7.7], [16, -7], [18, -5], [20, -2.9], [22, -2.5], [23, -2.5], [24, -2.5], [25, -2.5]])
    if exit == 'w':
        cv_init = np.array([[-15.6, 8], [-15.75, 7.7], [-16, 7], [-18, 5], [-20, 2.9], [-22, 2.5], [-23, 2.5], [-24, 2.5], [-25, 2.5]])

    assert cv_init is not None
    cv_smoothed, s_accumulated = smooth_ployline(cv_init)
    return cv_smoothed  # , s_accumulated

def roundabout_ref_line(entrance, exit):
    theta = None
    radius = 17.5
    if entrance == 's':
        if exit == 'e':
            theta = np.linspace(1.65 * np.pi, 1.85 * np.pi, 2000)
        elif exit == 'n':
            theta = np.linspace(-0.35 * np.pi, 0.35 * np.pi, 4000)
        elif exit == 'w':
            theta = np.linspace(-0.35 * np.pi, 0.85 * np.pi, 6000)
        else:
            theta = np.linspace(-0.35 * np.pi, 1.35 * np.pi, 8000)

    if entrance == 'e':
        if exit == 'n':
            theta = np.linspace(0.15 * np.pi, 0.35 * np.pi, 2000)
        elif exit == 'w':
            theta = np.linspace(0.15 * np.pi, 0.85 * np.pi, 4000)
        elif exit == 's':
            theta = np.linspace(0.15 * np.pi, 1.35 * np.pi, 6000)
        else:
            theta = np.linspace(0.15 * np.pi, 1.85 * np.pi, 8000)

    if entrance == 'n':
        if exit == 'w':
            theta = np.linspace(0.65 * np.pi, 0.85 * np.pi, 2000)
        elif exit == 's':
            theta = np.linspace(0.65 * np.pi, 1.35 * np.pi, 4000)
        elif exit == 'e':
            theta = np.linspace(0.65 * np.pi, 1.85 * np.pi, 6000)
        else:
            theta = np.linspace(-1.35 * np.pi, 0.35 * np.pi, 8000)

    if entrance == 'w':
        if exit == 's':
            theta = np.linspace(1.15 * np.pi, 1.35 * np.pi, 2000)
        elif exit == 'e':
            theta = np.linspace(1.15 * np.pi, 1.85 * np.pi, 4000)
        elif exit == 'n':
            theta = np.linspace(-0.85 * np.pi, 0.35 * np.pi, 6000)
        else:
            theta = np.linspace(-0.85 * np.pi, 0.85 * np.pi, 8000)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    ref_line = np.vstack((x, y))
    return ref_line.T


def record_ref_line_distance2exit(ref_line):
    """
    for calculate vehicle distance to exit which computation expensive
    store each point distance to exit
    only employ when T=0s or vehicle changes ref_line
    """
    cv = ref_line
    gap_list = np.zeros(len(cv))
    for point in range(len(ref_line) - 1):
        gap = np.sqrt((cv[point, 0] - cv[point + 1, 0]) ** 2 + (cv[point, 1] - cv[point + 1, 1]) ** 2)
        gap_list[point:] += gap
    ref_line_distance2exit = max(gap_list) - np.array(gap_list)
    # ref_line_distance2exit = np.flipud(gap_list)
    # print(ref_line_distance2exit[0:10])
    return ref_line_distance2exit

def entrance_ref_line(entrance, exit):
    """
    ref_line of entrance
    """
    ref_line = None
    if entrance == 'w':
        x = np.linspace(-100, -25, 2000)
        y = -2.5 * np.ones_like(x)
        ref_line = np.vstack((x, y))
    if entrance == 'e':
        x = np.linspace(100, 25, 2000)
        y = 2.5 * np.ones_like(x)
        ref_line = np.vstack((x, y))
    if entrance == 'n':
        y = np.linspace(100, 25, 2000)
        x = -2.5 * np.ones_like(y)
        ref_line = np.vstack((x, y))
    if entrance == 's':
        y = np.linspace(-100, -25, 2000)
        x = 2.5 * np.ones_like(y)
        ref_line = np.vstack((x, y))
    return ref_line.T

def exit_ref_line(entrance, exit):
    """
    ref_line of exit
    """
    ref_line = None
    if exit == 'w':
        x = np.linspace(-25.1, -100, 2000)
        y = 2.5 * np.ones_like(x)
        ref_line = np.vstack((x, y))
    if exit == 'e':
        x = np.linspace(25.1, 100, 2000)
        y = -2.5 * np.ones_like(x)
        ref_line = np.vstack((x, y))
    if exit == 'n':
        y = np.linspace(25.1, 100, 2000)
        x = 2.5 * np.ones_like(y)
        ref_line = np.vstack((x, y))
    if exit == 's':
        y = np.linspace(-25.1, -100, 2000)
        x = -2.5 * np.ones_like(y)
        ref_line = np.vstack((x, y))
    return ref_line.T

def concatenate_ref_lane(entrance, exit):
    ref_lane1 = entrance_ref_line(entrance, exit)
    ref_lane2 = enter_roundabout_ref_line(entrance, exit)
    ref_lane3 = roundabout_ref_line(entrance, exit)
    ref_lane4 = exit_roundabout_ref_line(entrance, exit)
    ref_lane5 = exit_ref_line(entrance, exit)
    ref_lane = np.vstack((ref_lane1, ref_lane2))
    ref_lane = np.vstack((ref_lane, ref_lane3))
    ref_lane = np.vstack((ref_lane, ref_lane4))
    ref_lane = np.vstack((ref_lane, ref_lane5))
    return ref_lane

def default_exit_and_state(entrance, exit):
    """
    default exit of each entrance
    """
    state = None
    velocity = np.random.uniform(6, 9)  # initial velocity will be [6, 9)
    heading = None

    if entrance == 'w':
        state = [-100, -2.5]
        heading = 0 * math.pi
    if entrance == 'e':
        state = [100, 2.5]
        heading = math.pi
    if entrance == 'n':
        state = [-2.5, 100]
        heading = 1.5 * math.pi
    if entrance == 's':
        state = [2.5, -100]
        heading = 0.5 * math.pi

    state.append(velocity)
    state.append(heading)
    state.append(ALL_GAP_LIST[entrance][exit][0])
    state.append(entrance)
    state.append(exit)
    ori_dis2des = ALL_GAP_LIST[entrance][exit][0]
    if entrance == 'w':
        ori_dis2des -= 20
    return state[0], state[1], velocity, heading, ori_dis2des-20, SPEED_LIMIT

def record_all_possible_ref_line():
    """
    instead of calculate ref_line at each time
    store all possible ref_line and its distance2exit(gap list)
    """
    possible_ref_line_list = []
    possible_ref_line_distance2exit_list = []
    total_length = []
    for entrance in POSSIBLE_ENTRANCE:
        entrance_possible_ref_line = []
        entrance_possible_ref_line_distance2exit = []
        entrance_possible_total_length = []
        for exit in ENTRANCE_EXIT_RELATION[entrance]:
            ref_line = concatenate_ref_lane(entrance, exit)
            entrance_possible_ref_line.append(ref_line)
            entrance_possible_ref_line_distance2exit.append(record_ref_line_distance2exit(ref_line))
            entrance_possible_total_length.append(max(record_ref_line_distance2exit(ref_line)))
        possible_ref_line_list.append(dict(zip(ENTRANCE_EXIT_RELATION[entrance], entrance_possible_ref_line)))
        possible_ref_line_distance2exit_list.append(dict(zip(ENTRANCE_EXIT_RELATION[entrance], entrance_possible_ref_line_distance2exit)))
        total_length.append(dict(zip(ENTRANCE_EXIT_RELATION[entrance], entrance_possible_total_length)))
    return dict(zip(POSSIBLE_ENTRANCE, possible_ref_line_list)), dict(zip(POSSIBLE_ENTRANCE, possible_ref_line_distance2exit_list)), dict(zip(POSSIBLE_ENTRANCE, total_length))


def find_dis2des(entrance, exit, x, y):
    ref_line = ALL_REF_LINE[entrance][exit]
    gap_list = ALL_GAP_LIST[entrance][exit]
    index = np.argmin(np.sqrt((ref_line[:,0] - x)**2 + (ref_line[:,1] - y)**2))
    dis2des = gap_list[index]
    return dis2des

print('Initialize roundabout environment...')
ALL_REF_LINE, ALL_GAP_LIST, REF_LINE_TOTAL_LENGTH = record_all_possible_ref_line()
CONFLICT_RELATION = {'s': {'e': {'we': 99.78564593363033, 'wn': 99.78564593363033}, 'w': {'we': 154.76351633116838, 'wn': 154.76351633116838}, 'n': {'we': 127.27458121290213, 'wn': 127.27458121290213}}, 'w': {'e': {'se': 99.66532772509943, 'sw': 99.66532772509943, 'sn': 99.66532772509943}, 'n': {'se': 127.15365197519074, 'sw': 127.15365197519074, 'sn': 127.15365197519074}}}
CONFLICT_RELATION_STATE = {'s': {'e': {'we': (6.362653632019546, -16.60060155498124), 'wn': (6.362653632019546, -16.60060155498124)},
                                 'w': {'we': (6.362653632019546, -16.60060155498124), 'wn': (6.362653632019546, -16.60060155498124)},
                                 'n': {'we': (6.362653632019546, -16.60060155498124), 'wn': (6.362653632019546, -16.60060155498124)}},
                           'w': {'e': {'se': (6.362653632019546, -16.60060155498124), 'sw': (6.362653632019546, -16.60060155498124), 'sn': (6.362653632019546, -16.60060155498124)},
                                 'n': {'se': (6.362653632019546, -16.60060155498124), 'sw': (6.362653632019546, -16.60060155498124), 'sn': (6.362653632019546, -16.60060155498124)}}}
print('Initialize done')
