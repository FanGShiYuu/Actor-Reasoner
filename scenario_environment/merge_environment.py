import matplotlib.pyplot as plt
from params import *
import numpy as np
from scipy.interpolate import splrep, splev
import math

def RGB_to_Hex(rgb):
    RGB = rgb.split(',')
    color = '#'
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    print(color)
    return color

def scenario_outfit(ax, color=RGB_to_Hex('202,202,202')):
    # ax.plot(np.arange(0, 200, 1), 5 * np.ones(shape=(200,)), c=color)  # 横线
    # ax.plot(np.arange(0, 60, 1), np.zeros(shape=(60,)), c=color)
    # ax.plot(np.arange(65, 170, 1), np.zeros(shape=(105,)), c=color)
    # ax.plot(np.arange(175, 200, 1), np.zeros(shape=(25,)), c=color)

    ax.plot([0, 200], [5, 5], c=color)  # 横线
    ax.plot([0, 80], [0, 0], c=color)
    ax.plot([105, 150], [0, 0], c=color)
    ax.plot([175, 200], [0, 0], c=color)

    ax.plot([30, 80], [-10, 0], c=color)  # 横线
    ax.plot([55, 105], [-10, 0], c=color)
    ax.plot([150, 200], [0, -10], c=color)
    ax.plot([175, 225], [0, -10], c=color)

    ax.plot([42.5, 92.5], [-10, 0], c=color, linestyle='--')
    ax.plot([162.5, 212.5], [0, -10], c=color, linestyle='--')
    ax.plot([0, 200], [2.5, 2.5], c=color, linestyle='--')

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
    if entrance == 's':
        return True
    else:
        return False

def if_right_turning(entrance, exit):
    return False

def if_left_turning(entrance, exit):
    if entrance == 'm':
        return True
    else:
        return False

def merging_area_ref_line(entrance, exit):
    cv_init = None
    if entrance == 's' and exit == 'm':
        cv_init = np.array([[145, 2.5], [152, 2.5], [157, 2.5], [170, -1.7], [177, -3], [179, -3.2], [180, -3.6]])
    if entrance == 's' and exit == 's':
        cv_init = np.array([[75, 2.5], [100, 2.5], [110, 2.5], [120, 2.5], [155, 2.5]])
    if entrance == 'm' and exit == 's':
        cv_init = np.array([[75, -3.6], [76, -3.2], [78, -3], [84, -1.8], [100, 2.5], [105, 2.5], [110, 2.5], [115, 2.5]])
    assert cv_init is not None
    cv_smoothed, s_accumulated = smooth_ployline(cv_init)
    return cv_smoothed  # , s_accumulated


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
    if entrance == 's' and exit == 'm':
        x = np.linspace(0, 145, 2000)
        y = 2.5 * np.ones_like(x)
        ref_line = np.vstack((x, y))

    if entrance == 's' and exit == 's':
        x = np.linspace(0, 75, 2000)
        y = 2.5 * np.ones_like(x)
        ref_line = np.vstack((x, y))

    if entrance == 'm' and exit == 's':
        x = np.linspace(42.5, 75, 2000)
        y = np.linspace(-10, -3.5, 2000)
        ref_line = np.vstack((x, y))
    return ref_line.T

def exit_ref_line(entrance, exit):
    """
    ref_line of exit
    """
    ref_line = None
    if entrance == 's' and exit == 'm':
        x = np.linspace(180, 212.5, 2000)
        y = np.linspace(-3.5, -10, 2000)
        ref_line = np.vstack((x, y))

    if entrance == 's' and exit == 's':
        x = np.linspace(155, 200, 2000)
        y = 2.5 * np.ones_like(x)
        ref_line = np.vstack((x, y))

    if entrance == 'm' and exit == 's':
        x = np.linspace(115, 200, 2000)
        y = 2.5 * np.ones_like(x)
        ref_line = np.vstack((x, y))
    return ref_line.T


def concatenate_ref_lane(entrance, exit):
    ref_lane1 = entrance_ref_line(entrance, exit)
    ref_lane2 = merging_area_ref_line(entrance, exit)
    ref_lane3 = exit_ref_line(entrance, exit)
    ref_lane = np.vstack((ref_lane1, ref_lane2))
    ref_lane = np.vstack((ref_lane, ref_lane3))
    return ref_lane

def default_exit_and_state(entrance, exit):
    """
    default exit of each entrance
    """
    state = None
    velocity = np.random.uniform(6, 9)  # initial velocity will be [6, 9)
    heading = None

    if entrance == 's':
        state = [0, 2.5]
        heading = math.pi
    if entrance == 'm':
        state = [42.5, -10]
        heading = math.pi

    state.append(velocity)
    state.append(heading)
    state.append(ALL_GAP_LIST[entrance][exit][0])
    state.append(entrance)
    state.append(exit)
    ori_dis2des = ALL_GAP_LIST[entrance][exit][0]
    if entrance == 's':
        ori_dis2des -= 20
    return state[0], state[1], velocity, heading, ori_dis2des-15, SPEED_LIMIT

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

print('Initialize merge environment...')
ALL_REF_LINE, ALL_GAP_LIST, REF_LINE_TOTAL_LENGTH = record_all_possible_ref_line()
CONFLICT_RELATION = {'s': {'s': {'ms': 97}}, 'm': {'s': {'ss': 100, 'sm': 100}}}
CONFLICT_RELATION_STATE = {'s': {'s': {'ms': (100, 2.5)}}, 'm': {'s': {'ss': (100, 2.5), 'sm': (100, 2.5)}}}
print('Initialize done')