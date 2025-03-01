import copy
import tools
from params import *
if Scenario_name == 'intersection':
    from scenario_environment import intersection_environment as environment
elif Scenario_name == 'merge':
    from scenario_environment import merge_environment as environment
elif Scenario_name == 'roundabout':
    from scenario_environment import roundabout_environment as environment
else:
    raise ValueError('no such environment, check Scenario_name in params')
from params import *

class Bayesian_Agent:
    """solve 1v1 nash game for now"""
    def __init__(self, hdv_info, cav_info, action_type):
        self.ego_info = hdv_info
        self.ego_state = [hdv_info.x, hdv_info.y, hdv_info.speed, hdv_info.heading, hdv_info.dis2des, hdv_info.entrance, hdv_info.exit]
        self.ego_central_vertices = environment.ALL_REF_LINE[self.ego_state[5]][self.ego_state[6]]
        self.ego_aggressiveness = hdv_info.aggressiveness

        self.inter_info = cav_info
        self.inter_state = [cav_info.x, cav_info.y, cav_info.speed, cav_info.heading, cav_info.dis2des, cav_info.entrance, cav_info.exit]
        self.inter_central_vertices = environment.ALL_REF_LINE[self.inter_state[5]][self.inter_state[6]]
        self.action_type = action_type
        # self.recede = self.if_recede()
        self.aggressiveness_distribution = [0.14, 0.41, 0.45]  # ['agg', 'nor', 'con']
        self.max_speed = self.get_max_speed()

    def get_max_speed(self):
        if self.ego_aggressiveness == 'agg':
            return Target_speed[0]
        elif self.ego_aggressiveness == 'nor':
            return Target_speed[1]
        elif self.ego_aggressiveness == 'con':
            return Target_speed[2]
        else:
            self.ego_aggressiveness = 'agg'
            print('input ego_info as cav, should not!')
            return Target_speed[0]

    def kinematic_model(self, state, action, temp):
        state_now = copy.deepcopy(state)
        state_now[2] += Action_space[action, 0] * Dt  # v'=v+a*Dt
        if state[0] == self.ego_state[0] and not temp:
            if state_now[2] > self.max_speed:
                state_now[2] = self.max_speed
            if state_now[2] < 0:
                state_now[2] = 0
        state_now[4] -= state_now[2] * Dt
        state_now[0], state_now[1] = tools.update_pos_from_dis2des_to_Cartesian(state_now[5], state_now[6], state_now[4])
        state_now[3] = tools.calculate_heading(state[0], state[1], state_now[0], state_now[1])
        return state_now

    def reward_weight(self, aggressiveness):
        if aggressiveness == 'agg':
            reward_weight = Weight_hv[0]
        elif aggressiveness == 'nor':
            reward_weight = Weight_hv[1]
        else:
            reward_weight = Weight_hv[2]
        return reward_weight

    def get_ttc_thr(self, aggressiveness):
        if aggressiveness == 'agg':
            ttc_thr = 2
        elif aggressiveness == 'nor':
            ttc_thr = 6
        else:
            ttc_thr = 7
        return ttc_thr

    def update_state(self, output_action=False):  # donnot forget for discrete action u need agrmax action; but for continuous, its minimize so reward should add '-' to find agrmax
        if not tools.if_passed_conflict_point(self.ego_info, self.inter_info):
            r = np.zeros(Action_length)
            for inter, inter_aggressiveness in enumerate(['agg', 'nor', 'con']):
                nash_equilibrium_solution = self.nash_equilibrium(inter_aggressiveness)
                if nash_equilibrium_solution == []:
                    inter_pure_strategy = 0
                else:
                    inter_pure_strategy = nash_equilibrium_solution[0][1]

                for ego_action in range(Action_length):
                    r[ego_action] += self.aggressiveness_distribution[inter] * \
                                     self.reward(ego_action, inter_pure_strategy, inter_vehicle_aggressiveness=inter_aggressiveness)[0]  # 0表示自身收益
            bayesian_pure_strategy = np.argmax(r)
            acc = Action_space[bayesian_pure_strategy, 0]
        else:
            acc = max(Acceleration_list)
        if not output_action:
            ego_info = tools.kinematic_model(self.ego_info, acc)
            return ego_info
        else:
            return acc

    # below is for discrete action
    def nash_equilibrium(self, inter_vehicle_aggressiveness):
        nash_matrix = np.zeros((Action_length, Action_length))
        ego_best_response, inter_best_response = self.get_best_response(inter_vehicle_aggressiveness=inter_vehicle_aggressiveness)
        for act in range(Action_length):
            nash_matrix[act, inter_best_response[act]] += 1
            nash_matrix[ego_best_response[act], act] += 1
        # print(nash_matrix)
        _ = [i.tolist() for i in np.where(nash_matrix == 2)]
        nash = list(zip(*_))
        return nash

    def get_best_response(self, inter_vehicle_aggressiveness):
        """输出收益最大的动作索引"""
        ego_reward_matrix = np.zeros((Action_length, Action_length))
        inter_reward_matrix = np.zeros((Action_length, Action_length))
        for act1 in range(Action_length):  # ego
            for act2 in range(Action_length):  # inter
                ego_reward, inter_reward = self.reward(act1=act1, act2=act2, inter_vehicle_aggressiveness=inter_vehicle_aggressiveness)
                ego_reward_matrix[act1, act2] = ego_reward
                inter_reward_matrix[act1, act2] = inter_reward

        # 计算best response
        inter_best_response = []
        ego_best_response = []
        for act in range(Action_length):
            inter_best_response.append(np.argmax(inter_reward_matrix[act, :]))
            ego_best_response.append(np.argmax(ego_reward_matrix[:, act]))
        return ego_best_response, inter_best_response

    def reward(self, act1, act2, inter_vehicle_aggressiveness):
        ego_state = self.kinematic_model(state=self.ego_state, action=act1, temp=True)
        inter_state = self.kinematic_model(state=self.inter_state, action=act2, temp=True)

        # 1st
        ego_dis2cv = np.amin(np.linalg.norm(self.ego_central_vertices - ego_state[0:2], axis=1))
        inter_dis2cv = np.amin(np.linalg.norm(self.inter_central_vertices - inter_state[0:2], axis=1))
        if environment.if_right_turning(self.ego_state[5], self.ego_state[6]) == 'rt':
            ego_reward1 = - max(0, ego_dis2cv) * 20
        else:
            ego_reward1 = - max(0.1, ego_dis2cv) * 10
        if environment.if_right_turning(self.inter_state[5], self.inter_state[6]) == 'rt':
            inter_reward1 = - max(0, inter_dis2cv) * 20
        else:
            inter_reward1 = - max(0.1, inter_dis2cv) * 10

        # 2st
        ego_reward2 = ego_state[2]
        inter_reward2 = inter_state[2]

        ego_destination = self.ego_central_vertices[-1]
        inter_destination = self.inter_central_vertices[-1]
        # 3rd
        ego_reward3 = - ((ego_state[0] - ego_destination[0])**2 + (ego_state[1] - ego_destination[1])**2)**0.5
        inter_reward3 = - ((inter_state[0] - inter_destination[0])**2 + (inter_state[1] - inter_destination[1])**2)**0.5

        # 4th
        dis = ((ego_state[0] - inter_state[0])**2 + (ego_state[1] - inter_state[1])**2)**0.5
        ego_ttc_thr = self.get_ttc_thr(self.ego_aggressiveness)
        inter_ttc_thr = self.get_ttc_thr(inter_vehicle_aggressiveness)
        ego_ttc = (dis / ego_state[2]) / ego_ttc_thr
        inter_ttc = (dis / inter_state[2]) / inter_ttc_thr
        ego_reward4 = (- 1/ego_ttc)
        inter_reward4 = (- 1/inter_ttc)
        # add up reward
        ego_reward = np.array([ego_reward1, ego_reward2, ego_reward3, ego_reward4])
        inter_reward = np.array([inter_reward1, inter_reward2, inter_reward3, inter_reward4])
        ego_reward_weight = self.reward_weight(self.ego_aggressiveness)
        inter_reward_weight = self.reward_weight(inter_vehicle_aggressiveness)
        return np.dot(ego_reward, ego_reward_weight), np.dot(inter_reward, inter_reward_weight)

    def state_without_inter_vehicle(self):
        reward_without_iv = []
        for act in range(Action_length):
            self_state = self.kinematic_model(state=self.ego_state, action=act, temp=True)
            central_vertices = self.ego_central_vertices
            dis2cv = np.amin(np.linalg.norm(central_vertices - self_state[0:2], axis=1))
            if environment.if_right_turning(self.ego_state[5], self.ego_state[6]) == 'rt':
                reward1 = - max(0.1, dis2cv) * 20
            else:
                reward1 = - max(0.1, dis2cv) * 10
            reward2 = self_state[2]
            destination = self.ego_central_vertices[-1]
            reward3 = - abs(self_state[0] - destination[0]) - abs(self_state[1] - destination[1]) * 2
            reward4 = 0
            reward = np.array([reward1, reward2, reward3, reward4])
            reward_weight = self.reward_weight(self.ego_aggressiveness)
            reward_without_iv.append(np.dot(reward, reward_weight) + reward2)
            # print(act, 'reward', reward, np.dot(reward, reward_weight))
        max_reward_action = np.argmax(reward_without_iv)
        self_state = self.kinematic_model(state=self.ego_state, action=max_reward_action, temp=False)
        return self_state

