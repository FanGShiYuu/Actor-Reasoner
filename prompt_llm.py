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

class PRE_DEF_PROMPT():
    """
    These rules can be modified to test if changing prompt leads to different behaviour pattern of our agent.
    """

    def __init__(self):
        self.SYSTEM_MESSAGE_PREFIX = """You are now act as a autonomous vehicle motion planner, who generate safe decision. 
    Except for generate decision, to improve safety, you should also share your intention to express what you going to do to your surrounding vehicle. 
    Now you are driving at an intersection."""

        self.TRAFFIC_RULES = """
    1. Try to keep a safe distance to the car in front of you.
    2. DO NOT change lane frequently. If you want to change lane, double-check the safety of vehicles on target lane.
    """

        self.DECISION_CAUTIONS = """
    1. You must output a decision when you finish this task. Your final output decision must be unique and not ambiguous. For example you cannot say "I can either keep lane or accelerate at current time".
    2. In every conflict between you and the above vehicle, only one vehicle can pass first at each conflict. Based on above information, share your intention to other vehicles
    3. Your decision and intention to surrounding vehicle should be consistent.

    """
    # 2. You need to always remember your vehicle ID, your available actions and your surrounding vehicle (which you should give suggestions).
    # 3. Once you have a decision, you should check the safety with all the vehicles affected by your decision.
    # 4. If you verify a decision is unsafe, you should start a new one and verify its safety again from scratch.

    def get_traffic_rules(self):
        return self.TRAFFIC_RULES

    def get_decision_cautions(self):
        return self.DECISION_CAUTIONS


ACTIONS_DESCRIPTION = {
    'IDLE': 'remain in the current lane with current speed',
    'FASTER': 'accelerate the vehicle',
    'SLOWER': 'decelerate the vehicle'
}

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func
    return decorator

class getAvailableActions:
    # def __init__(self, ) -> None:
    @prompts(name='Get Available Actions',
             description="""Useful before you make decisions, this tool let you know what are your available actions in this situation. The input to this tool should be 'ego'.""")
    def _get_available_actions(self, ego_info):
        """
        Get the list of currently available actions.
        Lane changes are not available on the boundary of the road, and speed changes are not available at
        maximal or minimal speed.
        :return: the list of available actions
        """
        # if not isinstance(self.action_type, DiscreteMetaAction):
        #     raise ValueError("Only discrete meta-actions can be unavailable.")
        actions = ['IDLE']
        if ego_info.speed < ego_info.max_speed:
            actions.append('FASTER')
        if ego_info.speed > 0:
            actions.append('SLOWER')
        return actions

    def inference(self, ego_info) -> str:
        outputPrefix = 'You can ONLY use one of the following actions: IDLE, FASTER, SLOWER\n '
        # availableActions = self._get_available_actions(ego_info)
        # for action in availableActions:
        #     outputPrefix += action + '--' + ACTIONS_DESCRIPTION[action] + ':'
        #     if 'IDLE' in availableActions:
        #         outputPrefix += 'You should check idle action as FIRST priority. \n'
        #     if 'FASTER' in availableActions:
        #         outputPrefix += 'Consider acceleration action carefully. \n'
        #     if 'SLOWER' in availableActions:
        #         outputPrefix += 'The deceleration action is LAST priority. \n'
        # outputPrefix += """\nTo check decision safety you should follow steps:
        # Step 1: Get the vehicles in this lane that you may affect. Acceleration, deceleration and idle action affect the Current lane, while left and right lane changes affect the Adjacent lane.
        # Step 2: If there are vehicles, check safety between ego and all vehicles in the action lane ONE by ONE.
        # Step 3: If you find There is no car driving on your "current lane" and you have no conflict with any other vehicle, you can drive faster ! but not too fast to follow the traffic rules.
        # Step 4: If you want to make lane change consider :"Safety Assessment for Lane Changes:" Safe means it is safe to change ,If you want to do IDLE, FASTER, SLOWER, you should consider "Safety Assessment in Current Lane:"
        # """
        return outputPrefix

class isAccelerationConflictWithCar:
    def __init__(self) -> None:
        self.TIME_HEAD_WAY = 5.0
        self.VEHICLE_LENGTH = 5.0
        self.acceleration = 3.0

    @prompts(name='Is Acceleration Conflict With Car',
             description="""useful when you want to know whether acceleration is safe with a specific car, ONLY when your decision is accelerate. The input to this tool should be a string, representing the id of the car you want to check.""")
    def inference(self, ego_info, leading_vehicle_info) -> str:
        if leading_vehicle_info is not None:
            relativeSpeed = ego_info.speed + self.acceleration - leading_vehicle_info.speed
            distance = ego_info.dis2des - leading_vehicle_info.dis2des - self.VEHICLE_LENGTH * 2
            ttc = distance / relativeSpeed
            if ttc > 20:
                return f"acceleration is safe with Veh#{leading_vehicle_info.id}. \n"
            elif 20 >= ttc > 10:
                return f"acceleration may not safe with Veh#{leading_vehicle_info.id}, should be careful if you want to accelerate. \n"
            elif 10 >= ttc > 5:
                return f'acceleration will cause danger, you can not accelerate. \n'
            else:
                return f'acceleration will cause serious danger, must slower your speed. \n'
        else:
            return f"acceleration is safe."

class isKeepSpeedConflictWithCar:
    def __init__(self) -> None:
        self.TIME_HEAD_WAY = 5.0
        self.VEHICLE_LENGTH = 5.0

    @prompts(name='Is Keep Speed Conflict With Car',
             description="""useful when you want to know whether keep speed is safe with a specific car, ONLY when your decision is keep_speed. The input to this tool should be a string, representing the id of the car you want to check.""")
    def inference(self, ego_info, leading_vehicle_info, rearing_vehicle_info) -> str:
        message = ""
        if leading_vehicle_info is not None:
            relativeSpeed = ego_info.speed - leading_vehicle_info.speed
            distance = ego_info.dis2des - leading_vehicle_info.dis2des - self.VEHICLE_LENGTH * 2
            ttc = distance / relativeSpeed
            if ttc > 20:
                message += f"keep lane with current speed is safe with Veh#{leading_vehicle_info.id}. \n"
            elif 20 >= ttc > 10:
                message += f"keep lane with current speed may not safe with Veh#{leading_vehicle_info.id}, should consider decelerate. \n"
            elif 10 >= ttc > 5:
                message += f'keep lane with current speed will cause danger, you should consider decelerate. \n'
            else:
                message += f'keep lane with current speed will cause serious danger, must decelerate. \n'

        if rearing_vehicle_info is not None:
            relativeSpeed = rearing_vehicle_info.speed - ego_info.speed
            distance = rearing_vehicle_info.dis2des - ego_info.dis2des - self.VEHICLE_LENGTH * 2
            ttc = distance / relativeSpeed
            if ttc > 20:
                message += f"keep lane with current speed is safe with Veh#{rearing_vehicle_info.id}. \n"
            elif 20 >= ttc > 10:
                message += f"keep lane with current speed may not safe with Veh#{rearing_vehicle_info.id}, should consider accelerate. \n"
            elif 10 >= ttc > 5:
                message += f'keep lane with current speed will cause danger, you should consider accelerate. \n'
            else:
                message += f'keep lane with current speed will cause serious danger, must accelerate. \n'
        return message

class isDecelerationSafe:
    def __init__(self) -> None:
        self.TIME_HEAD_WAY = 3.0
        self.VEHICLE_LENGTH = 5.0
        self.deceleration = 6.0

    @prompts(name='Is Deceleration Safe',
             description="""useful when you want to know whether deceleration is safe, ONLY when your decision is decelerate.The input to this tool should be a string, representing the id of the car you want to check.""")
    def inference(self, ego_info, rearing_vehicle_info) -> str:
        if rearing_vehicle_info is not None:
            relativeSpeed = rearing_vehicle_info.speed - (ego_info.speed - self.deceleration)
            distance = rearing_vehicle_info.dis2des - ego_info.dis2des - self.VEHICLE_LENGTH * 2
            ttc = distance / relativeSpeed
            if ttc > 20:
                return f"deceleration with current speed is safe with Veh#{rearing_vehicle_info.id}. \n"
            elif 20 >= ttc > 10:
                return f"deceleration with current speed may not safe with Veh#{rearing_vehicle_info.id}. \n"
            elif 10 >= ttc > 5:
                return f'deceleration with current speed will cause danger, if you have no other choice, try not to decelerate so fast as much as possible. \n'
            else:
                return f"deceleration with current speed may be conflict with Veh#{rearing_vehicle_info.id}, you should maintain speed or accelerate. \n"
        else:
            return f"acceleration is safe."

def available_action(toolModels, ego_info):
    # Use tools to analyze the situation
    available_action_tool = next((tool for tool in toolModels if isinstance(tool, getAvailableActions)), None)
    # Use tools to analyze the situation
    available_action = {}
    available_lanes_analysis = available_action_tool.inference(ego_info)
    available_action[available_action_tool] = available_lanes_analysis
    return available_action

def interaction_vehicle(ego_info, other_info):
    ego_direction = 'going straight' if environment.if_going_straight(ego_info.entrance, ego_info.exit) else 'turning'
    other_info_direction = 'going straight' if environment.if_going_straight(other_info.entrance, other_info.exit) else 'turning'
    other_info_action = tools.acc2action(other_info.speed, other_info.acc)
    accelerate_safety_analysis = check_safety_with_conflict_vehicles(ego_info, other_info)
    msg = ''
    msg += f'Your are now {ego_direction}, these are vehicles information you should pay attention and share your intention to them when making decision: \n'
    if not tools.if_passed_conflict_point(ego_info, other_info):
        msg += f'Your surrounding vehicle is now {other_info_direction}, its ACTUAL last action is {other_info_action}, ' \
               f'the position of conflict point between you and him is ({environment.CONFLICT_RELATION_STATE[ego_info.entrance][ego_info.exit][str(other_info.entrance) + str(other_info.exit)]}). ' \
               f'his speed is {round(other_info.speed, 1)}, distance to conflict point is {round(tools.get_dis2cp(other_info, ego_info), 1)}. ' \
               f'Your speed is {ego_info.speed}, distance to conflict point is {round(tools.get_dis2cp(ego_info, other_info), 1)}. ' \
               f'Based on your states and his states, for you, {accelerate_safety_analysis}. \n'
    else:
        msg += f'You has no conflict with Veh#{other_info.id}'
    return msg

def check_safety_in_current_lane(toolModels, ego_info, other_info):
    # lane_cars_id -- {'lane_0': {'leadingCar': None, 'rearingCar': IDMVehicle #224: [173.94198546   0.        ]}}
    # availabel_lane -- {'currentLaneID': 'lane_0', 'leftLane': '', 'rightLane': ''}
    safety_analysis = {
        'acceleration_conflict': None,
        'keep_speed_conflict': None,
        'deceleration_conflict': None
    }

    # Extract tools from toolModels
    acceleration_tool = next((tool for tool in toolModels if isinstance(tool, isAccelerationConflictWithCar)), None)
    keep_speed_tool = next((tool for tool in toolModels if isinstance(tool, isKeepSpeedConflictWithCar)), None)
    deceleration_tool = next((tool for tool in toolModels if isinstance(tool, isDecelerationSafe)), None)
    leading_vehicle_info, rearing_vehicle_info = tools.get_leading_rearing_vehicle_in_same_lane(ego_info, other_info)

    # Check for conflicts if there is a car in the current lane
    if leading_vehicle_info is not None:  # 如果ego前面有车
        safety_analysis['acceleration_conflict'] = acceleration_tool.inference(ego_info, leading_vehicle_info)
    if leading_vehicle_info is not None or rearing_vehicle_info is not None:
        safety_analysis['keep_speed_conflict'] = keep_speed_tool.inference(ego_info, leading_vehicle_info, rearing_vehicle_info)
    if rearing_vehicle_info is not None:  # 如果ego后面有车
        safety_analysis['deceleration_conflict'] = deceleration_tool.inference(ego_info, rearing_vehicle_info)
    return safety_analysis

def check_safety_with_conflict_vehicles(ego_info, other_info):
    dangerous_level = tools.evaluate_safety_with_conflict_vehicles(ego_info, other_info)

    if dangerous_level == 0:
        return 'acceleration is safe, you should FASTER'
    elif dangerous_level == 1:
        return 'acceleration may not safe, should be careful if you want to accelerate'
    elif dangerous_level == 2:
        return 'acceleration will cause danger, you can not accelerate'
    elif dangerous_level == 3:
        return 'acceleration will cause serious danger, consider decelerate.'  # You output decision have to be SLOWER (note that it is in capital letters)!!!
    elif dangerous_level == -1:
        return 'surround vehicle stopped, better accelerate for efficiency'
    else:
        raise ValueError(f'dangerous_level is {dangerous_level}, not in 0/1/2/3 check prompt.py-check_safety_with_conflict_vehicles & tools.py-evaluate_safety_with_conflict_vehicles')

def check_conflict_point_occupied(ego_info, other_info):
    if abs(tools.get_dis2cp(other_info, ego_info)) < 4 and other_info.speed == 0:
        return 'You can not accelerate! A vehicle has stopped on your planning trajectory, you should let him pass first'
    else:
        return ''

def format_decision_info(available_action_msg, interaction_vehicle_msg, current_lane_safety_msg, conflict_lane_safety_msg, conflict_point_occupied):
    formatted_message = ""

    # Add available actions information
    formatted_message += "\nAvailable Actions:\n"
    for tool, action_info in available_action_msg.items():
        formatted_message += f"- {action_info}\n"

    # information of vehicle to interact
    formatted_message += "\nSurrounding Vehicle Information:\n"
    formatted_message += interaction_vehicle_msg

    formatted_message += conflict_point_occupied

    # Safety assessment in the current lane
    # formatted_message += "\nSafety Assessment in Current Lane:\n"
    # for action, safety in current_lane_safety_msg.items():
    #     formatted_message += f"- {action.capitalize().replace('_', ' ')}: {safety}\n"

    # Safety assessment with conflict vehicles
    # formatted_message += "\nSafety Assessment with Vehicle Not in Current Lane but Have Conflict with Ego Vehicle:\n"
    # for action, safety in conflict_lane_safety_msg.items():
    #     formatted_message += f"- {action.capitalize().replace('_', ' ')}: {safety}\n"

    # Most dangerous conflict info which same pattern as memory
    # if most_dangerous_info['delta ttcp'] is not None:
    #     formatted_message += f"\n Currently, the most dangerous collision information with the ego vehicle is as follows: \n"
    #     formatted_message += f"The time to collision is {most_dangerous_info['delta ttcp']}s, ego vehicle's the distance to the collision point is {most_dangerous_info['distance to conflict']} m, " \
    #                          f"and the current speed of the ego vehicle is {most_dangerous_info['speed']}, the distance to the collision point of conflict vehicle is {most_dangerous_info['distance to conflict (others)']} m, its speed is {most_dangerous_info['speed (others)']}\n"
    # else:
    #     formatted_message += f"\n Currently, ego vehicle do not have conflict \n"
    #     formatted_message += f"\n Conflict info is empty \n"
    # print(formatted_message)
    return formatted_message
