from sqlalchemy.sql.functions import random
import random
from params import *
import tools
if Scenario_name == 'intersection':
    from scenario_environment import intersection_environment as environment
elif Scenario_name == 'merge':
    from scenario_environment import merge_environment as environment
elif Scenario_name == 'roundabout':
    from scenario_environment import roundabout_environment as environment
else:
    raise ValueError('no such environment, check Scenario_name in params')


class Vehicle:
    def __init__(self, entrance, exit, aggressiveness, id):
        self.id = id
        self.entrance = entrance
        self.exit = exit
        self.aggressiveness = aggressiveness
        self.initialize_info()

    def initialize_info(self):
        x, y, speed, heading, dis2des, max_speed = environment.default_exit_and_state(self.entrance, self.exit)
        self.x = x
        self.y = y
        self.heading = heading
        if self.entrance == 'm':
            random_init_dis2des = 0
        else:
            random_init_dis2des = random.randint(0, 30)
        self.dis2des = dis2des - random_init_dis2des
        self.x, self.y = tools.update_pos_from_dis2des_to_Cartesian(self.entrance, self.exit, self.dis2des)
        self.speed = speed
        self.max_speed = max_speed
        self.acc = 0


