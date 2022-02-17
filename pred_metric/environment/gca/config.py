from gym_collision_avoidance.envs.config import Example as EnvConfig

class CollisionConfig(EnvConfig):
    def __init__(self):
        self.NEAR_GOAL_THRESHOLD = 0.1
        self.DT = 0.1
        self.TRAIN_SINGLE_AGENT = True
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 2

        EnvConfig.__init__(self)

        self.NEAR_GOAL_THRESHOLD = 0.1
        self.DT = 0.1
        self.TRAIN_SINGLE_AGENT = True
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 2
