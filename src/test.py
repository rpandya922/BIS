from agents import *
from models import *
from utils.World import *

# instantiate the class
# robot = SCARA(SublevelSafeSet(), dT)
# robot = RobotArm(SublevelSafeSet(), dT);
# robot = Unicycle(SafeSet(), dT);
# robot = Ball(SafeSet(), dT);
# robot = SharedGoalsBall(SafeSet(d_min=3), dT)
# human = InteractiveHumanBall2D(SafeSet(d_min=1, k_v=2), dT);
# human = HumanBall2D(MobileAgent, dT)
dT = 0.05
robot = SharedGoalsSCARA(SublevelSafeSet(), dT, use_intent_pred=True)
human = BayesianHumanBall(MobileAgent, dT)
robot.set_partner_agent(human)
human.set_partner_agent(robot)
w = World(dT, human, robot)
base.run()
