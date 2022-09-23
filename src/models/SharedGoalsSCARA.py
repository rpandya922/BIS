import numpy as np

from .SCARA import SCARA
from .KinematicModel import KinematicModel

class SharedGoalsSCARA(SCARA):
    def __init__(self, agent, dT, auto = True, init_state = [-4,-4, 0, 0, 4.5, 4.5]):
        SCARA.__init__(self, agent, dT, auto, init_state)
        
    def set_partner_agent(self, agent):
        self.partner = agent
        # set the possible goals to be the same as partner's set of goals
        self.possible_goals = self.partner.possible_goals
        # set the goal to be the closest goal to self
        self.goal_idx = np.argmin(np.linalg.norm(self.possible_goals[[0,1,2]] - self.get_P(), axis=0))
        self.goal = self.possible_goals[:,[self.goal_idx]]

    def set_goals_from_partner(self, goals):
        self.possible_goals = goals
        # compute closest goal
        self.goal_idx = np.argmin(np.linalg.norm(self.possible_goals[[0,1,2]] - self.get_P(), axis=0))
        self.goal = self.possible_goals[:,[self.goal_idx]]

    def update_goal(self):
        dx = self.get_P() - self.goal[[0,1,2]]
        dv = self.get_V() - self.goal[[3,4,5]]

        if np.linalg.norm(dx) < 0.3 and np.linalg.norm(dv) < 0.5:
            self.goal_achieved = self.goal_achieved + 1
            total_goals_achieved = self.partner.goal_achieved + self.goal_achieved
            self.possible_goals[:,self.goal_idx] = self.goals[:,total_goals_achieved+3]

            # compute closest goal
            self.goal_idx = np.argmin(np.linalg.norm(self.possible_goals[[0,1,2]] - self.get_P(), axis=0))
            self.goal = self.possible_goals[:,[self.goal_idx]]

            # set partner's possible goals to be the same set 
            self.partner.set_goals_from_partner(self.possible_goals)
    
    def load_model(self, render, loader, color=[0.1, 0.5, 0.8, 1], scale=0.5):
        KinematicModel.load_model(self, render, loader, color, scale)

        pos = [-4, -4, 0]
        theta1 = 0;
        theta2 = 0;
        l1 = 4.5;
        l2 = 4.5;

        self.agent_model = render.attachNewNode('agent')

        ret1 = loader.loadModel("resource/cube")
        ret1.reparentTo(self.agent_model)
        ret1.setColor(color[0], color[1], color[2], color[3]);
        ret1.setScale(l1/2, 0.1, 0.1);
        ret1.setPos(pos[0]+l1/2, pos[1], pos[2]);

        pivot1 = self.agent_model.attachNewNode("arm1-pivot")
        pivot1.setPos(pos[0], pos[1], pos[2]) # Set location of pivot point
        ret1.wrtReparentTo(pivot1) # Preserve absolute position
        pivot1.setH(theta1) # Rotates environ around pivot

        ret2 = loader.loadModel("resource/cube")
        ret2.reparentTo(self.agent_model)
        ret2.setColor(color[0], color[1], color[2], color[3]);
        ret2.setScale(l2/2, 0.1, 0.1);
        ret2.setPos(pos[0]+l2/2+l1, pos[1], pos[2]);

        pivot2 = pivot1.attachNewNode("arm2-pivot")
        pivot2.setPos(l1, 0, 0) # Set location of pivot point
        ret2.wrtReparentTo(pivot2) # Preserve absolute position
        pivot2.setH(theta2) # Rotates environ around pivot

        self.robot_arm1 = pivot1
        self.robot_arm2 = pivot2

        self.goal_model = None

    def redraw_model(self):
        self.robot_arm1.setH(self.x[0,0] / np.pi * 180);
        self.robot_arm2.setH(self.x[1,0] / np.pi * 180);

    def get_ee_state(self, x):
        l1 = self.l[0]
        l2 = self.l[1]
        theta1 = x[0]
        theta2 = x[1]
        px = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2) + self.base[0]
        py = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) + self.base[1]
        p = np.vstack([px, py])

        vx = (-l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)) * x[2]  - l2 * np.sin(theta1 + theta2) * x[3];
        vy = l1 * np.cos(theta1) * x[2] + l2 * np.cos(theta1 + theta2) * x[3];
        v = np.vstack([vx, vy])

        return np.vstack([p, v])

    def dynamics(self, x, u):
        """
        dynamics function necessary for bayesian inference
        """
        next_x = self.A() @ x + self.B() @ u
        return self.get_ee_state(next_x)