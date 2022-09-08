import numpy as np

from .Ball import Ball
from .KinematicModel import KinematicModel

class SharedGoalsBall(Ball):
    def __init__(self, agent, dT, auto=True, init_state = [-5,-5,0,0]):
        Ball.__init__(self, agent, dT, auto, init_state)

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
        # self.agent_model = self.add_BB8(list(self.get_P()[:,0]), color, scale, render.attachNewNode('agent'))
        self.agent_model = self.add_sphere([self.goal[0], self.goal[1],0], color, scale, render.attachNewNode('agent'))
        # self.goal_model = self.add_sphere([self.goal[0], self.goal[1],0], color[:-1]+[0.5], scale, render.attachNewNode('goal'))
        self.goal_model = None

    
    def redraw_model(self):
        self.agent_model.setPos(self.get_P()[0], self.get_P()[1], 0)
        # self.goal_model.setPos(self.goal[0], self.goal[1], 0)

    def dynamics(self, x, u):
        """
        dynamics function necessary for bayesian inference
        """
        return self.A() @ x + self.B() @ u

