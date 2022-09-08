from .HumanBall2D import HumanBall2D
import numpy as np
from scipy.special import softmax

class BayesEstimator():
    def __init__(self, thetas, dynamics, prior=None, beta=0.7):

        self.thetas = thetas
        self.dynamics = dynamics
        self.beta = beta

        n_theta = thetas.shape[1]
        if prior is None:
            prior = np.ones(n_theta) / n_theta
        else:
            assert len(prior) == n_theta
        self.belief = prior
        
        self.actions = np.mgrid[-4:4:21j, -4:4:21j].reshape(2,-1).T

    def project_action(self, action):
        # passed-in action will be a column vector
        a = action.flatten()
        # find closest action
        dists = np.linalg.norm(self.actions - a, axis=1)
        a_idx = np.argmin(dists)

        return self.actions[a_idx], a_idx

    def update_belief(self, state, action):
        # project chosen action to discrete set
        _, a_idx = self.project_action(action)

        # consider the next state if each potential action was chosen
        next_states = np.array([self.dynamics.step(state, a[:,None]) for a in self.actions]) # dynamics.step expects column vectors
        rs = np.array([-np.linalg.norm(state - s) for s in next_states])[:,None]

        # assume optimal trajectory is defined by straight line towards goal, so reward is negative distance from goal
        opt_rewards = np.linalg.norm((next_states - self.thetas[None,:,:]), axis=1)

        Q_vals = rs - opt_rewards

        # compute probability of choosing each action
        prob_action = softmax(self.beta * Q_vals, axis=0)
        # get row corresponding to chosen action
        y_i = prob_action[a_idx]

        # update belief
        new_belief = (y_i * self.belief) / np.sum(y_i * self.belief)
        self.belief = new_belief

        return new_belief

    def copy(self):
        return BayesEstimator(self.thetas.copy(), self.dynamics, self.belief.copy(), self.beta)

# TODO: make this inherit from HumanBall2D
class BayesianHumanBall(HumanBall2D):

    """
    This 2D human ball estimates the robot's goal and changes its own goal accordingly.
    """

    def __init__(self, agent, dT, auto = True, init_state=[5,5,0,0]):
        HumanBall2D.__init__(self, agent, dT, auto, init_state)

        # setting multiple possible goals for the human
        self.possible_goals = self.goals[:,0:3]
        self.goal_idx = np.argmin(np.linalg.norm(self.possible_goals[[0,1,2]] - self.get_P(), axis=0))
        self.goal = self.possible_goals[:,[self.goal_idx]]

        self.belief = BayesEstimator(self.possible_goals, None)

    def set_partner_agent(self, agent):
        self.partner = agent

    def set_goals_from_partner(self, goals):
        self.possible_goals = goals
        # compute closest goal
        self.goal_idx = np.argmin(np.linalg.norm(self.possible_goals[[0,1,2]] - self.get_P(), axis=0))
        self.goal = self.possible_goals[:,[self.goal_idx]]
        # TODO: reset belief here

    def update_goal(self):
        dx = self.get_P() - self.goal[[0,1,2]]
        dv = self.get_V() - self.goal[[3,4,5]]

        if np.linalg.norm(dx) < 0.3 and np.linalg.norm(dv) < 0.5:
            self.goal_achieved = self.goal_achieved + 1
            # self.goal = np.vstack(self.goals[:,self.goal_achieved])
            self.possible_goals[:,self.goal_idx] = self.goals[:,self.goal_achieved+3]

            # compute closest goal
            self.goal_idx = np.argmin(np.linalg.norm(self.possible_goals[[0,1,2]] - self.get_P(), axis=0))
            self.goal = self.possible_goals[:,[self.goal_idx]]

            # set partner's possible goals to be the same set 
            self.partner.set_goals_from_partner(self.possible_goals)
        
    def load_model(self, render, loader, color=[0.8, 0.3, 0.2, 1], scale=0.5):
        self.render = render
        self.agent_model = self.add_sphere(list(self.get_P()[:,0]), [0.8, 0.3, 0.2, 1], scale, render.attachNewNode('agent'));
        goal_models = []
        for i in range(self.possible_goals.shape[1]):
            goal_models.append(self.add_sphere([self.possible_goals[:,i][0], self.possible_goals[:,i][1], 0], [0.8, 0.3, 0.2, 0.5], scale, render.attachNewNode(f'goal_{i}')))
        self.goal_models = goal_models
        self.goal_model = None
        # self.goal_model = goal_models[self.goal_idx]

    def redraw_model(self):
        self.agent_model.setPos(self.get_P()[0], self.get_P()[1], 0)
        # self.goal_model.setPos(self.goal[0], self.goal[1], 0)
        for i, goal_model in enumerate(self.goal_models):
            goal_model.setPos(self.possible_goals[:,i][0], self.possible_goals[:,i][1], 0)
        
