import numpy as np
import torch
softmax = torch.nn.Softmax(dim=1)
from collections import deque

from .SCARA import SCARA
from .KinematicModel import KinematicModel
from predictor.intention_predictor import create_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SharedGoalsSCARA(SCARA):
    def __init__(self, agent, dT, auto = True, init_state = [-4,-4, 0, 0, 4.5, 4.5], use_intent_pred = False):
        SCARA.__init__(self, agent, dT, auto, init_state)

        self.use_intent_pred = use_intent_pred
        if self.use_intent_pred:
            horizon_len = 20
            hist_len = 5
            goal_mode = "dynamic"
            self.intent_predictor = create_model(horizon_len=horizon_len, goal_mode=goal_mode)
            self.intent_predictor.load_state_dict(torch.load("../data/models/bis_intention_predictor.pt", map_location=device))
            self.horizon_len = horizon_len
            self.hist_len = hist_len
            self.goal_mode = goal_mode
            self.intention_data = {"xh_hist": deque(), "xr_hist": deque(), "goals_hist": deque()}
        
    def set_partner_agent(self, agent):
        self.partner = agent
        # set the possible goals to be the same as partner's set of goals
        self.possible_goals = self.partner.possible_goals
        n_goals = self.possible_goals.shape[1]
        self.goal_probs = np.ones((n_goals,)) / n_goals
        # set the goal to be the closest goal to self
        self.goal_idx = np.argmin(np.linalg.norm(self.possible_goals[[0,1,2]] - self.get_P(), axis=0))
        self.goal = self.possible_goals[:,[self.goal_idx]]

    def set_goals_from_partner(self, goals):
        self.possible_goals = goals
        # compute closest goal
        self.goal_idx = np.argmin(np.linalg.norm(self.possible_goals[[0,1,2]] - self.get_P(), axis=0))
        self.goal = self.possible_goals[:,[self.goal_idx]]

    def u_ref(self, x=None):
        if x is None:
            x = self.get_PV()
        inv_J = self.inv_J()
        dp = self.observe((self.goal - x)[[0,1]]);
        dis = np.linalg.norm(dp);
        v = self.observe(self.get_V())[[0,1]];

        if dis > 2:
            u0 = 0.2 * inv_J * dp - self.k_v * self.observe(self.x[[2,3],0]);
        else:
            u0 = 0.1 * inv_J * dp - self.k_v * self.observe(self.x[[2,3],0]);

        return u0

    def get_nominal_plan(self, horizon=5, return_controls=False, xr0=None, goal=None):
        # ignore safe control for plan
        if xr0 is None:
            robot_x = self.x
        else:
            robot_x = xr0
        
        if goal is None:
            goal = self.goal
        
        robot_states = np.zeros((4, horizon))
        robot_controls = np.zeros((2, horizon))
        for i in range(horizon):
            xx = np.vstack((robot_x, np.zeros((2,1))))
            goal_u = self.u_ref(xx)
            robot_x = self.dynamics(robot_x, goal_u)
            robot_states[:,[i]] = robot_x
            robot_controls[:,[i]] = goal_u
        if return_controls:
            return robot_states, robot_controls
        return robot_states

    def get_intent_pred(self, obstacle):
        if not self.use_intent_pred:
            return

        # update the intention data
        self.intention_data["xh_hist"].append(obstacle.x_est)
        if len(self.intention_data["xh_hist"]) > self.hist_len:
            self.intention_data["xh_hist"].popleft()
        self.intention_data["xr_hist"].append(self.x)
        if len(self.intention_data["xr_hist"]) > self.hist_len:
            self.intention_data["xr_hist"].popleft()
        self.intention_data["goals_hist"].append(self.possible_goals[0:4,:])
        if len(self.intention_data["goals_hist"]) > self.hist_len:
            self.intention_data["goals_hist"].popleft()
        
        # if the intention data is not long enough, return
        if len(self.intention_data["xh_hist"]) < self.hist_len:
            return
        
        # get the intention prediction
        xh_hist = np.hstack(self.intention_data["xh_hist"])
        xr_hist = np.hstack(self.intention_data["xr_hist"])
        goals_hist = np.dstack(self.intention_data["goals_hist"])
        goals_hist = goals_hist.reshape((goals_hist.shape[0]*goals_hist.shape[1], goals_hist.shape[2]))

        # compute the robot's nominal plan towards its current goal
        xr_future = self.get_nominal_plan(horizon=self.horizon_len, return_controls=False, xr0=self.x, goal=self.goal)
        
        # convert to torch tensors to input to NN model
        traj_hist = torch.tensor(np.vstack((xh_hist, xr_hist)).T).float().to(device).unsqueeze(0)
        goals_hist = torch.tensor(goals_hist.T).float().to(device).unsqueeze(0)
        xr_future = torch.tensor(xr_future.T).float().to(device).unsqueeze(0)
        
        # query intention prediction model
        goal_probs = softmax(self.intent_predictor(traj_hist, xr_future, goals_hist)).detach().numpy()[0]
        self.goal_probs = goal_probs

    def update(self, obstacle):
        """Update phase. 1. update score, 2. predict intention of human 3. update goal, 4. update self state estimation, 5. update the nearest point on self to obstacle, 6. calculate control input, 7. update historical trajectory.

        Args:
            obstacle (KinematicModel()): the obstacle
        """
        self.time = self.time + 1
        self.update_score(obstacle)
        self.get_intent_pred(obstacle)
        self.update_goal()
        self.kalman_estimate_state()
        self.update_m(obstacle.m)
        self.calc_control(obstacle)
        self.update_trace()

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