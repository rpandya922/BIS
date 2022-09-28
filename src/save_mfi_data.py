import numpy as np

from agents import SublevelSafeSet, MobileAgent
from models import SharedGoalsSCARA, BayesianHumanBall
from utils.Record import Record

def simulate_interaction(horizon=200):
    fps = 20
    dT = 1 / fps

    # goals are randomly initialized inside objects
    robot = SharedGoalsSCARA(SublevelSafeSet(), dT)
    human = BayesianHumanBall(MobileAgent, dT)
    robot.set_partner_agent(human)
    human.set_partner_agent(robot)

    xh_traj = np.zeros((human.n, horizon))
    xr_traj = np.zeros((robot.n, horizon))
    possible_goals = np.zeros((*human.possible_goals.shape, horizon))
    h_goals = np.zeros((human.goal.shape[0], horizon))
    r_goals = np.zeros((robot.goal.shape[0], horizon))
    h_goal_reached = np.zeros(horizon)
    h_goal_idx = np.zeros(horizon)

    for i in range(horizon):
        # save data
        xh_traj[:,[i]] = human.x
        xr_traj[:,[i]] = robot.x
        possible_goals[:,:,i] = human.possible_goals
        h_goals[:,[i]] = human.goal
        r_goals[:,[i]] = robot.goal
        h_goal_reached[i] = int(human.is_goal_reached())
        h_goal_idx[i] = np.argmin(np.linalg.norm(human.possible_goals - human.goal, axis=0))

        # move both agents
        human.update(robot)
        human.move()
        robot.update(human)
        robot.move()

    return xh_traj, xr_traj, possible_goals, h_goal_reached, h_goal_idx

def propogate_goal_reached(h_goal_reached, h_goal_idx):
    goal_idxs = np.zeros_like(h_goal_idx)
    # propogate backwards the index of the reached goal to the previous time steps
    goals_reached = np.where(h_goal_reached)[0]
    if len(goals_reached) > 0:
        curr_goal = h_goal_idx[goals_reached[-1]]

        # cheat a bit and use the actual human goal for any remaining time steps
        goal_idxs[goals_reached[-1]:] = h_goal_idx[goals_reached[-1]:]

    for i in range(goals_reached[-1], -1, -1):
        if h_goal_reached[i] == 1:
            curr_goal = h_goal_idx[i]
        goal_idxs[i] = curr_goal
    return goal_idxs

def create_dataset(n_trajectories=1):
    horizon = 200

    all_xh_traj = []
    all_xr_traj = []
    all_goals = []
    all_h_goal_reached = []

    # labels
    goal_reached = []
    goal_idx = []

    for i in range(n_trajectories):
        xh_traj, xr_traj, goals, h_goal_reached, h_goal_idx = simulate_interaction(horizon=horizon)
        h_goal_idx = propogate_goal_reached(h_goal_reached, h_goal_idx)

        # if this is the first iteration, initialize the arrays
        if i == 0:
            all_xh_traj = np.zeros((*xh_traj.shape, n_trajectories))
            all_xr_traj = np.zeros((*xr_traj.shape, n_trajectories))
            all_goals = np.zeros((*goals.shape, n_trajectories))
            all_h_goal_reached = np.zeros((*h_goal_reached.shape, n_trajectories))
            goal_reached = np.zeros((1, n_trajectories))
            goal_idx = np.zeros((*h_goal_idx.shape, n_trajectories))

        all_xh_traj[:,:,i] = xh_traj
        all_xr_traj[:,:,i] = xr_traj
        all_goals[:,:,:,i] = goals
        all_h_goal_reached[:,i] = h_goal_reached
        goal_reached[:,i] = h_goal_reached[-1]
        goal_idx[:,i] = h_goal_idx

    return all_xh_traj, all_xr_traj, all_goals, all_h_goal_reached, goal_reached, goal_idx

def save_data(path="../data/simulated_interactions.npz", n_trajectories=10):

    all_xh_traj, all_xr_traj, all_goals, all_h_goal_reached, goal_reached, goal_idx = create_dataset(n_trajectories=n_trajectories)

    np.savez(path, all_xh_traj=all_xh_traj, all_xr_traj=all_xr_traj, all_goals=all_goals, all_h_goal_reached=all_h_goal_reached, goal_reached=goal_reached, goal_idx=goal_idx)

if __name__ == "__main__":
    save_data()
