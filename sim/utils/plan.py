import numpy as np
import torch
from scipy.spatial.transform import Rotation as SCR
from collections import namedtuple
from sim.utils.agent_controller import constant_headaway
from sim.utils import agent_controller
from collections import defaultdict
from trajdata import AgentType, UnifiedDataset
from trajdata.maps import MapAPI
from trajdata.simulation import SimulationScene
from sim.utils.sim_utils import rt2pose, pose2rt
from sim.utils.agent_controller import IDM, AttackPlanner, ConstantPlanner
import os
import json

Model = namedtuple('Models', ['model_path', 'controller', 'controller_args'])


class planner:
    def __init__(self, plan_list, dt=0.2, unified_map=None, ground=None):
        self.unified_map = unified_map
        self.ground = ground
        self.PREDICT_STEPS = 20
        self.NUM_NEIGHBORS = 3
        
        self.rectify_angle = 0
        if self.unified_map is not None:
            self.rectify_angle = self.unified_map.rectify_angle
        
        # plan_list: a, b, height, yaw, v, model_path, controller, controller_args: dict
        self.stats, self.route, self.controller, self.ckpts, self.wlhs = {}, {}, {}, {}, {}
        self.dt = dt
        self.ATTACK_FREQ = 3
        for iid, args in enumerate(plan_list):
            model = Model(*args[5:])
            self.stats[f'agent_{iid}'] = torch.tensor(args[:5])  # a, b, height, yaw, v
            self.stats[f'agent_{iid}'][3] += self.rectify_angle
            self.route[f'agent_{iid}'] = None
            self.ckpts[f'agent_{iid}'] = os.path.join(model.model_path, 'gs.pth')
            with open(os.path.join(model.model_path, 'wlh.json')) as f:
                self.wlhs[f'agent_{iid}'] = json.load(f)
            self.controller[f'agent_{iid}'] = getattr(agent_controller, model.controller)(**model.controller_args)
            if model.controller == "AttackPlanner":
                self.ATTACK_FREQ = model.controller_args["ATTACK_FREQ"]

    def update_ground(self, ground):
        self.ground = ground

    def update_agent_route(self):
        assert self.unified_map is not None, "Map shouldn't be None to forecast agent path"
        for iid, stat in self.stats.items():
            path = self.unified_map.get_route(stat)
            if path is None:
                print("path not found at ", self.stats)
            if path is not None:
                self.route[iid] = torch.from_numpy(np.hstack([path[:, :2], path[:, -1:]]))

    def ground_height(self, u, v):
        cam_poses, cam_height, _ = self.ground
        cam_poses = torch.from_numpy(cam_poses)
        cam_dist = np.sqrt(
            (cam_poses[:-1, 0, 3] - u) ** 2 + (cam_poses[:-1, 2, 3] - v) ** 2
        )
        nearest_cam_idx = np.argmin(cam_dist, axis=0)
        nearest_c2w = cam_poses[nearest_cam_idx]
        
        nearest_w2c = np.linalg.inv(nearest_c2w)
        uv_local = nearest_w2c[:3, :3] @ np.array([u, 0, v]) + nearest_w2c[:3, 3]
        uv_local[1] = 0
        uv_world = nearest_c2w[:3, :3] @ uv_local + nearest_c2w[:3, 3]
        
        return uv_world[1] + cam_height

    def plan_traj(self, t, ego_stats):
        all_stats = [ego_stats]
        for iid, stat in self.stats.items():
            all_stats.append(stat[[0, 1, 3, 4]])  # a, b, yaw, v
        all_stats = torch.stack(all_stats, dim=0)
        future_states = constant_headaway(all_stats, num_steps=self.PREDICT_STEPS, dt=self.dt)

        b2ws = {}
        for iid, stat in self.stats.items():
            # find closet neighbors
            curr_xy_agents = all_stats[:, :2]
            distance_agents = torch.norm(curr_xy_agents - stat[:2], dim=-1)
            neighbor_idx = torch.argsort(distance_agents)[1:self.NUM_NEIGHBORS + 1]
            neighbors = future_states[neighbor_idx]

            controller = self.controller[iid]
            if type(controller) is IDM:
                next_xyrv = controller.update(state=stat[[0, 1, 3, 4]], path=self.route[iid], dt=self.dt,
                                              neighbors=neighbors)
            elif type(controller) is AttackPlanner:
                safe_neighbors = neighbors[1:, ...]
                next_xyrv = controller.update(state=stat[[0, 1, 3, 4]], unified_map=self.unified_map, dt=0.1,
                                              neighbors=safe_neighbors, attacked_states=future_states[0],
                                              new_plan=((t // self.dt) % self.ATTACK_FREQ == 0))
            elif type(controller) is ConstantPlanner:
                next_xyrv = controller.update(state=stat[[0, 1, 3, 4]], dt=self.dt)
            else:
                raise NotImplementedError
            next_stat = torch.zeros_like(stat)
            next_stat[[0, 1, 3, 4]] = next_xyrv.float()
            next_stat[2] = stat[2]
            self.stats[iid] = next_stat

            h = self.ground_height(next_xyrv[0].numpy(), next_xyrv[1].numpy())
            b2w = np.eye(4)
            b2w[:3, :3] = SCR.from_euler('y', [-stat[3] - np.pi / 2 - self.rectify_angle]).as_matrix()
            b2w[:3, 3] = np.array([next_stat[0], h + stat[2], next_stat[1]])
            b2ws[iid] = torch.tensor(b2w).float().cuda()

        return [b2ws, {}]


class UnifiedMap:
    def __init__(self, datapath, version, scene_name):
        self.datapath = datapath
        self.version = version

        self.dataset = UnifiedDataset(
            desired_data=[self.version],
            data_dirs={
                self.version: self.datapath,
            },
            cache_location="~/.unified_data_cache",
            only_types=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: 50.0),
            desired_dt=0.1,
            num_workers=4,
            verbose=True,
        )

        self.map_api = MapAPI(self.dataset.cache_path)

        self.scene = None
        for scene in list(self.dataset.scenes()):
            if scene.name == scene_name:
                self.scene = scene
        assert self.scene is not None, f"Can't find scene {scene_name}"
        self.vector_map = self.map_api.get_map(
            f"{self.version}:{self.scene.location}"
        )
        self.ego_start_pos, self.ego_start_yaw = self.get_start_pose()
        self.rectify_angle = 0
        if self.ego_start_yaw < 0:
            self.ego_start_yaw += np.pi
            self.rectify_angle = np.pi
        self.PATH_LENGTH = 100

    def get_start_pose(self):
        sim_scene: SimulationScene = SimulationScene(
            env_name=self.version,
            scene_name=f"sim_scene",
            scene=self.scene,
            dataset=self.dataset,
            init_timestep=0,
            freeze_agents=True,
        )
        obs = sim_scene.reset()
        assert obs.agent_name[0] == 'ego', 'The first agent is not ego'
        # We consider position of the first ego frame as origin
        # This suppose is ok when the first frame front camera pose is set as origin
        ego_start_pos = obs.curr_agent_state.position[0]
        ego_start_yaw = obs.curr_agent_state.heading[0]
        return ego_start_pos.numpy(), ego_start_yaw.item()

    def xyzr_local2world(self, stat):
        alpha = np.arctan(stat[0] / stat[1])
        beta = self.ego_start_yaw - alpha
        dist = np.linalg.norm(stat[:2])
        delta_x = dist * np.cos(beta)
        delta_y = dist * np.sin(beta)

        world_stat = np.zeros(4)
        world_stat[0] = delta_x + self.ego_start_pos[0]
        world_stat[1] = delta_y + self.ego_start_pos[1]
        world_stat[3] = stat[3] + self.ego_start_yaw

        return world_stat

    def batch_xyzr_world2local(self, stat):
        beta = np.arctan((stat[:, 1] - self.ego_start_pos[1]) / (stat[:, 0] - self.ego_start_pos[0]))
        alpha = self.ego_start_yaw - beta
        dist = np.linalg.norm(stat[:, :2] - self.ego_start_pos, axis=1)
        delta_x = dist * np.sin(alpha)
        delta_y = dist * np.cos(alpha)

        local_stat = np.zeros_like(stat)
        local_stat[:, 0] = delta_x
        local_stat[:, 1] = delta_y
        local_stat[:, 3] = stat[:, 3] - self.ego_start_yaw

        return local_stat

    def get_route(self, stat):
        # stat: a, b, height, yaw, v
        curr_xyzr = self.xyzr_local2world(stat[:4].numpy())
        
        # lanes = self.vector_map.get_current_lane(curr_xyzr, max_dist=5, max_heading_error=np.pi/3)
        lanes = self.vector_map.get_current_lane(curr_xyzr)

        if len(lanes) > 0:
            curr_lane = lanes[0]
            path = self.batch_xyzr_world2local(curr_lane.center.xyzh)
            total_path_length = np.linalg.norm(curr_lane.center.xy[1:] - curr_lane.center.xy[:-1], axis=1).sum()
            # random select next lanes until reach PATH_LENGTH
            while total_path_length < self.PATH_LENGTH:
                next_lanes = list(curr_lane.next_lanes)
                if len(next_lanes) == 0:
                    break
                next_lane = self.vector_map.get_road_lane(next_lanes[np.random.randint(len(next_lanes))])
                path = np.vstack([path, self.batch_xyzr_world2local(next_lane.center.xyzh)])
                total_path_length += np.linalg.norm(next_lane.center.xy[1:] - next_lane.center.xy[:-1], axis=1).sum()
                curr_lane = next_lane
        else:
            path = None
        return path
