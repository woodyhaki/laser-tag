import numpy as np
from multiagent.core import World, Agent, Landmark,generate_obstacles
from multiagent.scenario import BaseScenario
import pdb
from infrastructure.utils import control_effort

nearest_obstacle = 3
nearest_robot = 1

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_p = 3  # x, y, heading
        world.dim_c = 0
        num_agents = 4
        self.episode_length = 100
        self.boudary = [-2, 2, -2, 2]  # x_min, x_max, y_min, y_max
        world.agents = [Agent() for _ in range(num_agents)]
        for a in world.agents:
            a.prev_action = np.array([0.0, 0.0, 0.0])  # vx, vy, omega
            
        # overwrite action space dimension
        for i,agent in enumerate(world.agents):
            if i == 0:
                agent.name = "enemy"
                agent.adversary = False
                agent.size = 0.075
                agent.silent = True
                agent.collide = False
            else:
                agent.name = "ally"
                agent.adversary = True
                agent.size = 0.05
                agent.silent = True
                agent.collide = False
            agent.action_dim = 3  # vx, vy, omega

        self.initial_states = [[0,2.5,1.57],[0,-2.5,1.57]]
        self.follow_stable_steps = 0
        world.hit_obstacles = 0
        world.hit_boundary = 0
        world.hit_others = 0
        
        #---------------------------------
        world.hit_num = 0
        world.current_consecutive_hits = 0
        world.consecutive_5_hits = 0
        world.consecutive_10_hits = 0
        #---------------------------------

        self.reset_world(world)
        
        self.safe_distance = 0.85
        self.reward_weights =  {
                                        'w_target':          80.0,
                                        'w_obstacle':        80.0,
                                        'w_control_eff':     10.0,
                                        'w_inter_robot':     50.0,
                                        'w_visible':         300.0,
                                        'w_occlusion':       50.0,
                                        'w_cross_fire':      80.0,
                                        'w_smooth_action':   50.0
                               }
        
        # self.obstacle_states = np.array(    [       [0, 0, 0.3],
        #                                                  [1, 0, 0.3],
        #                                                      [-1, 0, 0.3],
        #                                                            [-1, 1, 0.3],
        #                                                 #                  [1, -1, 0.3],
        #                                                 #                     [0, -2, 0.3],
        #                                                 #                      [0, 2, 0.3]
        #                                                     ],dtype=np.float32)
        self.obstacle_states = generate_obstacles(n_obstacles=7, xy_range=(-2, 2), r=0.3)
        world.obstacle_states = self.obstacle_states
        world.reward_weights = self.reward_weights
        return world

    def reset_world(self, world):
        i = 0
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-2, +2, 2)
            agent.state.p_ang = np.random.uniform(-np.pi, np.pi)  # heading θ
            # agent.state.p_pos = np.array(self.initial_states[i][0:2],dtype=np.float32)
            # agent.state.p_ang = np.array(self.initial_states[i][2],dtype=np.float32)
            i = i + 1
            
        world.hit_obstacles = 0
        world.hit_boundary = 0
        world.hit_others = 0
        world.hit_num = 0
        world.current_consecutive_hits = 0
        world.consecutive_5_hits = 0
        world.consecutive_10_hits = 0

    def is_vision_blocked(self,p1, p2, robot_radius, center, obstacle_radius):
        d = p2 - p1
        path_length = np.linalg.norm(d)
        if path_length == 0:
            return False, p1
        
        d_normalized = d / path_length
        f = center - p1
        projection = np.dot(f, d_normalized)
        
        if projection < 0:
            closest_point = p1
        elif projection > path_length:
            closest_point = p2
        else:
            closest_point = p1 + projection * d_normalized
        
        distance = np.linalg.norm(center - closest_point)
        total_radius = (robot_radius + obstacle_radius)*0.7
        
        return distance < total_radius

    def compute_visibility_score(self, robot_pos, robot_heading, target_pos,ally=False):
        delta = target_pos - robot_pos
        angle_to_target = np.arctan2(delta[1], delta[0])
        angle_diff = angle_to_target - robot_heading
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  # [-π, π]

        fov_half_angle = np.deg2rad(45)
        fov_score = 1.0 - np.clip(np.abs(angle_diff) / fov_half_angle, 0.0, 1.0)  # [0,1]

        if np.abs(angle_diff) > fov_half_angle:
            fov_score = -np.clip((np.abs(angle_diff) - fov_half_angle) / fov_half_angle, 0.0, 1.0)
        shooting_score = 0.0
        fov_fire_half_angle = np.deg2rad(5)
        if np.abs(angle_diff) < fov_fire_half_angle:
            shooting_score = 1.0 * (1.0 - (np.abs(angle_diff) / fov_fire_half_angle))
            
        occlusion_penalty = 0.0
        block_ob_id = 0

        for (ox, oy, r) in self.obstacle_states:
            res = self.is_vision_blocked(robot_pos, target_pos, 0.15, np.array([ox, oy]), r)
            block_ob_id += 1
            #print(f"{block_ob_id} {[ox, oy]} | {res}")
            
            if res == True: 
                occlusion_penalty += 1.0
                break

        occlusion_score = -2 if occlusion_penalty > 1e-3 else 2

        visibility_score = fov_score + shooting_score
        
        #if ally:
        #  print(f"fov_score {fov_score} occlusion_score {occlusion_score} occlusion_penalty {occlusion_penalty} shooting_score {shooting_score} visibility_score {visibility_score}")
        #print()
        return visibility_score, occlusion_score  # ∈ [-1, 1]

    def compute_cross_fire_cone(self, self_state, other_state):
        heading = self_state[2]

        # Relative vectors
        relative_vectors = other_state[:, :2] - self_state[:2]

        # Relative angles
        angle_to_others = np.arctan2(relative_vectors[:, 1], relative_vectors[:, 0])

        # Relative angle difference (wrapped to [-π, π])
        angle_diff = angle_to_others - heading
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        fov_half_angle = np.deg2rad(5)

        # Penalty score: the closer to the front, the larger the penalty;
        # the more deviated from the cone center, the smaller the penalty;
        # no penalty if outside the cone
        penalty_score = np.clip(1.0 - np.abs(angle_diff) / fov_half_angle, 0.0, 1.0)

        # Or a smoother version:
        # penalty_score = np.exp(- (angle_diff / fov_half_angle)**2)

        # Total penalty (you can adjust the coefficient)
        cone_penalty = -5.0 * np.sum(penalty_score)

        return cone_penalty

    def reward(self, agent, world):
        enemy = world.agents[0]
        #ally = world.agents[1]
        
        # if agent.name == "ally":
        #     dist = np.linalg.norm(agent.state.p_pos - enemy.state.p_pos)
        # else:
        #     dist = np.linalg.norm(agent.state.p_pos - ally.state.p_pos)
            
        if agent.name == "ally":
            dist = np.linalg.norm(agent.state.p_pos - enemy.state.p_pos)
        else:
            dist = 0
            for a in world.agents:
                if a.name == "ally":
                    dist += np.linalg.norm(agent.state.p_pos - a.state.p_pos)
        
        reward = 0
        ##--------------- Reward for hitting the boundary-------------------------------------
        if agent.state.p_pos[0] < self.boudary[0] or agent.state.p_pos[0] > self.boudary[1] or \
           agent.state.p_pos[1] < self.boudary[2] or agent.state.p_pos[1] > self.boudary[3]:
            reward += -1000
            world.hit_boundary += 1
        
        desired_distance = 1.8  # Desired distance to the enemy
        current_robot_position = agent.state.p_pos
        current_robot_heading = agent.state.p_ang
        
        w_target = self.reward_weights['w_target']
        w_obstacle = self.reward_weights['w_obstacle']
        w_control_eff = self.reward_weights['w_control_eff']
        w_inter_robot = self.reward_weights['w_inter_robot']
        w_visible = self.reward_weights['w_visible']
        w_cross_fire = self.reward_weights['w_cross_fire']
        w_smooth_action = self.reward_weights['w_smooth_action']
            
        if agent.name == "ally":
            ##-------------- Reward for following ------------------------------------------------
            residual = abs(dist - desired_distance)
            follow_reward = (-residual)
            # if dist < 0.85:
            #     reward -= 600
            ##-------------- Reward for hitting the enemy ---------------------------------------
            target_position = enemy.state.p_pos
            visibility_score, occusion_score = self.compute_visibility_score(current_robot_position,current_robot_heading,target_position,ally=True)
            
            if visibility_score > 0.9:
                world.hit_num += 1

                # Track consecutive hits
                if not hasattr(world, 'current_consecutive_hits'):
                    world.current_consecutive_hits = 0
                    world.consecutive_5_hits = 0
                    world.consecutive_10_hits = 0
                
                world.current_consecutive_hits += 1

                if world.current_consecutive_hits == 5:
                    world.consecutive_5_hits += 1
                if world.current_consecutive_hits == 10:
                    world.consecutive_10_hits += 1
            else:
                # Reset counter on a miss
                world.current_consecutive_hits = 0
            
            #print(f"visibility_score {visibility_score}")
            
            ##---------------Reward for control effort--------------------------------------------
            max_control = 1.0  + 1.0 + (np.pi / 4) ** 2
            control_eff = control_effort(agent.action.u, omega_weight=2.0)
            control_eff_reward = -max(0, (control_eff - max_control)**2)

            #---------------Compute rewards for avoid obstacles-----------------------------------
            obstacle_dist = current_robot_position -self.obstacle_states[:,:2]
            obstacle_dist = np.linalg.norm(obstacle_dist,axis=1)
            obstacle_reward = 0
            mask = obstacle_dist < self.safe_distance
            obstacle_reward = -np.sum(np.clip(1.0 / (obstacle_dist[mask] + 1e-6), 0, 10))
          #  pdb.set_trace()
            ## Hit the obstacles
            if (abs(obstacle_dist - 0.5 - 0.15) < 0.1).any():
                reward += -300
                world.hit_obstacles += 1

            # ---------------- Reward for action smoothness ---------------------
            angle_change_reward = 0
            if hasattr(agent, 'prev_action'):
                prev_vel = np.array(agent.prev_action[:2])
                curr_vel = np.array(agent.action.u[:2])

                norm_prev = np.linalg.norm(prev_vel)
                norm_curr = np.linalg.norm(curr_vel)

                if norm_prev > 1e-3 and norm_curr > 1e-3:
                    # 计算夹角
                    cos_theta = np.clip(np.dot(prev_vel, curr_vel) / (norm_prev * norm_curr), -1.0, 1.0)
                    angle_diff = np.arccos(cos_theta)
                    angle_change_reward = -angle_diff**2
            agent.prev_action = np.copy(agent.action.u)

            inter_robot_reward = 0
            
            ##----------------- Add cross fire reward--------------------------------------------
            cross_fire_reward = 0
            current_robot_position = agent.state.p_pos.tolist()
            current_robot_heading = agent.state.p_ang
            self_state = [current_robot_position[0],current_robot_position[1],current_robot_heading]
            other_state = []
            for id,a in enumerate(world.agents):
                if id == 0 or a == agent:
                    continue
                p = a.state.p_pos.tolist()
                other_state.append([p[0],p[1],a.state.p_ang])
            cross_fire_reward = self.compute_cross_fire_cone(np.array(self_state),np.array(other_state))
            ##------------------------------------------------------------------------------------
            ##----------------- Add inter robot reward--------------------------------------------
            inter_robots_dist = np.linalg.norm(np.array(self_state[:2]) - np.array(other_state)[:,:2],axis=1)
            mask_inter_robot = inter_robots_dist < self.safe_distance
            inter_robot_reward = -np.sum(np.clip(1.0 / (inter_robots_dist[mask_inter_robot] + 1e-6), 0, 10))
            if (inter_robots_dist < self.safe_distance).any():
                reward -= 1000
                world.hit_others += 1
            # for id,a in enumerate(world.agents):
            #     if id == 0 or a == agent:
            #         continue
            #     p = a.state.p_pos.tolist()
            #     other_state.append([p[0],p[1],a.state.p_ang])
                
                
            #pdb.set_trace()

            self.info =     {
                "weighted_reward_follow":         w_target * follow_reward,
                "weighted_reward_obstacle":       w_obstacle * obstacle_reward,
                "weighted_reward_control_eff":    w_control_eff * control_eff_reward,
                "weighted_reward_inter_robot":    w_inter_robot * inter_robot_reward,
                "weighted_reward_visible":        w_visible * visibility_score,
                "weighted_reward_cross_fire":     w_cross_fire * cross_fire_reward,
                "weighted_reward_angle_change":   w_smooth_action * angle_change_reward,

                "hit_obstacles":                world.hit_obstacles,
                "hit_boundary":                 world.hit_boundary,
                "hit_others":                   world.hit_others,
                
                "consecutive_5_hits":           world.consecutive_5_hits,
                "consecutive_10_hits":          world.consecutive_10_hits,
                "hit_num":                      world.hit_num
                
            }
            ##----------------------------------------------
            reward = reward + \
                        w_target * follow_reward + \
                            w_obstacle * obstacle_reward + \
                                w_control_eff * control_eff_reward + \
                                    w_inter_robot * inter_robot_reward + \
                                        w_visible * (visibility_score +  occusion_score) + \
                                            w_cross_fire * cross_fire_reward + \
                                                w_smooth_action * angle_change_reward
            ##----------------------------------------------
        else:
            reward = reward + dist * w_target * 0.1
            # if dist < 0.85:
            #     reward -= 600
            # current_robot_position = ally.state.p_pos
            # current_robot_heading = ally.state.p_ang
            # target_position = enemy.state.p_pos
            # visibility_score, occusion_score = self.compute_visibility_score(current_robot_position,current_robot_heading,target_position,ally=False)
            visibility_score = 0
            occusion_score = 0
            # ---------------- Reward for action smoothness ---------------------
            angle_change_reward = 0
            if hasattr(agent, 'prev_action'):
                prev_vel = np.array(agent.prev_action[:2])
                curr_vel = np.array(agent.action.u[:2])

                norm_prev = np.linalg.norm(prev_vel)
                norm_curr = np.linalg.norm(curr_vel)

                if norm_prev > 1e-3 and norm_curr > 1e-3:
                    cos_theta = np.clip(np.dot(prev_vel, curr_vel) / (norm_prev * norm_curr), -1.0, 1.0)
                    angle_diff = np.arccos(cos_theta)
                    angle_change_reward = -angle_diff**2
            agent.prev_action = np.copy(agent.action.u)
 
            #---------------Compute rewards for avoid obstacles-----------------------------------
            obstacle_dist = enemy.state.p_pos -self.obstacle_states[:,:2]
            obstacle_dist = np.linalg.norm(obstacle_dist,axis=1)
            obstacle_reward = 0
            mask = obstacle_dist < self.safe_distance
            obstacle_reward = -np.sum(np.clip(1.0 / (obstacle_dist[mask] + 1e-6), 0, 10))
            ## Hit the obstacles
            if (abs(obstacle_dist - 0.5 - 0.15) < 0.1).any():
                #reward -= 300
                world.hit_obstacles += 1

            max_control = 1.0  + 1.0 + (np.pi / 4) ** 2
            control_eff = control_effort(agent.action.u, omega_weight=2.0)
            control_eff_reward = -max(0, (control_eff - max_control)**2)

            reward += (-1) * w_visible * visibility_score * 0.2
            reward += (-1) * occusion_score * 0.2
            reward +=  w_control_eff * control_eff_reward
            reward +=  w_smooth_action * angle_change_reward
            reward +=  w_obstacle * obstacle_reward
            
        return reward

    def observation_0(self, agent, world):
        '''
        This version of observation function contains no information about the other agents.
        '''
        self_pos = agent.state.p_pos
        self_theta = agent.state.p_ang

        other = [a for a in world.agents if a is not agent][0]
        other_pos = other.state.p_pos
        other_theta = other.state.p_ang

        delta_pos = other_pos - self_pos
        delta_theta = other_theta - self_theta

        # Transform obstacles from global to agent's local coordinate frame
        cos_theta = np.cos(-self_theta)
        sin_theta = np.sin(-self_theta)
        
        
        local_relative_x = delta_pos[0] * cos_theta - delta_pos[1] * sin_theta
        local_relative_y = delta_pos[0]  * sin_theta + delta_pos[1] * cos_theta
        local_enemy_relative = [local_relative_x, local_relative_y]
        enemy_norm = np.linalg.norm(local_enemy_relative)
        local_enemy_relative_bearing = local_enemy_relative / enemy_norm

        obstacle_states_local = []
        for obstacle in world.obstacle_states:
            # Extract obstacle properties
            obs_x, obs_y, obs_radius = obstacle
            
            # Calculate relative position
            rel_x = obs_x - self_pos[0]
            rel_y = obs_y - self_pos[1]
            
            # Rotate to agent's local frame
            local_x = rel_x * cos_theta - rel_y * sin_theta
            local_y = rel_x * sin_theta + rel_y * cos_theta
            
            # Store local obstacle state (x, y, radius in local frame)
            obstacle_states_local.extend([local_x, local_y, obs_radius])

        return np.concatenate( [self_pos, [self_theta], [delta_theta], local_enemy_relative_bearing, [enemy_norm], \
                                 obstacle_states_local]    )

    def observation(self, agent, world):
        '''
        TODO: add nearest 
        '''
        self_pos = agent.state.p_pos
        self_theta = agent.state.p_ang

        other = [a for a in world.agents if a is not agent]
        if agent.name == 'ally':
            enemy = [a for a in other if a.name == 'enemy'][0]
            
            enemy_pos = enemy.state.p_pos
            enemy_theta = enemy.state.p_ang

            delta_pos = enemy_pos - self_pos
            delta_theta = enemy_theta - self_theta

            # Transform to agent's local coordinate frame
            cos_theta = np.cos(-self_theta)
            sin_theta = np.sin(-self_theta)
            
            local_relative_x = delta_pos[0] * cos_theta - delta_pos[1] * sin_theta
            local_relative_y = delta_pos[0] * sin_theta + delta_pos[1] * cos_theta
            local_enemy_relative = [local_relative_x, local_relative_y]
            enemy_norm = np.linalg.norm(local_enemy_relative)
            local_enemy_relative_bearing = local_enemy_relative / enemy_norm

            # Convert and sort obstacles by distance in local frame
            local_obstacles = []
            for obstacle in world.obstacle_states:
                obs_x, obs_y, obs_radius = obstacle
                
                rel_x = obs_x - self_pos[0]
                rel_y = obs_y - self_pos[1]
                
                local_x = rel_x * cos_theta - rel_y * sin_theta
                local_y = rel_x * sin_theta + rel_y * cos_theta
                
                dist = np.sqrt(local_x**2 + local_y**2)
                local_obstacles.append((dist, local_x, local_y, obs_radius))

            # Sort by distance (ascending)
            # local_obstacles.sort(key=lambda x: x[0])
            local_obstacles_ob = local_obstacles[:nearest_obstacle]

            # Flatten into observation list
            obstacle_states_local = []
            for _, x, y, r in local_obstacles_ob:
                obstacle_states_local.extend([x, y, r])

            # Convert and sort other robots by distance in local frame
            neighbors_states_local = []
            for other_agent in other:
                if other_agent.name == 'enemy':
                    continue
                ally_pos = other_agent.state.p_pos
                ally_theta = other_agent.state.p_ang
                delta_pos = ally_pos - self_pos
                delta_theta = ally_theta - self_theta
                local_relative_x = delta_pos[0] * cos_theta - delta_pos[1] * sin_theta
                local_relative_y = delta_pos[0]  * sin_theta + delta_pos[1] * cos_theta
                local_neighbor_relative = [local_relative_x, local_relative_y]
                neighbor_norm = np.linalg.norm(local_neighbor_relative)
                local_neighbor_relative_bearing = local_neighbor_relative / neighbor_norm
                neighbors_states_local.append((local_neighbor_relative_bearing[0], local_neighbor_relative_bearing[1], neighbor_norm, delta_theta))

            #neighbors_states_local.sort(key=lambda x: x[2])
            local_robot_ob = neighbors_states_local[:nearest_robot]

            others_states_local = []
            for x, y, r, d_theta in local_robot_ob:
                others_states_local.extend([x, y, r, d_theta])
            #pdb.set_trace()
            return np.concatenate(    [self_pos, [self_theta], [delta_theta], local_enemy_relative_bearing, [enemy_norm], \
                                        obstacle_states_local, others_states_local]    )
        else:
            ally_pos = other[0].state.p_pos
            ally_theta = other[0].state.p_ang
            delta_theta_final = 0
            local_enemy_relative_bearing = [0, 0]
            enemy_norm = 0
            
            # Transform to agent's local coordinate frame
            cos_theta = np.cos(-self_theta)
            sin_theta = np.sin(-self_theta)
            # Convert and sort obstacles by distance in local frame
            local_obstacles = []
            for obstacle in world.obstacle_states:
                obs_x, obs_y, obs_radius = obstacle
                
                rel_x = obs_x - self_pos[0]
                rel_y = obs_y - self_pos[1]
                
                local_x = rel_x * cos_theta - rel_y * sin_theta
                local_y = rel_x * sin_theta + rel_y * cos_theta
                
                dist = np.sqrt(local_x**2 + local_y**2)
                local_obstacles.append((dist, local_x, local_y, obs_radius))

            # Sort by distance (ascending)
            # local_obstacles.sort(key=lambda x: x[0])
            local_obstacles_ob = local_obstacles[:nearest_obstacle]

            obstacle_states_local = []
            for _, x, y, r in local_obstacles_ob:
                obstacle_states_local.extend([x, y, r])

            neighbors_states_local = []
            for other_agent in other:
                ally_pos = other_agent.state.p_pos
                ally_theta = other_agent.state.p_ang
                delta_pos = ally_pos - self_pos
                delta_theta = ally_theta - self_theta
                local_relative_x = delta_pos[0] * cos_theta - delta_pos[1] * sin_theta
                local_relative_y = delta_pos[0]  * sin_theta + delta_pos[1] * cos_theta
                local_neighbor_relative = [local_relative_x, local_relative_y]
                neighbor_norm = np.linalg.norm(local_neighbor_relative)
                local_neighbor_relative_bearing = local_neighbor_relative / neighbor_norm
                neighbors_states_local.append((local_neighbor_relative_bearing[0], local_neighbor_relative_bearing[1], neighbor_norm, delta_theta))

            #neighbors_states_local.sort(key=lambda x: x[2])
            local_robot_ob = neighbors_states_local[:nearest_robot]

            others_states_local = []
            for x, y, r,d_theta in local_robot_ob:
                others_states_local.extend([x, y, r, d_theta])

            return np.concatenate(    [self_pos, [self_theta], [delta_theta_final], local_enemy_relative_bearing, [enemy_norm], \
                                        obstacle_states_local, others_states_local]    )

    def done(self, agent, world):
        # if agent.state.p_pos[0] < self.boudary[0] or agent.state.p_pos[0] > self.boudary[1] or \
        #    agent.state.p_pos[1] < self.boudary[2] or agent.state.p_pos[1] > self.boudary[3]:
        #     return True
        
        # obstacle_dist = agent.state.p_pos -self.obstacle_states[:,:2]
        # obstacle_dist = np.linalg.norm(obstacle_dist,axis=1)
        # #mask = obstacle_dist < self.safe_distance
        # ## Hit the obstacles
        # if (abs(obstacle_dist - 0.5 - 0.15) < 0.2).any():
        #     return True
        return False
    
    
    def info(self,agent,world):
        return self.info

