import numpy as np
import torch


def transform_lidar_to_ogm(ego_fused_lidar, map_size=64, resolution=0.1):
    """Rasterise LiDAR scans into occupancy grid maps centred on the robot.

    ego_fused_lidar: (batch, t, num_ray, num_sources) ranges in metres; non-positive
        ranges are ignored. All sources are drawn into the same map.
    Returns ogm: (batch, t, map_size, map_size) with 1 for occupied cells.
    """
    batch_size, t, num_ray, num_agents = ego_fused_lidar.shape

    angles = np.linspace(0, 2 * np.pi, num_ray)

    ogm = np.full((batch_size, t, map_size, map_size), -1, dtype=np.float32)

    center_index = map_size // 2
    
    for b in range(batch_size):
        for a in range(t):
            for i in range(num_agents):
                rob_lidar = ego_fused_lidar[b, a, :, i]

                distance_x = rob_lidar * np.cos(angles)
                distance_y = rob_lidar * np.sin(angles)

                valid_mask = rob_lidar > 0

                x_indices = (distance_x[valid_mask] / resolution).astype(int) + center_index
                y_indices = (distance_y[valid_mask] / resolution).astype(int) + center_index

                valid_x = (x_indices >= 0) & (x_indices < map_size)
                valid_y = (y_indices >= 0) & (y_indices < map_size)

                valid_indices = valid_x & valid_y

                ogm[b, a, x_indices[valid_indices], y_indices[valid_indices]] = 1

    ogm[ogm == -1] = 0
    
    return ogm


def transform_point_clouds(other_lidar, translation_2d, is_current=False):
    """Re-express LiDAR scans in a frame translated by `translation_2d`.

    other_lidar: (batch, t, num_ray) ranges in metres.
    translation_2d: (batch, t, 2) offset of the target frame per time step
        (the last one is used for all steps when is_current is set).
    Returns the scans re-binned by angle in the target frame, (batch, t, num_ray);
    points further than 7 m are dropped.
    """
    n_rollout_threads, t, n_angles = other_lidar.shape

    transformed_lidar = np.zeros((n_rollout_threads, t, n_angles))

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    for thread_idx in range(n_rollout_threads):
        for time_idx in range(t):
            if is_current:
                tx, ty = translation_2d[thread_idx, t-1]
            else:
                tx, ty = translation_2d[thread_idx, time_idx]

            robot_points_x = other_lidar[thread_idx, time_idx] * np.cos(angles)
            robot_points_y = other_lidar[thread_idx, time_idx] * np.sin(angles)

            valid_points_mask = (other_lidar[thread_idx, time_idx] > 0)
            valid_robot_points_x = robot_points_x[valid_points_mask]
            valid_robot_points_y = robot_points_y[valid_points_mask]
            valid_angles = angles[valid_points_mask]

            ego_points_x = valid_robot_points_x - tx
            ego_points_y = valid_robot_points_y - ty

            ego_points_r = np.sqrt(ego_points_x**2 + ego_points_y**2)
            ego_points_angles = np.arctan2(ego_points_y, ego_points_x)

            valid_ego_points_mask = (0 < ego_points_r) & (ego_points_r <= 7)
            ego_points_r = ego_points_r[valid_ego_points_mask]
            ego_points_angles = ego_points_angles[valid_ego_points_mask]

            ego_points_angles_normalized = np.mod(ego_points_angles, 2 * np.pi)
            ego_points_indices = np.floor(ego_points_angles_normalized / (2 * np.pi / n_angles)).astype(int)
            ego_points_indices = np.mod(ego_points_indices, n_angles)
            
            for i, index in enumerate(ego_points_indices):
                transformed_lidar[thread_idx][time_idx][index] = ego_points_r[i]

    return transformed_lidar


def shannon_entropy(ogm):
    """Pixel-wise Bernoulli entropy (nats) averaged over the forecast samples.

    ogm: (samples, batch, steps, H, W) occupancy probabilities.
    Returns (batch, steps, H, W).
    """
    eps = 1e-8

    entropy_map = - (ogm * torch.log(ogm + eps) + (1 - ogm) * torch.log(1 - ogm + eps))

    entropy_map = entropy_map.mean(dim=0)
    
    return entropy_map
