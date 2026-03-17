from isaaclab.utils.math import (
    quat_apply,
    quat_conjugate,
    quat_apply_inverse,
    quat_mul,
    yaw_quat,
    matrix_from_quat
)
import torch

def projected_gravity(root_rot):
    '''
        root_rot: (T, 4)
    '''
    T = root_rot.shape[0]
    gravity = torch.tensor([0.0, 0.0, -1.0], dtype=root_rot.dtype).repeat(T, 1)
    return quat_apply_inverse(root_rot, gravity)
    

def root_lin_vel_r(root_rot, root_lin_vel_w):
    '''
        root_rot: (T, 4)
        root_lin_vel_w: (T, 3)
    '''
    inv_rot = quat_conjugate(root_rot)
    root_lin_vel_r = quat_apply(inv_rot, root_lin_vel_w)
    return root_lin_vel_r

def root_ang_vel_r(root_rot, root_ang_vel_w):
    '''
        root_rot: (T, 4)
        root_ang_vel_w: (T, 3)
    '''
    inv_rot = quat_conjugate(root_rot)
    root_ang_vel_r = quat_apply(inv_rot, root_ang_vel_w)
    return root_ang_vel_r

def body_pos_r(body_pos_w, root_pos, root_rot):
    '''
        body_pos_w: (T, N, 3)
        root_pos: (T, 3)
        root_rot: (T, 4)
    '''
    T, N, _ = body_pos_w.shape
    root_pos_exp = root_pos.unsqueeze(1).repeat(1, N, 1)
    root_rot_exp = root_rot.unsqueeze(1).repeat(1, N, 1)
    rel_body_pos = body_pos_w - root_pos_exp

    flat_root_rot = root_rot_exp.reshape(T * N, 4)
    flat_body_pos = rel_body_pos.reshape(T * N, 3)
    body_pos_r = quat_apply_inverse(flat_root_rot, flat_body_pos)
    return body_pos_r.reshape(T, N, 3)

def body_ori_r(body_quat_w, root_rot):
    """
    Args:
        body_quat_w: (T, N, 4)        
	    root_rot:    (T, 4)
    Returns:
        (T, N, 6)
    """
    T, N, _ = body_quat_w.shape
    
    root_rot_exp = root_rot.unsqueeze(1).repeat(1, N, 1).reshape(T * N, 4)
    flat_body_quat = body_quat_w.reshape(T * N, 4)
    
    inv_root_rot = quat_conjugate(root_rot_exp)
    body_quat_r = quat_mul(inv_root_rot, flat_body_quat)
    mat = matrix_from_quat(body_quat_r)
    return mat[..., :2].reshape(T, N, -1)

def body_lin_vel_r(body_lin_vel_w, root_lin_vel_w, root_rot):
    """
    Args:
        body_lin_vel_w: (T, N, 3) 世界坐标系下的身体线速度
        root_lin_vel_w: (T, 3)    世界坐标系下的基座线速度
        root_rot:       (T, 4)    世界坐标系下的基座四元数
    Returns:
        (T, N, 3) 基座坐标系下的相对线速度
    """
    T, N, _ = body_lin_vel_w.shape
    
    root_lin_vel_exp = root_lin_vel_w.unsqueeze(1).repeat(1, N, 1).reshape(T * N, 3)
    root_rot_exp = root_rot.unsqueeze(1).repeat(1, N, 1).reshape(T * N, 4)
    flat_body_lin_vel = body_lin_vel_w.reshape(T * N, 3)
    
    rel_lin_vel = flat_body_lin_vel - root_lin_vel_exp

    body_lin_vel_r = quat_apply_inverse(root_rot_exp, rel_lin_vel)
    return body_lin_vel_r.reshape(T, N, 3)

def body_ang_vel_r(body_ang_vel_w, root_ang_vel_w, root_rot):
    """
    Args:
        body_ang_vel_w: (T, N, 3) 世界坐标系下的身体角速度
        root_ang_vel_w: (T, 3)    世界坐标系下的基座角速度
        root_rot:       (T, 4)    世界坐标系下的基座四元数
    Returns:
        (T, N, 3) 基座坐标系下的相对角速度
    """
    T, N, _ = body_ang_vel_w.shape
    
    root_ang_vel_exp = root_ang_vel_w.unsqueeze(1).repeat(1, N, 1).reshape(T * N, 3)
    root_rot_exp = root_rot.unsqueeze(1).repeat(1, N, 1).reshape(T * N, 4)
    flat_body_ang_vel = body_ang_vel_w.reshape(T * N, 3)
    
    rel_ang_vel = flat_body_ang_vel - root_ang_vel_exp
    
    body_ang_vel_r = quat_apply_inverse(root_rot_exp, rel_ang_vel)
    return body_ang_vel_r.reshape(T, N, 3)