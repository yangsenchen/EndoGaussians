
import os
import torch
import numpy as np
import open3d as o3d
import time
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, quat_mult
from external import build_rotation
# from colormap import colormap
from copy import deepcopy
import matplotlib.pyplot as plt
from PIL import Image

RENDER_MODE = 'color'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'depth'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'centers'  # 'color', 'depth' or 'centers'

ADDITIONAL_LINES = None  # None, 'trajectories' or 'rotations'
# ADDITIONAL_LINES = 'trajectories'  # None, 'trajectories' or 'rotations'
# ADDITIONAL_LINES = 'rotations'  # None, 'trajectories' or 'rotations'

REMOVE_BACKGROUND = False  # False or True
# REMOVE_BACKGROUND = True  # False or True

FORCE_LOOP = False  # False or True
# FORCE_LOOP = True  # False or True

w, h = 640, 512
near, far = 0.01, 100.0
view_scale = 3.9
fps = 20
traj_frac = 25  # 4% of points
traj_length = 15
def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()


def init_camera(y_angle=0., center_dist=2.4, cam_height=1.3, f_ratio=0.82):
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k


def load_scene_data(seq, exp, seg_as_col=False):
    g = "output/20240118-070719-cut10depreinit/cut10depreinit/params.npz"
    params = dict(np.load(g))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        if REMOVE_BACKGROUND:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg


def make_lineset(all_pts, cols, num_lines):
    linesets = []
    for pts in all_pts:
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


# def calculate_trajectories(scene_data, is_fg):
#     in_pts = [data['means3D'][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
#     num_lines = len(in_pts[0])
#     cols = np.repeat(colormap[np.arange(len(in_pts[0])) % len(colormap)][None], traj_length, 0).reshape(-1, 3)
#     out_pts = []
#     for t in range(len(in_pts))[traj_length:]:
#         out_pts.append(np.array(in_pts[t - traj_length:t + 1]).reshape(-1, 3))
#     return make_lineset(out_pts, cols, num_lines)


# def calculate_rot_vec(scene_data, is_fg):
#     in_pts = [data['means3D'][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
#     in_rotation = [data['rotations'][is_fg][::traj_frac] for data in scene_data]
#     num_lines = len(in_pts[0])
#     cols = colormap[np.arange(num_lines) % len(colormap)]
#     inv_init_q = deepcopy(in_rotation[0])
#     inv_init_q[:, 1:] = -1 * inv_init_q[:, 1:]
#     inv_init_q = inv_init_q / (inv_init_q ** 2).sum(-1)[:, None]
#     init_vec = np.array([-0.1, 0, 0])
#     out_pts = []
#     for t in range(len(in_pts)):
#         cam_rel_qs = quat_mult(in_rotation[t], inv_init_q)
#         rot = build_rotation(cam_rel_qs).cpu().numpy()
#         vec = (rot @ init_vec[None, :, None]).squeeze()
#         out_pts.append(np.concatenate((in_pts[t] + vec, in_pts[t]), 0))
#     return make_lineset(out_pts, cols, num_lines)


def render(w2c, k, timestep_data):
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far)
        im, _, depth, _ = Renderer(raster_settings=cam)(**timestep_data)
        return im, depth


def rgbd2pcd(im, depth, w2c, k, show_depth=False, project_to_cam_w_scale=None):
    d_near = 1.5
    d_far = 6
    invk = torch.inverse(torch.tensor(k).cuda().float())
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    radial_depth = depth[0].reshape(-1)
    def_rays = (invk @ def_pix.T).T
    def_radial_rays = def_rays / torch.linalg.norm(def_rays, ord=2, dim=-1)[:, None]
    pts_cam = def_radial_rays * radial_depth[:, None]
    z_depth = pts_cam[:, 2]
    if project_to_cam_w_scale is not None:
        pts_cam = project_to_cam_w_scale * pts_cam / z_depth[:, None]
    pts4 = torch.concat((pts_cam, pix_ones), 1)
    pts = (c2w @ pts4.T).T[:, :3]
    if show_depth:
        cols = ((z_depth - d_near) / (d_far - d_near))[:, None].repeat(1, 3)
    else:
        cols = torch.permute(im, (1, 2, 0)).reshape(-1, 3)
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols


def visualize(seq, exp):
    # create a folder with the name of exp
    save_path = f"./output/{exp}/images"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    scene_data, is_fg = load_scene_data(seq, exp)

    w2c, k = init_camera()

    poses_arr = np.load("/hpc2hdd/home/ychen950/Dynamic3DGaussians/data/cut/poses_bounds.npy")
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    poses = poses.transpose((2,0,1))
    pose =poses[0]
    c2w = np.vstack([pose[:, :4], np.array([[0,0,0,1]])])
    w2c = np.linalg.inv(c2w)
    h, w, f = int(pose[0, 4]), int(pose[1, 4]), pose[2, 4]
    k = np.array([[f, 0, (w-1)*0.5], [0, f, (h-1)*0.5], [0, 0, 1]])


    # vis = o3d.visualization.Visualizer()
    # vis.create_window(width=int(w * view_scale), height=int(h * view_scale), visible=True)

    for i in range(len(scene_data)):
        im, depth = render(w2c, k, scene_data[i])
        torch.permute(im, (1, 2, 0)).reshape(-1, 3)

        plt.imsave(os.path.join(save_path, f"d{i}.png"), depth.cpu().numpy()[0], cmap='gray')

        img = im.cpu().numpy()

        mask = (img < 0).any(axis=-1)
        img[mask] = [0, 0, 0]
        
        img = img * 255
        img = np.transpose(img.astype(np.uint8), (1, 2, 0))
        img = Image.fromarray(img)
        img.save(os.path.join(save_path, f"im{i}.png"))
        

        pts, cols = rgbd2pcd(im, depth, w2c, k, show_depth=(RENDER_MODE == 'depth'))
        # save as ply file
        pcd = o3d.geometry.PointCloud()
        pcd.points = pts
        pcd.colors = cols
        o3d.io.write_point_cloud(os.path.join(save_path, f"pcd{i}.ply"), pcd)
   


if __name__ == "__main__":
    exp_name = ""
    for sequence in ["cut10","cut"]: # , "boxes", "football", "juggle", "softball", "tennis"
        visualize(sequence, exp_name)


        # ImportError: /usr/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found 
        # (required by /hpc2hdd/home/ychen950/.conda/envs/dgdiff/lib/python3.7/site-packages/diff_gaussian_rasterization/_C.cpython-37m-x86_64-linux-gnu.so)
        # sudo apt-get update && sudo apt-get upgrade