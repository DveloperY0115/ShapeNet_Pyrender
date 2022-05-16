import os
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# NOTE: when rendering on headless server, register the following env variables
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = "1"
sys.path.append(".")
sys.path.append("..")

import argparse
import csv
import time


import trimesh
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
import cv2
import numpy as np
from tqdm import tqdm

from render import *
from utils.math import *

def _render_shapenet_sample(
    shapenet_src_dir: str, sample_id: str, 
    save_dir: str,
    height: int, width: int,
    textureless: bool,
    ) -> None:
    sample_dir = os.path.join(shapenet_src_dir, str(sample_id))
    assert os.path.exists(sample_dir), "[!] Path {} does not exist".format(sample_dir)

    # if not a directory, ignore and quit
    if not os.path.isdir(sample_dir):
        return

    # specify the file name of the mesh
    # if mesh file is missing, ignore and quit
    mesh_file = os.path.join(sample_dir, "models/model_normalized.obj")
    if not os.path.exists(mesh_file):
        return

    # create directories for outputs
    sample_save_dir = os.path.join(save_dir, str(sample_id))
    sample_img_dir = os.path.join(sample_save_dir, "image")
    sample_depth_dir = os.path.join(sample_save_dir, "depth")
    sample_mask_dir = os.path.join(sample_save_dir, "mask")

    if not os.path.exists(sample_save_dir):
        os.mkdir(sample_save_dir)
    if not os.path.exists(sample_img_dir):
        os.mkdir(sample_img_dir)
    if not os.path.exists(sample_depth_dir):
        os.mkdir(sample_depth_dir)
    if not os.path.exists(sample_mask_dir):
        os.mkdir(sample_mask_dir)

    # specify camera intrinsics and extrinsics
    phis = [
        0.0, 
        np.pi / 3, 
        np.pi / 2, 
        2 * np.pi / 3, 
        np.pi, 
        4 * np.pi / 3, 
        3 * np.pi / 2,
        5 * np.pi / 3,
    ]
    thetas = (np.pi / 2.0) * np.ones_like(phis)  # fixed elevation

    # load mesh
    if textureless:
        print("[!] Rendering textureless mesh: {}".format(sample_id))
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color((1.0, 1.0, 1.0))
        box = mesh.get_axis_aligned_bounding_box()
        mesh = mesh.translate(-box.get_center())
        mesh_scale = ((box.get_max_bound() - box.get_min_bound()) ** 2).sum()
        mesh = mesh.scale(1 / mesh_scale, center=(0, 0, 0))
    else:
        mesh = trimesh.load(mesh_file)
        #mesh_scale = (np.array(mesh.extents) ** 2).sum()
        #mesh = mesh.scaled(0.35 * mesh_scale)

    camera_params = {}

    # render and save the results
    for view_idx, (theta, phi) in enumerate(zip(thetas, phis)):
        img, depth, mask, K, E = render_mesh(
            view_idx,
            mesh,
            theta,
            phi,
            height, width,
            flags=pyrender.RenderFlags.NONE,
        )
    
        cv2.imwrite(os.path.join(sample_img_dir, "{}.jpg".format(view_idx)), img)
        cv2.imwrite(os.path.join(sample_depth_dir, "{}.exr".format(view_idx)), depth)
        cv2.imwrite(os.path.join(sample_mask_dir, "{}.jpg".format(view_idx)), mask)

        camera_params["world_mat_{}".format(view_idx)] = E
        camera_params["camera_mat_{}".format(view_idx)] = K
    np.savez(os.path.join(sample_save_dir, "cameras.npz"), **camera_params)

def render_shapenet_samples(
    shapenet_src_dir: str,
    save_dir: str,
    height: int, width: int,
    sample_csv: str = None,
    textureless: bool = False,
    ) -> None:

    # get sample directories
    if sample_csv is None:
        sample_ids = [d for d in os.listdir(shapenet_src_dir)]
    else:
        with open(sample_csv, "r", encoding="utf-8") as f:
            content = csv.reader(f)

            sample_ids = []

            for idx, line in enumerate(content):
                if idx != 0:
                    fullID = line[0]
                    sample_id = fullID.split(".")[-1]
                    sample_ids.append(sample_id)

    # create the save directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    start_t = time.time()

    for sample_id in tqdm(sample_ids):
        _render_shapenet_sample(
            shapenet_src_dir,
            sample_id,
            save_dir,
            height, width,
            textureless,
        )

    print("[!] Took {} seconds".format(time.time() - start_t))
    print("[!] Done.")

if __name__ == "__main__":
    if sys.platform == "darwin":
        print(
            "[!] Pyrender yields slightly different projection matrix on macOS. \
            We highly recommend you to run this script on other OS such as Linux, Windows, etc. \
            For details of problematic behavior of Pyrender, please refer to \
            https://pyrender.readthedocs.io/en/latest/_modules/pyrender/camera.html#IntrinsicsCamera.get_projection_matrix."
        )
        quit(-1)

    # parse arguments
    parser = argparse.ArgumentParser()
    #parser.add_argument("--shapenet_path", type=str, default="data/shapenet_example")
    parser.add_argument("--shapenet_path", type=str, default="../ShapeNetCore.v2/02958343")
    parser.add_argument("--sample_csv", type=str, default="./sedan.csv", help="CSV holding IDs samples to be rendered")
    #parser.add_argument("--sample_csv", type=str, default=None, help="CSV holding IDs samples to be rendered")
    parser.add_argument("--save_path", type=str, default="PaintMe_Debug")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--textureless", type=bool, default=True)
    args = parser.parse_args()

    # render
    render_shapenet_samples(
        args.shapenet_path, 
        args.save_path, 
        args.height, args.width,
        args.sample_csv,
        args.textureless,
    )
