import argparse
import csv
import os
import sys
import time
from typing import Optional

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# NOTE: when rendering on headless server, register the following env variables
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_DEVICE_ID"] = "1"


import cv2
import numpy as np
import open3d as o3d

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
from tqdm import tqdm
import trimesh

# import modules defined in the project
sys.path.append(".")
sys.path.append("..")
from render import *
from utils.math import *


def _render_shapenet_sample(
    shapenet_src_dir: str,
    sample_id: str,
    save_dir: str,
    height: int,
    width: int,
    textureless: bool,
) -> None:
    """
    Renders a ShapeNet model and save the rendered outputs.

    Args:
        shapenet_src_dir (str): The directory containing 3D shape's geometry, material, and texture files.
        sample_id (str): The ShapeNet ID of the shape being rendered.
        save_dir (str): The directory where rendering outputs are saved.
        height (int): The height of output images.
        width (int): The width of output images.
        textureless (bool): The flag for toggling textureless rendering mode. If True, the texture images
            of a 3D shapes are ignored and the mesh is uniformly colored with white and rendered.
    """
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
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color((1.0, 1.0, 1.0))

        # scale the mesh
        box = mesh.get_axis_aligned_bounding_box()
        mesh = mesh.translate(-box.get_center())
        mesh_scale = ((box.get_max_bound() - box.get_min_bound()) ** 2).sum()
        mesh = mesh.scale(1 / mesh_scale, center=(0, 0, 0))
    else:
        raise ValueError("Currently unsupported.")
        # mesh = trimesh.load(mesh_file)

    camera_params = {}

    # render and save the results
    for view_idx, (theta, phi) in enumerate(zip(thetas, phis)):
        # TODO: Replace the hard-coded numbers
        # TODO: How to adjust the size of objects to fit the entire image?
        fx = 23.027512 / 0.0369161
        fy = 23.027512 / 0.0369161

        if view_idx in (0, 4):
            fx = 1.0 * fx
            fy = 1.5 * fy
            theta = np.pi / 2.0
        elif view_idx in (2, 6):
            fx = 2.2 * fx
            fy = 2.5 * fy
        elif view_idx in (1, 3, 5, 7):
            fx = 1.3 * fx
            fy = 1.6 * fy

        img, depth, mask, K, E = render_mesh(
            mesh,
            theta,
            phi,
            fx,
            fy,
            height,
            width,
            flags=pyrender.RenderFlags.NONE,
        )

        # save images
        cv2.imwrite(os.path.join(sample_img_dir, f"{view_idx}.jpg"), img)
        cv2.imwrite(os.path.join(sample_depth_dir, f"{view_idx}.exr"), depth)
        cv2.imwrite(os.path.join(sample_mask_dir, f"{view_idx}.jpg"), mask)

        # save camera parameters (intrinsics & extrinsics)
        camera_params[f"world_mat_{view_idx}"] = E
        camera_params[f"camera_mat_{view_idx}"] = K

    np.savez(os.path.join(sample_save_dir, "cameras.npz"), **camera_params)


def render_shapenet_samples(
    shapenet_root: str,
    save_dir: str,
    height: int,
    width: int,
    sample_csv: Optional[str] = None,
    textureless: bool = False,
) -> None:
    """
    Renders shapes from ShapeNet located under dataset root directory.

    Args:
        shapenet_root (str): The root directory of ShapeNet dataset.
        save_dir (str): The directory where rendering outputs are saved.
        height (int): The height of output images.
        width (int): The width of output images.
        sample_csv (str): A CSV file containing ShapeNet IDs of 3D shapes to be rendered.
            If None, all 3D shapes found under the root directory are rendered. Set to None by default.
        textureless (bool): The flag for toggling textureless rendering mode. If True, the texture images
            of a 3D shapes are ignored and the mesh is uniformly colored with white and rendered.
            Set to False by default.
    """

    # get sample directories
    if sample_csv is None:
        sample_ids = [d for d in os.listdir(shapenet_root)]
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
            shapenet_root,
            sample_id,
            save_dir,
            height,
            width,
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
    parser.add_argument("--shapenet_path", type=str, default="../ShapeNetCore.v2/02958343")
    parser.add_argument(
        "--sample_csv",
        type=str,
        default="./sedan.csv",
        help="CSV holding IDs samples to be rendered",
    )
    parser.add_argument("--save_path", type=str, default="PaintMe_Data")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--textureless", type=bool, default=True)
    args = parser.parse_args()

    # render
    render_shapenet_samples(
        args.shapenet_path,
        args.save_path,
        args.height,
        args.width,
        args.sample_csv,
        args.textureless,
    )
