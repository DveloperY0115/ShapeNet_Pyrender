# render.py - Functions used for rendering ShapeNet models

import sys

sys.path.append(".")
sys.path.append("..")

import trimesh
import pyrender

from utils.math import *


def render_mesh(
        view_idx: int,
        mesh: trimesh.Scene,
        theta: float,
        phi: float,
        height: int,
        width: int,
        znear: float = 0.01,
        zfar: float = 1500,
        flags=pyrender.RenderFlags.FLAT,
    ) -> Tuple[np.array, np.array, np.array]:
    """
    Renders a mesh loaded as open3d.geometry.TriangleMesh object.

    Args:
    - mesh: trimesh.Mesh.
        A mesh to be rendered.
    - theta: Float.
        Angle between positive direction of y axis and displacement vector.
    - phi: Float.
        Angle between positive direction of x axis and displacement vector.
    - height: Int.
        Height of the viewport.
    - width: Int.
        Width of the viewport.
    - znear: Float.
        The nearest visible depth.
    - zfar: Float.
        The farthest visible depth.

    Returns:
    - color: A Numpy array of shape (3, height, width).
    - depth: A Numpy array of shape (1, height, width).
    - mask: A Numpy array of shape (1, height, width).
    - K: A Numpy array of shape (3, 4).
    - E: A Numpy array of shape (3, 4).
    """
    # set camera intrinsics
    fx = 7.227512 / 0.0369161
    fy = 12.227512 / 0.0369161

    if view_idx in (0, 4):
        fy = 1.3 * fy
    elif view_idx in (2, 6):
        fx = 1.75 * fx
        fy = 1.75 * fy
    K = build_camera_intrinsic(fx, fy, height, width)
    
    # set camera extrinsics
    E = build_camera_extrinsic(
        1.2, theta, phi,
        np.array([0., 1., 0.])
    )

    # parse mesh data
    scene = pyrender.Scene.from_trimesh_scene(mesh)


    # add camera
    cam = pyrender.IntrinsicsCamera(
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        znear=znear,
        zfar=zfar,
    )
    K = cam.get_projection_matrix(width, height)

    # set camera extrinsics
    E = build_camera_extrinsic(
        1.6, theta, phi,
        np.array([0., 1., 0.])
    )
    cam_node = pyrender.Node(camera=cam, matrix=E)
    scene.add_node(cam_node)

    # add light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3)
    light_node = pyrender.Node(light=light, matrix=np.eye(4))
    scene.add_node(light_node, parent_node=cam_node)

    # render
    render = pyrender.OffscreenRenderer(width, height)
    img, depth = render.render(scene, flags=flags)
    depth[depth == 0.0] = np.inf
    mask = ~np.isinf(depth)
    mask = np.expand_dims(mask, axis=-1)
    mask = np.repeat(mask, 3, axis=-1)
    img_wo_bg = img * mask  # clear out background
    mask = (mask.astype(np.uint8) * 255).astype(np.uint8)

    return img_wo_bg, depth, mask, K[:3, :], E[:3, :]
