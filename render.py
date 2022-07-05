# render.py - Functions used for rendering ShapeNet models

import sys

sys.path.append(".")
sys.path.append("..")

import open3d as o3d
import trimesh
import pyrender

from utils.math import *


def render_mesh(
    mesh: trimesh.Scene,
    theta: float,
    phi: float,
    fx: float,
    fy: float,
    height: int,
    width: int,
    znear: float = 0.01,
    zfar: float = 10.0,
    flags=pyrender.RenderFlags.FLAT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Renders a mesh loaded as open3d.geometry.TriangleMesh object.

    Args:
        - view_idx (int):
        - mesh (trimesh.Mesh): A mesh to be rendered.
        - theta (float): Angle between positive direction of y axis and displacement vector.
        - phi (float): Angle between positive direction of x axis and displacement vector.
        - fx (float): A focal length along x-axis of image plane.
        - fy (float): A focal length along y-axis of image plane.
        - height (int): Height of the viewport.
        - width (int): Width of the viewport.
        - znear (float): The nearest visible depth.
        - zfar (float): The farthest visible depth.

    Returns:
        - color (np.ndarray): An array of shape (3, height, width).
        - depth (np.ndarray): An array of shape (1, height, width).
        - mask (np.ndarray): An array of shape (1, height, width).
        - K (np.ndarray): An array of shape (3, 4).
        - E (np.ndarray): An array of shape (3, 4).
    """
    # build camera intrinsic matrix
    K = build_camera_intrinsic(fx, fy, height, width)

    # parse mesh data
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        scene = pyrender.Scene(bg_color=(0.0, 0.0, 0.0))

        verts = np.asarray(mesh.vertices).astype(np.float32)
        faces = np.asarray(mesh.triangles).astype(np.int32)
        colors = np.asarray(mesh.vertex_colors).astype(np.float32)
        normals = np.asarray(mesh.vertex_normals).astype(np.float32)
        mesh = pyrender.Mesh(
            primitives=[
                pyrender.Primitive(
                    positions=verts,
                    normals=normals,
                    color_0=colors,
                    indices=faces,
                    mode=pyrender.GLTF.TRIANGLES,
                )
            ],
            is_visible=True,
        )
        scene.add_node(pyrender.Node(mesh=mesh, matrix=np.eye(4)))
    else:
        scene = pyrender.Scene.from_trimesh_scene(mesh, bg_color=(0.0, 0.0, 0.0))

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
        4.5,
        theta,
        phi,
        np.array([0.0, 1.0, 0.0]),
    )
    cam_node = pyrender.Node(camera=cam, matrix=E)
    scene.add_node(cam_node)

    # add light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=12.5)
    light_mat = np.eye(4)
    light_mat[:-1, 3] = np.array([1.0, 1.0, 1.0])
    light_node = pyrender.Node(light=light, matrix=light_mat)
    scene.add_node(light_node, parent_node=cam_node)

    # render
    render = pyrender.OffscreenRenderer(width, height)
    img, depth = render.render(scene, flags=flags)
    depth[depth == 0.0] = np.inf
    mask = ~np.isinf(depth)
    mask = np.expand_dims(mask, axis=-1)
    mask = np.repeat(mask, 3, axis=-1)
    mask = (mask.astype(np.uint8) * 255).astype(np.uint8)

    return img, depth, mask, K[:3, :], E[:3, :]
