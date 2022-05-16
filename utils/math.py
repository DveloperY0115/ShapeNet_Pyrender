# math.py - A collection of math utility functions

from typing import Tuple

import numpy as np

def build_camera_intrinsic(
    fx: float, 
    fy: float,
    height: int,
    width: int,
) -> np.array:
    """
    Build camera intrinsic matrix given
    focal length and viewport dimensions.

    Args:
    - fx: Float.
        Scaling factor along x-axis of the camera frame.
    - fy: Float.
        Scaling factor along y-axis of the camera frame.
    - height: Int.
        Height of the viewport.
    - width: Int.
        Width of the viewport.
        
    Returns:
    - K: Numpy array of shape (3, 3).
        Camera intrinsic matrix.
    """
    K = np.array([fx, 0, width / 2,
                  0, fy, height / 2,
                  0, 0, 1]).reshape((3, 3))
    return K

def build_camera_extrinsic(
    radius: float,
    theta: float,
    phi: float,
    up_v: np.array,
) -> Tuple[np.array, np.array]:
    """
    Build camera extrinsic (rotation, translation) given:
    - camera location represented in spherical coordinate system.
    - camera view direction specified by up & left vectors.

    This function assumes that the object is located at the origin
    of the world frame and the camera is orbiting around it.

    The coordinate system follows OpenGL's convention, using
    (1) x axis: horizontal direction, 
    (2) y axis: vertical direction,
    (3) z axis: direction coming out of the screen, cross product of 
        (1) and (2).

    The translation vector 't' stretching out to a point in 3D space
    with unit length can then be represented as:
    t = [sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi)]

    d    y
    \    |
     \   |
      \  |
       \ |
          -------------- x
        / 
       / 
      /
     z

    Args:
    - radius: Float.
        The length of the displacement vector.
        Set to 1.0 by default to represent unit direction.
    - theta: Float.
        Angle between positive direction of y axis and displacement vector.
    - phi: Float.
        Angle between positive direction of x axis and displacement vector.
    - up_v: Numpy array of shape (3, ).
        Up vector of the camera.
    
    Returns:
    - E: Numpy array of shape (4, 4).
        Camera extrinsic matrix in Affine matrix form.
    """
    # spherical -> cartesian
    t = _spherical_to_cartesian(radius, theta, phi)
    t_unit = t / np.linalg.norm(t)

    # compute camera orientation
    right_v = np.cross(t_unit, up_v)
    right_v = right_v / np.linalg.norm(right_v)
    up_v = np.cross(right_v, t_unit)
    up_v = up_v / np.linalg.norm(up_v)

    up_v = np.reshape(up_v, (3, 1))
    right_v = np.reshape(right_v, (3, 1))
    t = np.reshape(t, (3, 1))
    t_unit = np.reshape(t_unit, (3, 1))
    R = np.concatenate([-right_v, up_v, t_unit], axis=-1)

    # construct 4 x 4 extrinsic matrix
    E = np.concatenate([R, t], axis=-1)
    fourth_row = np.array([0., 0., 0., 1.])
    fourth_row = np.reshape(fourth_row, (1, 4))

    E = np.concatenate([E, fourth_row], axis=0)

    return E

def _spherical_to_cartesian(
    radius: float,
    theta: float,
    phi: float,
) -> np.array:
    """
    Convert a coordinate represented in spherical coordinate system
    to Cartesian coordinate system. This function follows the OpenGL's
    coordinate convention, see the comment of 'build_camera_extrinsic' 
    for details.

    Args:
    - radius: Float.
        The length of the displacement vector.
        Set to 1.0 by default to represent unit direction.
    - theta: Float.
        Angle between positive direction of y axis and displacement vector.
    - phi: Float.
        Angle between positive direction of x axis and displacement vector.

    Returns:
    - t: Numpy array of shape (3, ).
        3D coordinate vector following Cartesian coordinate system.
    """
    t = radius * np.array([
                np.sin(theta) * np.cos(phi), 
                np.cos(theta), 
                np.sin(theta) * np.sin(phi),
            ])
    return t