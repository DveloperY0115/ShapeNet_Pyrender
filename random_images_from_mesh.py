import copy
import glob
import os

import numpy as np
import pyrender
import trimesh
from PIL import Image
from tqdm import tqdm

from generate_sources_colors import generate_source_colors
from generate_sources_depths import generate_source_depths

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_DEVICE_ID"] = "3"


def random_images_from_mesh(
    root_dir, out_dir, height, width, colors=True, images_per_mesh=10
):
    files = glob.glob(root_dir + "/**/**.obj", recursive=True)
    img_datas = {}
    index = 0
    for path in tqdm(files):
        try:
            tmesh = trimesh.load(path)
            scene = pyrender.Scene.from_trimesh_scene(tmesh)
        except KeyboardInterrupt:
            break
        except:
            print(f"fail to rander {index}nd mesh")
        else:
            for _ in range(images_per_mesh):
                if colors:
                    img, view_direction = generate_source_colors(scene, height, width)
                    img_name = f"img_{index:08d}.png"
                    img_datas[img_name] = view_direction
                    img = Image.fromarray(img)
                    img.save(out_dir + img_name)
                    index += 1
                else:
                    depth, view_direction, Pi = generate_source_depths(
                        scene, height, width
                    )

    np.save("img_datas.npy", img_datas)


random_images_from_mesh("./ShapeNetCore.v2", "./shape_car_images/", 512, 512)
