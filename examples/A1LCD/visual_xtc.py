from ovito.io import import_file, export_file
from ovito.vis import Viewport
import math



pipeline = import_file("traj_equi.xtc")
pipeline.add_to_scene()

''''''

vp = Viewport()
vp.type = Viewport.Type.Perspective
vp.camera_pos = (-100, -150, 150)
vp.camera_dir = (2, 3, -3)
vp.fov = math.radians(60.0)

vp.render_image(filename="myimage.png", size=(800,600))