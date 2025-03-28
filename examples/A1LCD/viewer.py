import sys
import ovito
from ovito.io import import_file
from ovito.vis import Viewport
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QSlider, QVBoxLayout

# Create a global Qt application object.
app = QApplication(sys.argv)

# Create a Qt main window.
mainwin = QMainWindow()
mainwin.setWindowTitle('OVITO Trajectory Viewer')
central_widget = QWidget()
central_layout = QVBoxLayout(central_widget)
mainwin.setCentralWidget(central_widget)

# Create an OVITO virtual viewport.
vp = Viewport(type=Viewport.Type.Perspective, camera_dir=(2, 1, -1))

# Create the GUI widget associated with the OVITO virtual viewport.
vp_widget = ovito.gui.create_qwidget(vp)
central_layout.addWidget(vp_widget, 1)

# Create a time slider widget.
time_slider = QSlider(Qt.Horizontal)
time_slider.setTickInterval(1)
time_slider.setTickPosition(QSlider.TicksBothSides)
central_layout.addWidget(time_slider)

# When user moves the time slider, update the current time of the OVITO animation settings object.
def on_time_slider(frame: int):
    ovito.scene.anim.current_frame = frame
time_slider.valueChanged.connect(on_time_slider)

# Show the main window.
mainwin.resize(800, 600)
mainwin.show()

# Import a trajectory and add the model to the visualization scene.
pipeline = import_file('traj_equi.xtc')
pipeline.add_to_scene()

# Adjust value range of time slider to reflect number of trajectory frames.
time_slider.setMaximum(pipeline.num_frames - 1)

# Adjust viewport camera to show the entire scene.
vp.zoom_all((vp_widget.width(), vp_widget.height()))

# Start the Qt event loop.
sys.exit(app.exec())