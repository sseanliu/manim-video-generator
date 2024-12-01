from manim import *
import numpy as np
import os

# Disable all preview functionality
config.preview = False
config.show_in_file_browser = False
config.save_last_frame = False
config.write_to_movie = True
config.disable_caching = True
config.renderer = "cairo"
config.preview_command = ""

class ContainerScene(ThreeDScene):
    def render(self, preview=None):
        """Override render to prevent preview attempts"""
        self.setup()
        self.construct()
        self.tear_down()
        
        # Skip any preview attempts
        if hasattr(self, "renderer") and hasattr(self.renderer, "file_writer"):
            self.renderer.file_writer.close_movie_pipe()

class MainScene(ContainerScene):
    def construct(self):
        # Set camera
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        
        # Create axes
        axes = ThreeDAxes(
            x_range=[-3, 3],
            y_range=[-3, 3],
            z_range=[-2, 4],
            x_length=6,
            y_length=6,
            z_length=4
        )
        
        # Create grid
        grid = NumberPlane(
            x_range=[-3, 3],
            y_range=[-3, 3],
            background_line_style={
                "stroke_opacity": 0.4
            }
        )
        grid.rotate(PI/2, RIGHT)
        
        # Create geometric shapes
        shapes = VGroup()
        
        # Add different geometric objects
        cube = Cube(side_length=1, fill_opacity=0.8)
        sphere = Sphere(radius=0.5, fill_opacity=0.8)
        torus = Torus(major_radius=0.6, minor_radius=0.2, fill_opacity=0.8)
        
        shapes.add(cube, sphere, torus)
        
        # Position shapes
        cube.move_to(np.array([-2, 0, 1]))
        sphere.move_to(np.array([0, 0, 1]))
        torus.move_to(np.array([2, 0, 1]))
        
        # Color gradient for shapes
        for i, shape in enumerate(shapes):
            shape.set_color(interpolate_color(RED, BLUE, i/2))
        
        # Add everything to scene
        self.begin_ambient_camera_rotation(rate=0.2)
        
        # Create and animate axes and grid
        self.play(Create(axes), Create(grid), run_time=1)
        
        # Animate shapes
        self.play(Create(shapes), run_time=2)
        
        # Add some camera movement
        self.wait(2)
        
        # Final rotation
        self.play(
            Rotate(shapes, angle=PI, axis=UP),
            run_time=2
        )
        
        self.wait()
