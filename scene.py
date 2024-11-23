from manim import *
import numpy as np

class MainScene(ThreeDScene):
    def construct(self):
        # Set background color and camera
        self.camera.background_color = "#333333"
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
        
        # Create a complex surface
        def complex_surface(u, v):
            return np.array([
                u,
                v,
                0.5 * np.sin(2*u) * np.cos(2*v)
            ])
        
        surface = Surface(
            complex_surface,
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(32, 32),
            fill_opacity=0.7,
            checkerboard_colors=[BLUE_D, BLUE_E],
            stroke_opacity=0.5
        )
        
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
        self.play(Create(axes), Create(grid), run_time=1)
        
        # Animate surface
        self.play(Create(surface), run_time=2)
        self.play(
            surface.animate.shift(UP),
            surface.animate.set_opacity(0.8),
            run_time=1.5
        )
        
        # Animate shapes
        self.play(Create(shapes), run_time=1.5)
        
        # Complex animations
        self.play(
            *[
                Succession(
                    Rotate(shape, angle=2*PI, axis=RIGHT),
                    Rotate(shape, angle=2*PI, axis=UP)
                ) for shape in shapes
            ],
            run_time=3
        )
        
        # Wave animation for surface
        def wave_deform(mob, dt):
            time = self.renderer.time
            mob.become(
                Surface(
                    lambda u, v: np.array([
                        u,
                        v,
                        0.5 * np.sin(2*u + time) * np.cos(2*v)
                    ]),
                    u_range=[-2, 2],
                    v_range=[-2, 2],
                    resolution=(32, 32),
                    fill_opacity=0.7,
                    checkerboard_colors=[BLUE_D, BLUE_E],
                    stroke_opacity=0.5
                ).shift(UP)
            )
        
        surface.add_updater(wave_deform)
        self.wait(3)
        surface.clear_updaters()
        
        # Final rotation
        self.play(
            Rotate(surface, angle=PI, axis=UP),
            *[Rotate(shape, angle=PI, axis=UP) for shape in shapes],
            run_time=2
        )
        
        self.wait()
