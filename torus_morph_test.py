from manim import *
import numpy as np

class MainScene(ThreeDScene):
    def construct(self):
        # Set background color and camera
        self.camera.background_color = "#333333"
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        
        # Enable ambient camera rotation
        self.begin_ambient_camera_rotation(rate=0.2)
        
        # Create torus
        torus = Surface(
            lambda u, v: np.array([
                (2 + np.cos(v)) * np.cos(u),
                (2 + np.cos(v)) * np.sin(u),
                np.sin(v)
            ]),
            u_range=[0, TAU],
            v_range=[0, TAU],
            resolution=(32, 32),
            checkerboard_colors=[BLUE_D, BLUE_E],
            fill_opacity=0.8
        )
        
        # Create sphere
        sphere = Surface(
            lambda u, v: np.array([
                2 * np.cos(u) * np.cos(v),
                2 * np.cos(u) * np.sin(v),
                2 * np.sin(u)
            ]),
            u_range=[-PI/2, PI/2],
            v_range=[0, TAU],
            resolution=(32, 32),
            checkerboard_colors=[RED_D, RED_E],
            fill_opacity=0.8
        )
        
        # Add lighting
        self.camera.light_source.move_to(3*IN + 7*OUT + 5*RIGHT)
        self.renderer.camera.light_source.set_color(WHITE)
        
        # Initial animation
        self.play(Create(torus), run_time=1.5)
        self.wait(0.5)
        
        # Rotate torus while morphing to sphere
        self.play(
            Transform(torus, sphere),
            Rotate(torus, angle=2*PI, axis=UP),
            rate_func=smooth,
            run_time=3
        )
        
        # Final rotation
        self.play(
            Rotate(torus, angle=2*PI, axis=RIGHT),
            run_time=2
        )
        
        self.wait()
