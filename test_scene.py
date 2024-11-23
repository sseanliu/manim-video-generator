from manim import *
import numpy as np

class MainScene(Scene):
    def construct(self):
        # Set background color
        self.camera.background_color = "#333333"
        
        # Create objects
        circle = Circle(radius=2, color=BLUE)
        square = Square(side_length=2, color=RED)
        
        # Animate
        self.play(Create(circle))
        self.play(Transform(circle, square))
        
        # Clean up
        self.play(FadeOut(circle))
        self.wait()
