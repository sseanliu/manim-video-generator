from manim import *

class TestScene(Scene):
    def construct(self):
        # Create a simple circle
        circle = Circle(radius=2.0)
        circle.set_color(BLUE)
        
        # Create some text
        text = Text("Test Animation")
        text.to_edge(UP)
        
        # Show the circle and text
        self.play(Create(circle))
        self.play(Write(text))
        
        # Animate the circle
        self.play(circle.animate.scale(0.5))
        self.wait(1)
        
        # Fade out
        self.play(FadeOut(circle), FadeOut(text))
