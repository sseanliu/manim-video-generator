from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
import tempfile
import subprocess
import logging
import uuid
import shutil
import json
from manim import *
import google.generativeai as genai # Import Google AI SDK
from dotenv import load_dotenv
from datetime import datetime
import time
import random
import io
from html_generator import HTMLGenerator # This now uses Google AI

# Load environment variables (will load GOOGLE_API_KEY if present)
load_dotenv()

# Initialize Google AI client for Manim code generation in app.py
gemini_model_manim = None
try:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        # Configure the SDK. It's safe to call this even if html_generator also calls it.
        genai.configure(api_key=google_api_key)

        # Using gemini-2.5-flash-preview-04-17. Adjust if needed.
        gemini_model_manim = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
        logging.info("Google AI SDK configured for Manim generation with gemini-2.5-flash-preview-04-17.")
    else:
        logging.warning("GOOGLE_API_KEY not found. Manim AI generation will fall back to basic code.")
except Exception as e:
    logging.error(f"Failed to initialize Google AI SDK for Manim generation: {e}")

# Initialize Flask app
app = Flask(__name__, 
    static_url_path='/static',
    static_folder='static')
app.logger.setLevel(logging.INFO) # Use Flask's logger for consistency

# Initialize HTML Generator (which also initializes its own Google AI client)
html_generator = HTMLGenerator(app)

# Configure Manim
config.media_dir = "media"
config.video_dir = "videos"
config.images_dir = "images"
config.text_dir = "texts"
config.tex_dir = "tex"
config.log_dir = "log"
config.renderer = "cairo"
config.text_renderer = "cairo"
config.use_opengl_renderer = False

# Set up required directories
def setup_directories():
    """Create all required directories for the application"""
    directories = [
        os.path.join(app.root_path, 'static'),
        os.path.join(app.root_path, 'static', 'videos'),
        os.path.join(app.root_path, 'tmp'),
        os.path.join(app.root_path, 'media'),
        os.path.join(app.root_path, 'media', 'videos'),
        os.path.join(app.root_path, 'media', 'videos', 'scene'),
        os.path.join(app.root_path, 'media', 'videos', 'scene', '720p30'),
        os.path.join(app.root_path, 'media', 'videos', 'scene', '1080p60')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        app.logger.info(f'Created directory: {directory}')

# Set up directories at startup
setup_directories()

# Ensure static directory exists
os.makedirs(os.path.join(app.root_path, 'static', 'videos'), exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set media and temporary directories with fallback to local paths
if os.environ.get('DOCKER_ENV'):
    app.config['MEDIA_DIR'] = os.getenv('MEDIA_DIR', '/app/media')
    app.config['TEMP_DIR'] = os.getenv('TEMP_DIR', '/app/tmp')
else:
    app.config['MEDIA_DIR'] = os.path.join(os.path.dirname(__file__), 'media')
    app.config['TEMP_DIR'] = os.path.join(os.path.dirname(__file__), 'tmp')

# Ensure directories exist
os.makedirs(app.config['MEDIA_DIR'], exist_ok=True)
os.makedirs(app.config['TEMP_DIR'], exist_ok=True)
os.makedirs(os.path.join(app.config['MEDIA_DIR'], 'videos', 'scene', '720p30'), exist_ok=True)
os.makedirs(os.path.join(app.static_folder, 'videos'), exist_ok=True)


def sanitize_input(text):
    """Sanitize input text by removing extra whitespace and newlines"""
    return ' '.join(text.strip().split())

def sanitize_title(text):
    """Sanitize text for use in title"""
    text = sanitize_input(text)
    return text.replace('"', '').replace("'", "").strip()

def generate_manim_prompt(concept):
    """Generate a detailed prompt for GPT to create Manim code"""
    return f"""Create a detailed Manim animation to demonstrate and explain: {concept}.

Create a Scene class named MainScene that follows these requirements:

1. Scene Setup:
   - For 2D concepts like unit circles, use Scene with NumberPlane if axes are needed. When initializing `NumberPlane` or `Axes`, only use documented parameters. Avoid parameters like `background_rectangle` unless you are certain of its validity for the Manim version.
   - Add a clear title.
   - **Initial State for Animated Mobjects**: When mobjects are animated using `ValueTracker` and updaters, ensure their initial creation uses valid, non-degenerate geometry based on the `ValueTracker`'s initial value. A dot on a circle should be at its initial angle, and a radius line drawn to that dot *before* these are used in other mobjects.

2. Visualizing Angles (especially on Unit Circle):
   - To show the angle `theta` itself (the arc), prefer using `Arc(radius=..., start_angle=0, angle=tracker.get_value(), ...)` where `tracker` is a `ValueTracker` for the angle in radians. Add an updater to this Arc to change its `angle` parameter based on the tracker.
   - A `Line` mobject can represent the radius, updated to follow a `Dot` that moves on the circle based on the `tracker`.
   - Labels like `cos(theta)`, `sin(theta)`, and `theta` should be `MathTex` and updated to stay in appropriate positions relative to the changing lines and arc.

3. Code Structure:
   - Start with `from manim import *`.
   - Define `class MainScene(Scene):` with a `construct(self):` method.
   - Use `ValueTracker` for angle animations (e.g., `angle_tracker = ValueTracker(0)` for radians).
   - Use updaters (`.add_updater(...)` or `always_redraw(...)`) for mobjects that change based on the `ValueTracker`.
   - Animate the `ValueTracker` using `self.play(angle_tracker.animate.set_value(...))`.
   - **Fading Out All Mobjects**: If you want to fade out all mobjects at the end of a scene, use `self.play(FadeOut(*self.mobjects))` to correctly unpack the list of mobjects.

Example of Unit Circle Angle Animation (using Arc for the angle sweep):
```python
from manim import *

class MainScene(Scene):
    def construct(self):
        title = Title("Unit Circle Angle")
        self.play(Write(title))

        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            # axis_config={"include_numbers": True} # Example of a valid config
        )
        self.add(plane)

        circle = Circle(radius=1)
        dot = Dot(color=RED)
        radius_line = Line(ORIGIN, dot.get_center(), color=BLUE)
        angle_arc_radius = 0.3
        angle_display = Arc(radius=angle_arc_radius, start_angle=0, angle=0, color=ORANGE) # Initial angle is 0

        angle_tracker = ValueTracker(0) # Angle in radians

        dot.move_to(circle.point_at_angle(angle_tracker.get_value()))
        radius_line.put_start_and_end_on(ORIGIN, dot.get_center())
        self.add(circle, dot, radius_line, angle_display)

        dot.add_updater(lambda m: m.move_to(circle.point_at_angle(angle_tracker.get_value())))
        radius_line.add_updater(lambda m: m.put_start_and_end_on(ORIGIN, dot.get_center()))
        angle_display.add_updater(lambda m: m.become(Arc(radius=angle_arc_radius, start_angle=0, angle=angle_tracker.get_value(), color=ORANGE)))

        self.play(angle_tracker.animate.set_value(2 * PI), run_time=5)
        self.wait()
        # self.play(FadeOut(*self.mobjects)) # Example of fading all
```

Output ONLY valid Manim Python code for a single class `MainScene`. Do not include any explanations or markdown like ```python ... ```. The response must start with `from manim import *`.
"""

def generate_manim_code(concept):
    """Generate Manim code based on the concept using Gemini AI.

    Returns ``None`` if code generation fails."""
    if not gemini_model_manim:
        app.logger.error(
            "Gemini Manim model not initialized. Skipping video generation.")
        return None

    prompt = generate_manim_prompt(concept) # Your existing detailed prompt for Manim
    app.logger.info(f"Attempting to generate Manim code via Gemini for concept: {concept}")
    try:
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            temperature=0.2 # Low temperature for precision in Manim code
        )
        response = gemini_model_manim.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        if not response.candidates or not response.candidates[0].content.parts:
            app.logger.error(
                f"Gemini Manim code generation for '{concept}' returned no content or parts.")
            return None

        manim_code = response.candidates[0].content.parts[0].text.strip()

        # Clean up potential markdown fences (though prompt asks not to use them)
        if manim_code.startswith("```python"):
            manim_code = manim_code[len("```python"):].strip()
        if manim_code.endswith("```"):
            manim_code = manim_code[:-len("```")].strip()
        
        if not (
            "class MainScene(Scene):" in manim_code or
            "class MainScene(ThreeDScene):" in manim_code
        ):
            app.logger.error(
                f"Gemini generated Manim code for '{concept}' does not appear valid (missing MainScene). Received: {manim_code[:300]}")
            return None

        app.logger.info(f"Gemini AI Manim code for '{concept}' generated successfully.")
        return manim_code
    except Exception as e:
        app.logger.error(
            f"Error during Gemini AI Manim code generation for '{concept}': {str(e)}")
        return None

def select_template(concept):
    app.logger.info(f"Defaulting to AI (Gemini) Manim code generation for '{concept}'")
    return generate_manim_code(concept)

def generate_pythagorean_code():
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create triangle
        triangle = Polygon(
            ORIGIN, RIGHT*3, UP*4,
            color=WHITE
        )
        
        # Add labels using Text instead of MathTex
        a = Text("a", font_size=36).next_to(triangle, DOWN)
        b = Text("b", font_size=36).next_to(triangle, RIGHT)
        c = Text("c", font_size=36).next_to(
            triangle.get_center() + UP + RIGHT,
            UP+RIGHT
        )
        
        # Add equation using Text
        equation = Text("a² + b² = c²", font_size=36)
        equation.to_edge(UP)
        
        # Create the animation
        self.play(Create(triangle))
        self.play(Write(a), Write(b), Write(c))
        self.play(Write(equation))
        self.wait()'''

def generate_derivative_code():
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create coordinate system
        axes = Axes(
            x_range=[-2, 2],
            y_range=[-1, 2],
            axis_config={"include_tip": True}
        )
        
        # Add custom labels
        x_label = Text("x").next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y").next_to(axes.y_axis.get_end(), UP)
        
        # Create function
        def func(x):
            return x**2
            
        graph = axes.plot(func, color=BLUE)
        
        # Create derivative function
        def deriv(x):
            return 2*x
            
        derivative = axes.plot(deriv, color=RED)
        
        # Create labels
        func_label = Text("f(x) = x²").set_color(BLUE)
        deriv_label = Text("f'(x) = 2x").set_color(RED)
        
        # Position labels
        func_label.to_corner(UL)
        deriv_label.next_to(func_label, DOWN)
        
        # Create animations
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(graph), Write(func_label))
        self.wait()
        self.play(Create(derivative), Write(deriv_label))
        self.wait()'''

def generate_integral_code():
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create coordinate system
        axes = Axes(
            x_range=[-2, 2],
            y_range=[-1, 2],
            axis_config={"include_tip": True}
        )
        
        # Add custom labels
        x_label = Text("x").next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y").next_to(axes.y_axis.get_end(), UP)
        
        # Create function
        def func(x):
            return x**2
            
        graph = axes.plot(func, color=BLUE)
        
        # Create area
        area = axes.get_area(
            graph,
            x_range=[0, 1],
            color=YELLOW,
            opacity=0.3
        )
        
        # Create labels
        func_label = Text("f(x) = x²").set_color(BLUE)
        integral_label = Text("Area = 1/3").set_color(YELLOW)
        
        # Position labels
        func_label.to_corner(UL)
        integral_label.next_to(func_label, DOWN)
        
        # Create animations
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(graph), Write(func_label))
        self.wait()
        self.play(FadeIn(area), Write(integral_label))
        self.wait()'''

def generate_3d_surface_code():
    return '''from manim import *
import numpy as np

class MainScene(ThreeDScene):
    def construct(self):
        # Set up the axes with better spacing
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-2, 2, 0.5],
            x_length=6,
            y_length=6,
            z_length=4,
            axis_config={"include_tip": True}
        )
        
        # Create surface function
        def param_surface(u, v):
            x = u
            y = v
            z = np.sin(np.sqrt(x**2 + y**2))
            return np.array([x, y, z])
        
        # Create surface with optimized resolution
        surface = Surface(
            lambda u, v: param_surface(u, v),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(20, 20),
            should_make_jagged=False,
            stroke_opacity=0
        )
        
        # Add color and styling
        surface.set_style(
            fill_opacity=0.8,
            stroke_color=BLUE,
            stroke_width=0.5,
            fill_color=BLUE
        )
        surface.set_fill_by_value(
            axes=axes,
            colors=[(RED, -0.5), (YELLOW, 0), (GREEN, 0.5)],
            axis=2
        )
        
        # Set up the scene
        self.set_camera_orientation(
            phi=60 * DEGREES,
            theta=45 * DEGREES,
            zoom=0.6
        )
        
        # Animate
        self.begin_ambient_camera_rotation(rate=0.2)
        self.play(Create(axes))
        self.play(Create(surface))
        self.wait(2)
        self.stop_ambient_camera_rotation()
'''

def generate_sphere_code():
    return '''from manim import *

class MainScene(ThreeDScene):
    def construct(self):
        # Set up the scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        axes = ThreeDAxes(
            x_range=[-3, 3],
            y_range=[-3, 3],
            z_range=[-3, 3],
            x_length=6,
            y_length=6,
            z_length=6
        )
        
        # Create sphere
        radius = 2
        sphere = Surface(
            lambda u, v: np.array([
                radius * np.cos(u) * np.cos(v),
                radius * np.cos(u) * np.sin(v),
                radius * np.sin(u)
            ]),
            u_range=[-PI/2, PI/2],
            v_range=[0, TAU],
            checkerboard_colors=[BLUE_D, BLUE_E],
            resolution=(15, 32)
        )
        
        # Create radius line and label
        radius_line = Line3D(
            start=ORIGIN,
            end=[radius, 0, 0],
            color=YELLOW
        )
        r_label = Text("r", font_size=36).set_color(YELLOW)
        r_label.rotate(PI/2, RIGHT)
        r_label.next_to(radius_line, UP)
        
        # Create volume formula
        volume_formula = Text(
            "V = \\frac{4}{3}\\pi r^3"
        ).to_corner(UL)
        
        # Add everything to scene
        self.add(axes)
        self.play(Create(sphere))
        self.wait()
        self.play(Create(radius_line), Write(r_label))
        self.wait()
        self.play(Write(volume_formula))
        self.wait()
        
        # Rotate camera for better view
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()'''

def generate_cube_code():
    return '''from manim import *

class MainScene(ThreeDScene):
    def construct(self):
        # Set up the scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        axes = ThreeDAxes(
            x_range=[-3, 3],
            y_range=[-3, 3],
            z_range=[-3, 3]
        )
        
        # Create cube
        cube = Cube(side_length=2, fill_opacity=0.7, stroke_width=2)
        cube.set_color(BLUE)
        
        # Labels for sides
        a_label = Text("a", font_size=36).set_color(YELLOW)
        a_label.next_to(cube, RIGHT)
        
        # Surface area formula
        area_formula = Text(
            "A = 6a^2"
        ).to_corner(UL)
        
        # Add everything to scene
        self.add(axes)
        self.play(Create(cube))
        self.wait()
        self.play(Write(a_label))
        self.wait()
        self.play(Write(area_formula))
        self.wait()
        
        # Rotate camera for better view
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()'''

def generate_matrix_code():
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create matrices
        matrix_a = VGroup(
            Text("2  1"),
            Text("1  3")
        ).arrange(DOWN)
        matrix_a.add(SurroundingRectangle(matrix_a))
        
        matrix_b = VGroup(
            Text("1"),
            Text("2")
        ).arrange(DOWN)
        matrix_b.add(SurroundingRectangle(matrix_b))
        
        # Create multiplication symbol and equals sign
        times = Text("×")
        equals = Text("=")
        
        # Create result matrix
        result = VGroup(
            Text("4"),
            Text("7")
        ).arrange(DOWN)
        result.add(SurroundingRectangle(result))
        
        # Position everything
        equation = VGroup(
            matrix_a, times, matrix_b,
            equals, result
        ).arrange(RIGHT)
        
        # Create step-by-step calculations
        calc1 = Text("= [2(1) + 1(2)]")
        calc2 = Text("= [2 + 2]")
        calc3 = Text("= [4]")
        
        calcs = VGroup(calc1, calc2, calc3).arrange(DOWN)
        calcs.next_to(equation, DOWN, buff=1)
        
        # Create animations
        self.play(Create(matrix_a))
        self.play(Create(matrix_b))
        self.play(Write(times), Write(equals))
        self.play(Create(result))
        self.wait()
        
        self.play(Write(calc1))
        self.play(Write(calc2))
        self.play(Write(calc3))
        self.wait()'''

def generate_eigenvalue_code():
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create matrix and vector
        matrix = VGroup(
            Text("2  1"),
            Text("1  2")
        ).arrange(DOWN)
        matrix.add(SurroundingRectangle(matrix))
        
        vector = VGroup(
            Text("v₁"),
            Text("v₂")
        ).arrange(DOWN)
        vector.add(SurroundingRectangle(vector))
        
        # Create lambda and equation
        lambda_text = Text("λ")
        equation = Text("Av = λv")
        
        # Position everything
        group = VGroup(matrix, vector, lambda_text, equation).arrange(RIGHT)
        group.to_edge(UP)
        
        # Create characteristic equation steps
        char_eq = Text("det(A - λI) = 0")
        expanded = Text("|2-λ  1|")
        expanded2 = Text("|1  2-λ|")
        solved = Text("(2-λ)² - 1 = 0")
        result = Text("λ = 1, 3")
        
        # Position steps
        steps = VGroup(
            char_eq, expanded, expanded2,
            solved, result
        ).arrange(DOWN)
        steps.next_to(group, DOWN, buff=1)
        
        # Create animations
        self.play(Create(matrix), Create(vector))
        self.play(Write(lambda_text), Write(equation))
        self.wait()
        
        self.play(Write(char_eq))
        self.play(Write(expanded), Write(expanded2))
        self.play(Write(solved))
        self.play(Write(result))
        self.wait()'''

def generate_complex_code():
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Set up plane
        plane = ComplexPlane()
        self.play(Create(plane))
        
        # Create complex number
        z = 3 + 2j
        dot = Dot([3, 2, 0], color=YELLOW)
        
        # Create vector and labels
        vector = Arrow(
            ORIGIN, dot.get_center(),
            buff=0, color=YELLOW
        )
        re_line = DashedLine(
            ORIGIN, [3, 0, 0], color=BLUE
        )
        im_line = DashedLine(
            [3, 0, 0], [3, 2, 0], color=RED
        )
        
        # Add labels
        z_label = Text("z = 3 + 2i", font_size=36)
        z_label.next_to(dot, UR)
        re_label = Text("Re(z) = 3", font_size=36)
        re_label.next_to(re_line, DOWN)
        im_label = Text("Im(z) = 2", font_size=36)
        im_label.next_to(im_line, RIGHT)
        
        # Animations
        self.play(Create(vector))
        self.play(Write(z_label))
        self.wait()
        self.play(
            Create(re_line),
            Create(im_line)
        )
        self.play(
            Write(re_label),
            Write(im_label)
        )
        self.wait()'''

def generate_diff_eq_code():
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create differential equation
        eq = Text(
            "\\frac{dy}{dx} + 2y = e^x"
        )
        
        # Solution steps
        step1 = Text(
            "y = e^{-2x}\\int e^x \\cdot e^{2x} dx"
        )
        step2 = Text(
            "y = e^{-2x}\\int e^{3x} dx"
        )
        step3 = Text(
            "y = e^{-2x} \\cdot \\frac{1}{3}e^{3x} + Ce^{-2x}"
        )
        step4 = Text(
            "y = \\frac{1}{3}e^x + Ce^{-2x}"
        )
        
        # Arrange equations
        VGroup(
            eq, step1, step2, step3, step4
        ).arrange(DOWN, buff=0.5)
        
        # Create graph
        axes = Axes(
            x_range=[-2, 2],
            y_range=[-2, 2],
            axis_config={"include_tip": True}
        )
        
        # Plot particular solution (C=0)
        graph = axes.plot(
            lambda x: (1/3)*np.exp(x),
            color=YELLOW
        )
        
        # Animations
        self.play(Write(eq))
        self.wait()
        self.play(Write(step1))
        self.wait()
        self.play(Write(step2))
        self.wait()
        self.play(Write(step3))
        self.wait()
        self.play(Write(step4))
        self.wait()
        
        # Show graph
        self.play(
            FadeOut(VGroup(eq, step1, step2, step3, step4))
        )
        self.play(Create(axes), Create(graph))
        self.wait()'''

def generate_trig_code():
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create coordinate plane
        plane = NumberPlane(
            x_range=[-4, 4],
            y_range=[-2, 2],
            axis_config={"include_tip": True}
        )
        
        # Add custom labels
        x_label = Text("x").next_to(plane.x_axis.get_end(), RIGHT)
        y_label = Text("y").next_to(plane.y_axis.get_end(), UP)
        
        # Create unit circle
        circle = Circle(radius=1, color=BLUE)
        
        # Create angle tracker
        theta = ValueTracker(0)
        
        # Create dot that moves around circle
        dot = always_redraw(
            lambda: Dot(
                circle.point_at_angle(theta.get_value()),
                color=YELLOW
            )
        )
        
        # Create lines to show sine and cosine
        x_line = always_redraw(
            lambda: Line(
                start=[circle.point_at_angle(theta.get_value())[0], 0, 0],
                end=circle.point_at_angle(theta.get_value()),
                color=GREEN
            )
        )
        
        y_line = always_redraw(
            lambda: Line(
                start=[0, 0, 0],
                end=[circle.point_at_angle(theta.get_value())[0], 0, 0],
                color=RED
            )
        )
        
        # Create labels that follow the lines
        sin_label = Text("sin(θ)", color=GREEN)
        sin_label.add_updater(lambda m: m.next_to(x_line))

        cos_label = Text("cos(θ)", color=RED)
        cos_label.add_updater(lambda m: m.next_to(y_line))
        
        # Add everything to scene
        self.play(Create(plane), Write(x_label), Write(y_label))
        self.play(Create(circle))
        self.play(Create(dot))
        self.play(Create(x_line), Create(y_line))
        self.play(Write(sin_label), Write(cos_label))
        
        # Animate angle
        self.play(
            theta.animate.set_value(2*PI),
            run_time=4,
            rate_func=linear
        )
        self.wait()'''

def generate_quadratic_code():
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create coordinate system
        axes = Axes(
            x_range=[-4, 4],
            y_range=[-2, 8],
            axis_config={"include_tip": True}
        )
        
        # Add custom labels
        x_label = Text("x").next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y").next_to(axes.y_axis.get_end(), UP)
        
        # Create quadratic function
        def func(x):
            return x**2
            
        graph = axes.plot(
            func,
            color=BLUE,
            x_range=[-3, 3]
        )
        
        # Create labels and equation
        equation = Text("f(x) = x²").to_corner(UL)
        
        # Create dot and value tracker
        x = ValueTracker(-3)
        dot = always_redraw(
            lambda: Dot(
                axes.c2p(
                    x.get_value(),
                    func(x.get_value())
                ),
                color=YELLOW
            )
        )
        
        # Create lines to show x and y values
        v_line = always_redraw(
            lambda: axes.get_vertical_line(
                axes.input_to_graph_point(
                    x.get_value(),
                    graph
                ),
                color=RED
            )
        )
        h_line = always_redraw(
            lambda: axes.get_horizontal_line(
                axes.input_to_graph_point(
                    x.get_value(),
                    graph
                ),
                color=GREEN
            )
        )
        
        # Add everything to scene
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(graph))
        self.play(Write(equation))
        self.play(Create(dot), Create(v_line), Create(h_line))
        
        # Animate x value
        self.play(
            x.animate.set_value(3),
            run_time=6,
            rate_func=there_and_back
        )
        self.wait()'''

def generate_3d_surface_code():
    return '''from manim import *

class MainScene(ThreeDScene):
    def construct(self):
        # Configure the scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        
        # Create axes
        axes = ThreeDAxes()
        
        # Create surface
        def func(x, y):
            return np.sin(x) * np.cos(y)
            
        surface = Surface(
            lambda u, v: axes.c2p(u, v, func(u, v)),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=32,
            checkerboard_colors=[BLUE_D, BLUE_E]
        )
        
        # Add custom labels
        x_label = Text("x").next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y").next_to(axes.y_axis.get_end(), UP)
        z_label = Text("z").next_to(axes.z_axis.get_end(), OUT)
        
        # Create animations
        self.begin_ambient_camera_rotation(rate=0.2)
        self.play(Create(axes), Write(x_label), Write(y_label), Write(z_label))
        self.play(Create(surface))
        self.wait(2)
        self.stop_ambient_camera_rotation()
        self.wait()'''

def generate_sphere_code():
    return '''from manim import *

class MainScene(ThreeDScene):
    def construct(self):
        # Configure the scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        
        # Create axes
        axes = ThreeDAxes()
        
        # Create sphere
        sphere = Surface(
            lambda u, v: np.array([
                np.cos(u) * np.cos(v),
                np.cos(u) * np.sin(v),
                np.sin(u)
            ]),
            u_range=[-PI/2, PI/2],
            v_range=[0, TAU],
            checkerboard_colors=[BLUE_D, BLUE_E]
        )
        
        # Add custom labels
        x_label = Text("x").next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y").next_to(axes.y_axis.get_end(), UP)
        z_label = Text("z").next_to(axes.z_axis.get_end(), OUT)
        
        # Create animations
        self.begin_ambient_camera_rotation(rate=0.2)
        self.play(Create(axes), Write(x_label), Write(y_label), Write(z_label))
        self.play(Create(sphere))
        self.wait(2)
        self.stop_ambient_camera_rotation()
        self.wait()'''

def generate_basic_visualization_code():
    """Generate code for basic visualization."""
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create title
        title = Text("Mathematical Visualization", font_size=36).to_edge(UP)
        
        # Create axes
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-3, 3, 1],
            axis_config={"include_tip": True},
            x_length=10,
            y_length=6
        )
        
        # Add labels
        x_label = Text("x", font_size=24).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y", font_size=24).next_to(axes.y_axis.get_end(), UP)
        
        # Create function graphs
        sin_graph = axes.plot(lambda x: np.sin(x), color=BLUE)
        cos_graph = axes.plot(lambda x: np.cos(x), color=RED)
        
        # Create labels for functions
        sin_label = Text("sin(x)", font_size=24, color=BLUE).next_to(sin_graph, UP)
        cos_label = Text("cos(x)", font_size=24, color=RED).next_to(cos_graph, DOWN)
        
        # Create dot to track movement
        moving_dot = Dot(color=YELLOW)
        moving_dot.move_to(axes.c2p(-5, 0))
        
        # Create path for dot to follow
        path = VMobject()
        path.set_points_smoothly([
            axes.c2p(x, np.sin(x)) 
            for x in np.linspace(-5, 5, 100)
        ])
        
        # Animate everything
        self.play(Write(title))
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(sin_graph), Write(sin_label))
        self.play(Create(cos_graph), Write(cos_label))
        self.play(Create(moving_dot))
        
        # Animate dot following the sine curve
        self.play(
            MoveAlongPath(moving_dot, path),
            run_time=3,
            rate_func=linear
        )
        
        # Final pause
        self.wait()
'''

def generate_video(concept, temp_dir):
    """Generate a video visualization for the given concept"""
    try:
        concept = sanitize_input(concept)
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
        filename = f'scene_{timestamp}_{random_str}'
        
        # Get appropriate template code
        try:
            manim_code = select_template(concept.lower())
        except Exception as template_error:
            logger.error(f'Template selection error: {str(template_error)}')
            manim_code = None
        
        if not manim_code:
            return None
        
        # Write code to temporary file
        code_file = os.path.join(temp_dir, 'scene.py')
        with open(code_file, 'w') as f:
            f.write(manim_code)
        
        app.logger.info(f"--- AI Generated Manim Code ---\n{manim_code}\n-------------------------------")

        # Create media directory
        media_dir = os.path.join(temp_dir, 'media')
        os.makedirs(media_dir, exist_ok=True)
        
        # Run manim command with error handling
        output_file = os.path.join(app.static_folder, 'videos', f'{filename}.mp4')
        command = [
            'manim',
            'render',
            '-qm',  # medium quality
            '--format', 'mp4',
            '--media_dir', media_dir,
            code_file,
            'MainScene'
        ]
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=temp_dir,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else 'Unknown error during animation generation'
            logger.error(f'Manim error: {error_msg}')
            return None
        
        # Look for the video file in multiple possible locations
        possible_paths = [
            os.path.join(media_dir, 'videos', 'scene', '1080p60', 'MainScene.mp4'),
            os.path.join(media_dir, 'videos', 'scene', '720p30', 'MainScene.mp4'),
            os.path.join(media_dir, 'videos', 'MainScene.mp4'),
            os.path.join(temp_dir, 'MainScene.mp4')
        ]
        
        video_found = False
        for source_path in possible_paths:
            if os.path.exists(source_path):
                shutil.move(source_path, output_file)
                video_found = True
                break
        
        if not video_found:
            logger.error(f'Video not found in any of these locations: {possible_paths}')
            return None
        
        # Attempt to generate an HTML visualization if possible
        try:
            html_template = html_generator.generate_visualization(concept)
            html_url = html_generator.get_template_url(html_template) if html_template else None
        except Exception as e:
            app.logger.error(f"Error generating HTML visualization: {str(e)}")
            html_url = None
        
        return {
            'success': True,
            'video_url': url_for('static', filename=f'videos/{filename}.mp4'),
            'code': manim_code,
            'html_url': html_url
        }
        
    except subprocess.TimeoutExpired:
        logger.error('Animation generation timed out')
        return None
    except Exception as e:
        logger.error(f'Error in generate_video: {str(e)}')
        return None

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/visualization/<template_name>')
def serve_visualization(template_name):
    """Serve an HTML visualization"""
    return render_template(template_name)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        concept = data.get('concept', '')
        
        if not concept:
            return jsonify({'error': 'No concept provided'}), 400
            
        # Generate unique ID for this request
        request_id = str(uuid.uuid4())
        
        # Create temporary directory for this request
        temp_dir = os.path.join(app.config['TEMP_DIR'], request_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate both video and HTML visualization
        video_result = generate_video(concept, temp_dir)
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Return both video and HTML URLs
        response = {
            'video_url': video_result['video_url'] if video_result else None,
            'html_url': video_result['html_url'] if video_result else None
        }
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Error in generate endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/videos/<path:filename>')
def serve_video(filename):
    """Serve video files from static/videos directory."""
    try:
        return send_from_directory(
            os.path.join(app.root_path, 'static', 'videos'),
            filename,
            mimetype='video/mp4'
        )
    except Exception as e:
        app.logger.error(f"Error serving video {filename}: {str(e)}")
        return jsonify({'error': 'Video not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
