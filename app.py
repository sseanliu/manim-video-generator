from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import tempfile
import subprocess
import logging
import uuid
import shutil
import json
from manim import *
import openai
from dotenv import load_dotenv
from datetime import datetime
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

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
    """Generate a structured prompt for mathematical animations"""
    return f"""Create a Manim animation script that explains: "{concept}"

Follow these Manim Community guidelines:
1. Use appropriate mathematical objects and animations:
   - For geometry: Polygon, Circle, Square, Triangle, etc.
   - For graphs: Axes, NumberPlane, ParametricFunction
   - For 3D: ThreeDAxes, Surface, ParametricSurface
   - For text: MathTex for LaTeX, Text for regular text
   
2. Include these components:
   - Mathematical formulas using MathTex
   - Step-by-step animations with Write, Transform, Create
   - Clear labels and explanations
   - Appropriate coordinate systems (Axes, NumberPlane)
   - Color coding for different elements
   
3. Use these specific Manim features:
   - VMobject for custom shapes
   - ValueTracker for continuous animations
   - UpdateFromFunc for dynamic updates
   - always_redraw for dependent objects
   - MathTex for LaTeX equations
   
Example structure for a mathematical concept:
```python
from manim import *

class MainScene(Scene):
    def construct(self):
        # 1. Title and introduction
        title = Title(f"{concept}")
        self.play(Write(title))
        
        # 2. Setup coordinate system if needed
        plane = NumberPlane(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_opacity": 0.6
            }
        )
        
        # 3. Mathematical objects (use appropriate ones)
        equation = MathTex(r"f(x) = x^2")
        graph = plane.plot(lambda x: x**2, color=YELLOW)
        
        # 4. Step-by-step animations
        self.play(Create(plane))
        self.play(Write(equation))
        self.play(Create(graph))
        
        # 5. Explanatory animations
        dot = Dot().move_to(plane.c2p(2, 4))
        label = MathTex("(2, 4)").next_to(dot, RIGHT)
        
        # 6. Dynamic animations
        x_tracker = ValueTracker(2)
        dot.add_updater(
            lambda m: m.move_to(
                plane.c2p(
                    x_tracker.get_value(),
                    x_tracker.get_value()**2
                )
            )
        )
        
        # 7. Transformations and highlights
        self.play(x_tracker.animate.set_value(-2))
        
        # 8. Final summary
        summary = Text("This demonstrates f(x) = x²").to_edge(DOWN)
        self.play(Write(summary))
        self.wait()
```

Focus on:
1. Mathematical accuracy and clarity
2. Step-by-step visual explanations
3. Proper use of LaTeX for equations
4. Dynamic animations where appropriate
5. Clear visual hierarchy
6. Color-coding for understanding
7. Smooth transitions between steps

Use these Manim-specific features:
1. MathTex for equations
2. NumberPlane or Axes for coordinates
3. ValueTracker for animations
4. always_redraw for dynamic elements
5. Transform for smooth transitions
6. Create for geometric objects
7. Write for text and equations"""

def generate_pythagorean_code():
    """Generate specialized code for Pythagorean theorem visualization"""
    return '''from manim import *
import numpy as np
import math

class MainScene(Scene):
    def construct(self):
        # Create title
        title = Title("Pythagorean Theorem: a² + b² = c²")
        self.play(Write(title))
        
        # Create right triangle
        a, b = 3, 4
        c = math.sqrt(a**2 + b**2)
        
        triangle = Polygon(
            ORIGIN,
            RIGHT * a,
            RIGHT * a + UP * b,
            color=WHITE
        )
        
        # Add right angle square
        square_size = 0.3
        right_angle = Polygon(
            RIGHT * square_size + UP * 0,
            RIGHT * square_size + UP * square_size,
            RIGHT * 0 + UP * square_size,
            RIGHT * 0 + UP * 0,
            color=WHITE
        )
        right_angle.shift(RIGHT * 0 + UP * 0)
        
        # Create labels
        a_label = MathTex("a").next_to(triangle, DOWN)
        b_label = MathTex("b").next_to(triangle, RIGHT)
        c_label = MathTex("c").next_to(
            triangle.get_center() + UP * 0.5 + RIGHT * 0.5,
            UP + RIGHT
        )
        
        # Create squares
        a_square = Square(side_length=a, color=BLUE).shift(RIGHT * a/2 + DOWN * a/2)
        b_square = Square(side_length=b, color=RED).shift(RIGHT * (a + b/2) + UP * b/2)
        c_square = Square(side_length=c, color=GREEN)
        c_square.rotate(angle=math.atan2(b, a))
        c_square.shift(RIGHT * a/2 + UP * b/2)
        
        # Add area labels
        a_square_label = MathTex("a^2").move_to(a_square.get_center())
        b_square_label = MathTex("b^2").move_to(b_square.get_center())
        c_square_label = MathTex("c^2").move_to(c_square.get_center())
        
        # Create equation
        equation = MathTex("a^2", "+", "b^2", "=", "c^2")
        equation.to_edge(DOWN).shift(UP)
        
        # Animate everything
        self.play(Create(triangle), Create(right_angle))
        self.play(Write(a_label), Write(b_label), Write(c_label))
        self.wait()
        
        # Show squares
        self.play(
            Create(a_square),
            Create(b_square),
            Create(c_square)
        )
        self.play(
            Write(a_square_label),
            Write(b_square_label),
            Write(c_square_label)
        )
        self.wait()
        
        # Show equation
        self.play(
            TransformFromCopy(a_square_label, equation[0]),
            Write(equation[1]),
            TransformFromCopy(b_square_label, equation[2]),
            Write(equation[3]),
            TransformFromCopy(c_square_label, equation[4])
        )
        self.wait(2)
        
        # Highlight the relationship
        self.play(
            Indicate(a_square, color=BLUE),
            Indicate(equation[0], color=BLUE)
        )
        self.play(
            Indicate(b_square, color=RED),
            Indicate(equation[2], color=RED)
        )
        self.play(
            Indicate(c_square, color=GREEN),
            Indicate(equation[4], color=GREEN)
        )
        self.wait(2)
'''

def generate_quadratic_code():
    """Generate code for quadratic function visualization"""
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 9, 1],
            axis_config={"include_tip": True},
        ).add_coordinates()
        
        # Add labels
        labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        # Create quadratic function
        def quadratic(x):
            return x**2
            
        graph = axes.plot(quadratic, color=GREEN)
        
        # Create graph label
        graph_label = MathTex("f(x) = x^2").next_to(graph, UR)
        
        # Animate
        self.play(Create(axes), Create(labels))
        self.play(Create(graph), Write(graph_label))
        
        # Create moving dot and value label
        dot = Dot(color=YELLOW)
        dot.move_to(axes.c2p(-3, quadratic(-3)))
        value_label = MathTex("").next_to(dot, UP)
        
        def update_value_label(label):
            x = axes.p2c(dot.get_center())[0]
            y = quadratic(x)
            label.become(MathTex(f"({x:.1f}, {y:.1f})").next_to(dot, UP))
        
        value_label.add_updater(update_value_label)
        
        # Add everything to scene
        self.play(Create(dot), Create(value_label))
        
        # Animate dot movement
        self.play(
            MoveAlongPath(
                dot,
                graph,
                rate_func=linear,
                run_time=5
            )
        )
        self.wait()
'''

def generate_3d_sphere_code():
    """Generate code for 3D sphere visualization"""
    return '''from manim import *

class MainScene(ThreeDScene):
    def construct(self):
        # Configure the camera
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.camera.set_zoom(0.8)
        
        # Create the axes
        axes = ThreeDAxes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            z_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            z_length=4
        )
        
        # Create the sphere
        sphere = Surface(
            lambda u, v: np.array([
                np.cos(u) * np.cos(v),
                np.cos(u) * np.sin(v),
                np.sin(u)
            ]),
            u_range=[-PI/2, PI/2],
            v_range=[0, TAU],
            resolution=(20, 40),
            checkerboard_colors=[BLUE_D, BLUE_E],
            stroke_color=WHITE,
            stroke_width=0.5
        )
        
        # Create equation and title
        title = Text("Unit Sphere").to_corner(UL)
        equation = MathTex("x^2 + y^2 + z^2 = 1").next_to(title, DOWN)
        
        # Add everything to the scene
        self.add_fixed_in_frame_mobjects(title, equation)
        self.play(
            Write(title),
            Write(equation)
        )
        
        # Show the axes
        self.play(Create(axes))
        
        # Show the sphere
        self.play(Create(sphere))
        
        # Rotate the scene
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(2)
        
        # Move camera to different views
        self.move_camera(phi=0, theta=-90 * DEGREES, run_time=2)
        self.wait()
        self.move_camera(phi=90 * DEGREES, theta=0, run_time=2)
        self.wait()
        
        # Return to original position
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES, run_time=2)
        self.wait()
        
        self.stop_ambient_camera_rotation()
'''

def generate_cube_surface_area_code():
    """Generate code for 3D cube surface area visualization"""
    return '''from manim import *

class MainScene(ThreeDScene):
    def construct(self):
        # Configure the camera
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        
        # Create the axes
        axes = ThreeDAxes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            z_range=[-2, 2, 1]
        )
        
        # Create cube
        cube = Cube(side_length=2, fill_opacity=0.7, stroke_width=2)
        cube.set_color(BLUE)
        
        # Create labels
        title = Text("Cube Surface Area").to_corner(UL)
        formula = MathTex("SA = 6a^2").next_to(title, DOWN)
        side_label = MathTex("a = 2").next_to(formula, DOWN)
        
        # Add everything to scene
        self.add_fixed_in_frame_mobjects(title, formula, side_label)
        
        # Show cube construction
        self.play(
            Create(axes),
            Write(title)
        )
        self.play(Create(cube))
        self.wait()
        
        # Show formula
        self.play(Write(formula))
        self.play(Write(side_label))
        self.wait()
        
        # Highlight faces
        faces = cube.get_faces()
        colors = [RED, GREEN, BLUE, YELLOW, PURPLE, ORANGE]
        
        for face, color in zip(faces, colors):
            face_copy = face.copy()
            face_copy.set_color(color)
            face_copy.set_opacity(0.8)
            self.play(
                Transform(face, face_copy),
                run_time=0.5
            )
        
        # Calculate surface area
        result = MathTex("SA = 6(2)^2 = 24").next_to(side_label, DOWN)
        self.add_fixed_in_frame_mobjects(result)
        self.play(Write(result))
        
        # Rotate to show all faces
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(2)
        
        # Move camera to different views
        self.move_camera(phi=0, theta=-90 * DEGREES, run_time=2)
        self.wait()
        self.move_camera(phi=90 * DEGREES, theta=0, run_time=2)
        self.wait()
        
        # Return to original position
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES, run_time=2)
        self.wait()
        
        self.stop_ambient_camera_rotation()
'''

def generate_manim_code(concept):
    """Generate specialized Manim code based on mathematical concept"""
    try:
        concept = sanitize_input(concept)
        logger.info(f"Generating specialized Manim code for: {concept}")
        
        # Convert to lowercase for better matching
        concept_lower = concept.lower()
        
        # Define concept keywords and their corresponding templates
        concept_mapping = {
            'cube': {
                'keywords': ['cube', 'box', 'cubic'],
                'surface_area': generate_cube_surface_area_code,
                'default': generate_cube_surface_area_code
            },
            'sphere': {
                'keywords': ['sphere', 'ball', 'spherical'],
                'surface_area': generate_3d_sphere_code,
                'default': generate_3d_sphere_code
            },
            'quadratic': {
                'keywords': ['quadratic', 'parabola', 'x^2'],
                'default': generate_quadratic_code
            },
            'pythagorean': {
                'keywords': ['pythagorean', 'pythagoras', 'right triangle'],
                'default': generate_pythagorean_code
            }
        }
        
        # First, try to match the main concept
        for concept_type, mapping in concept_mapping.items():
            if any(keyword in concept_lower for keyword in mapping['keywords']):
                # Then check for specific aspects
                if 'surface area' in concept_lower and 'surface_area' in mapping:
                    return mapping['surface_area']()
                return mapping['default']()
                
        # If no specific template matches, use GPT to generate code
        prompt = f"""Create a Manim animation to demonstrate: {concept}

Please generate Python code that:
1. Creates a Scene class named MainScene
2. Uses appropriate Manim objects (MathTex, NumberPlane, etc.)
3. Includes step-by-step animations with self.play()
4. Adds clear mathematical labels and explanations
5. Uses dynamic animations where appropriate
6. Follows Manim best practices
7. Uses proper LaTeX commands with MathTex

Only output valid Manim Python code without any additional text or markdown."""
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a Manim expert specializing in mathematical animations.
                Create precise Manim code following these principles:
                1. Use appropriate mathematical objects
                2. Include step-by-step animations
                3. Add clear mathematical explanations
                4. Use dynamic animations where appropriate
                5. Follow Manim best practices
                6. Use proper LaTeX syntax
                Only output valid Manim Python code."""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        code = response.choices[0].message.content.strip()
        code = code.replace("```python", "").replace("```", "").strip()
        
        template = f'''from manim import *
import numpy as np
import math

{code}'''

        logger.info("Successfully generated specialized Manim code")
        return template

    except Exception as e:
        logger.error(f"Error generating Manim code: {str(e)}")
        return generate_quadratic_code()  # Use quadratic code as fallback

def validate_manim_code(code):
    """Validate the generated Manim code"""
    try:
        # Check for required imports
        if 'from manim import *' not in code:
            logger.error("Missing manim import")
            return False

        # Check for MainScene class
        if 'class MainScene' not in code:
            logger.error("Missing MainScene class")
            return False

        # Check for construct method
        if 'def construct(self)' not in code:
            logger.error("Missing construct method")
            return False

        # Check for potentially harmful operations
        dangerous_terms = ['os.system', 'subprocess', 'eval', 'exec', 'import os']
        for term in dangerous_terms:
            if term in code:
                logger.error(f"Found potentially harmful code: {term}")
                return False

        return True
    except Exception as e:
        logger.error(f"Error validating code: {str(e)}")
        return False

def setup_display():
    """Setup X11 display for Manim"""
    try:
        # Create necessary directories with proper permissions
        os.makedirs('/tmp/.X11-unix', mode=0o1777, exist_ok=True)
        os.makedirs('/tmp/.ICE-unix', mode=0o1777, exist_ok=True)
        
        # Set display environment variable
        os.environ['DISPLAY'] = ':99'
        os.environ['MPLBACKEND'] = 'Agg'
        
        # Configure font paths
        font_config = os.path.expanduser('~/.config/fontconfig')
        os.makedirs(font_config, exist_ok=True)
        
        return True
    except Exception as e:
        logger.error(f"Error setting up display: {str(e)}")
        return False

def setup_directories():
    """Setup necessary directories for video generation"""
    try:
        # Create main directories
        media_dir = os.path.join(app.root_path, 'media')
        static_dir = os.path.join(app.root_path, 'static')
        static_videos = os.path.join(static_dir, 'videos')
        
        # Ensure all required directories exist
        for directory in [media_dir, static_dir, static_videos]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    except Exception as e:
        logger.error(f"Error setting up directories: {str(e)}")
        raise

@app.before_first_request
def initialize():
    """Initialize application settings"""
    try:
        # Setup media directories
        media_dir = os.path.join(app.root_path, 'media')
        os.makedirs(media_dir, exist_ok=True)
        
        static_videos = os.path.join(app.static_folder, 'videos')
        os.makedirs(static_videos, exist_ok=True)
        
        # Setup display
        if not setup_display():
            logger.warning("Failed to setup display, but continuing...")
            
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}")
        
def cleanup_old_files():
    """Cleanup old temporary files"""
    try:
        media_dir = os.path.join(app.root_path, 'media')
        for item in os.listdir(media_dir):
            if item.startswith('tmp_'):
                path = os.path.join(media_dir, item)
                if os.path.isdir(path) and (time.time() - os.path.getctime(path)) > 3600:
                    shutil.rmtree(path)
    except Exception as e:
        logger.error(f"Error cleaning up old files: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_video():
    """Generate a video based on the concept"""
    temp_dir = None
    try:
        logger.info("Starting video generation")
        
        # Ensure directories exist
        setup_directories()
        
        # Cleanup old files
        cleanup_old_files()
        
        # Get form data
        if request.is_json:
            data = request.get_json()
            concept = data.get('concept', '')
        else:
            concept = request.form.get('concept', '')
            
        if not concept:
            logger.error("No concept provided")
            return jsonify({
                'success': False,
                'error': 'No concept provided'
            }), 400

        logger.info(f"Received concept: {concept}")

        # Create temporary directory with unique name
        media_dir = os.path.join(app.root_path, 'media')
        temp_dir = os.path.join(media_dir, f'tmp_{int(time.time())}')
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"Created temporary directory: {temp_dir}")

        # Create all necessary subdirectories
        video_dir = os.path.join(temp_dir, 'videos', 'scene', '1080p30')
        os.makedirs(video_dir, exist_ok=True)
        logger.info(f"Created video directory: {video_dir}")

        try:
            # Generate and validate code
            code = generate_manim_code(concept)
            if not validate_manim_code(code):
                raise ValueError("Generated code failed validation")

            # Create scene file
            scene_file = os.path.join(temp_dir, 'scene.py')
            with open(scene_file, 'w') as f:
                f.write(code)
            logger.info(f"Created scene file at: {scene_file}")

            # Setup environment for Manim
            env = os.environ.copy()
            env['DISPLAY'] = ':99'
            env['MPLBACKEND'] = 'Agg'
            env['MEDIA_DIR'] = temp_dir  # Set media directory explicitly

            # Run Manim command with improved options
            cmd = [
                "manim",
                "-pqm",
                "--format=mp4",
                "--media_dir", temp_dir,
                scene_file,
                "MainScene"
            ]
            logger.info(f"Running Manim command: {' '.join(cmd)}")
            
            # Run command with timeout
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=temp_dir,
                text=True,
                env=env
            )
            
            try:
                stdout, stderr = process.communicate(timeout=300)
                logger.info(f"Manim stdout: {stdout}")
                if stderr:
                    logger.warning(f"Manim stderr: {stderr}")
            except subprocess.TimeoutExpired:
                process.kill()
                raise Exception("Video generation timed out after 5 minutes")

            if process.returncode != 0:
                raise Exception(f"Manim failed with return code {process.returncode}")

            # Look for the video file in all possible locations
            possible_paths = [
                os.path.join(video_dir, 'MainScene.mp4'),
                os.path.join(temp_dir, 'videos', 'scene', '1080p30', 'MainScene.mp4'),
                os.path.join(temp_dir, 'videos', 'scene', '720p30', 'MainScene.mp4')
            ]

            video_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    video_file = path
                    logger.info(f"Found video file at: {path}")
                    break

            if not video_file:
                # List contents for debugging
                logger.error("Video file not found. Directory contents:")
                for root, dirs, files in os.walk(temp_dir):
                    logger.error(f"Directory: {root}")
                    logger.error(f"Subdirectories: {dirs}")
                    logger.error(f"Files: {files}")
                raise FileNotFoundError("Video file not found in any expected location")

            # Copy to static directory
            static_videos = os.path.join(app.static_folder, 'videos')
            os.makedirs(static_videos, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'video_{timestamp}.mp4'
            output_path = os.path.join(static_videos, output_file)
            
            shutil.copy2(video_file, output_path)
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Failed to copy video to: {output_path}")

            logger.info(f"Video generated successfully: {output_file}")
            return jsonify({
                'success': True,
                'video_url': url_for('static', filename=f'videos/{output_file}')
            })

        except Exception as e:
            logger.error(f"Error in video generation process: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error in generate_video route: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {str(e)}")

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up temporary files"""
    try:
        # Clean up temporary directories
        static_tmp = os.path.join(app.static_folder, 'tmp')
        if os.path.exists(static_tmp):
            for d in os.listdir(static_tmp):
                try:
                    path = os.path.join(static_tmp, d)
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        logger.info(f"Cleaned up temporary directory: {path}")
                except Exception as e:
                    logger.error(f"Error cleaning temp dir {d}: {str(e)}")
            
            # Remove the tmp directory itself
            try:
                shutil.rmtree(static_tmp)
                logger.info("Cleaned up main temporary directory")
            except Exception as e:
                logger.error(f"Error cleaning main temp dir: {str(e)}")
        
        # Clean up old videos (keep last 10)
        videos_dir = os.path.join(app.static_folder, 'videos')
        if os.path.exists(videos_dir):
            videos = sorted([
                os.path.join(videos_dir, f) 
                for f in os.listdir(videos_dir) 
                if f.endswith('.mp4')
            ], key=os.path.getctime)
            
            # Keep only the 10 most recent videos
            for video in videos[:-10]:
                try:
                    os.remove(video)
                    logger.info(f"Removed old video: {video}")
                except Exception as e:
                    logger.error(f"Error removing old video {video}: {str(e)}")
        
        return jsonify({
            'success': True,
            'message': 'Cleanup completed successfully'
        })
    
    except Exception as e:
        logger.error(f"Error in cleanup route: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
