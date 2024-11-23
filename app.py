from flask import Flask, render_template, request, jsonify, url_for, redirect
import os
from dotenv import load_dotenv
from auth.routes import auth_bp
from auth.clerk_auth import require_auth, get_user_id
import secrets
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# Clerk configuration
app.config['CLERK_PUBLISHABLE_KEY'] = os.getenv('CLERK_PUBLISHABLE_KEY')
app.config['CLERK_SECRET_KEY'] = os.getenv('CLERK_SECRET_KEY')
app.config['CLERK_FRONTEND_API'] = os.getenv('CLERK_FRONTEND_API')

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')

@app.route('/')
@require_auth
def index():
    """Render the main application page."""
    user_id = get_user_id()
    if not user_id:
        return redirect(url_for('auth.sign_in'))
    return render_template('index.html')

@app.route('/settings')
@require_auth
def settings():
    """Render the settings page."""
    user_id = get_user_id()
    if not user_id:
        return redirect(url_for('auth.sign_in'))
    return render_template('settings.html')

@app.route('/validate-openai-key', methods=['POST'])
@require_auth
def validate_openai_key():
    """Validate OpenAI API key."""
    api_key = request.json.get('api_key')
    
    if not api_key:
        return jsonify({'error': 'API key is required'}), 400
        
    if not api_key.startswith('sk-'):
        return jsonify({'error': 'Invalid API key format'}), 400
        
    # Test the API key with a simple request
    try:
        response = requests.get(
            'https://api.openai.com/v1/models',
            headers={'Authorization': f'Bearer {api_key}'}
        )
        if response.status_code == 200:
            return jsonify({'message': 'API key is valid'}), 200
        else:
            return jsonify({'error': 'Invalid API key'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Animation style presets
ANIMATION_STYLES = {
    'default': {
        'background_color': '#333333',
        'animation_time': '2',
        'quality': 'production_quality'
    },
    'presentation': {
        'background_color': '#ffffff',
        'animation_time': '3',
        'quality': 'production_quality'
    },
    'quick': {
        'background_color': '#333333',
        'animation_time': '1',
        'quality': 'medium_quality'
    }
}

def clean_code(code):
    """Clean and format the generated code."""
    # Remove any markdown code block syntax
    code = re.sub(r'```python\s*', '', code)
    code = re.sub(r'```\s*', '', code)
    
    # Fix common decimal literal issues
    code = re.sub(r'(\d+)\.([A-Z_]+)', r'\1 * \2', code)  # Fix decimals with constants
    code = re.sub(r'(\d+\.\d+)([A-Z_]+)', r'\1 * \2', code)  # Fix decimals with constants
    
    # Clean up whitespace
    lines = code.strip().split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove trailing whitespace
        line = line.rstrip()
        # Fix indentation (replace tabs with spaces)
        line = line.replace('\t', '    ')
        cleaned_lines.append(line)
    
    # Ensure proper imports
    code = '\n'.join(cleaned_lines)
    if 'from manim import *' not in code:
        code = 'from manim import *\n' + code
    if 'import numpy as np' not in code:
        code = 'import numpy as np\n' + code
        
    return code.strip()

def get_manim_prompt(prompt, style='default'):
    """Get the detailed prompt for Manim code generation."""
    return f"""
from manim import *
from manim.mobject.three_d.three_d_scene import ThreeDScene
from manim.mobject.three_d.shapes import Box
import numpy as np

class MainScene(ThreeDScene):
    def construct(self):
        # Set basic properties
        self.camera.background_color = "#333333"
        
        # Set camera orientation for 3D
        self.set_camera_orientation(
            phi=75 * DEGREES,
            theta=30 * DEGREES,
            zoom=0.6
        )
        self.begin_ambient_camera_rotation(rate=0.2)
        
        # Create axes
        axes = ThreeDAxes(
            x_range=[-2, 2],
            y_range=[-2, 2],
            z_range=[-2, 2]
        )
        
        # Create parabola
        parabola = ParametricFunction(
            lambda t: np.array([t, t**2, 0]),
            t_range=[-2, 2],
            color=BLUE
        )
        
        # Create box (3D)
        box = Box(
            width=2,
            height=2,
            depth=2,
            fill_color=GREEN,
            fill_opacity=0.7
        )
        box.shift(RIGHT * 2)
        
        # Create rhombus
        rhombus = Polygon(
            np.array([-1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([1, 0, 0]),
            np.array([0, -1, 0]),
            fill_color=YELLOW,
            fill_opacity=0.7
        )
        rhombus.shift(LEFT * 2)
        
        # Group objects
        objects = VGroup(axes, parabola, box, rhombus)
        
        # Animate
        self.play(Create(axes), run_time=1.5)
        self.wait(0.5)
        
        self.play(Create(parabola), run_time=2)
        self.wait(0.5)
        
        self.play(Create(box), run_time=2)
        self.wait(0.5)
        
        self.play(Create(rhombus), run_time=2)
        self.wait(0.5)
        
        # Animate intersections
        self.play(
            box.animate.shift(LEFT * 4),
            rhombus.animate.shift(RIGHT * 4),
            run_time=3
        )
        self.wait(2)
        
        # Clean up
        self.stop_ambient_camera_rotation()
        self.play(*[FadeOut(obj) for obj in objects], run_time=1)
        self.wait(1)
"""

def get_manim_3d_prompt(prompt, style='default'):
    """Get the detailed prompt for Manim 3D code generation with intersecting objects."""
    base_prompt = """Create a Manim animation scene that demonstrates the following in 3D:

{prompt}

Please follow these requirements EXACTLY:
1. Start with ONLY this import: from manim import *
2. Create a class named MainScene that inherits from ThreeDScene
3. For 3D objects, use ONLY these with CORRECT parameters:
   - Cube(side_length=2)
   - Sphere(radius=1)
   - Prism(dimensions=[width, height, depth])
   - Torus(major_radius=1, minor_radius=0.25)
4. For curves, use ParametricFunction with 3D coordinates
5. Set camera orientation using self.set_camera_orientation
6. Use self.play() for animations
7. Add self.wait() between animations
8. Add helpful comments
9. Use ambient camera rotation if needed

Example structure:
from manim import *

class MainScene(ThreeDScene):
    def construct(self):
        # Set up the scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        
        # Create objects with correct parameters
        torus = Torus(major_radius=1, minor_radius=0.25)
        cube = Cube(side_length=2)
        sphere = Sphere(radius=1)
        
        # Create a 3D curve
        curve = ParametricFunction(
            lambda t: np.array([np.cos(t), np.sin(t), t/2]),
            t_range=[-2*PI, 2*PI],
            color=YELLOW
        )
        
        # Animate
        self.play(Create(curve))
        self.wait(1)"""

    return base_prompt.format(prompt=prompt)

def generate_manim_code(prompt, style='default'):
    """Generate Manim code with enhanced prompt engineering."""
    try:
        # Initialize OpenAI client
        client = OpenAI()
        
        # Get the appropriate prompt template
        prompt_template = get_manim_3d_prompt(prompt, style)
        
        # Add specific examples for 3D objects
        system_prompt = """You are a Manim expert. Generate ONLY working Python code for 3D animations.
        IMPORTANT RULES:
        1. Use ONLY 'from manim import *' as the import
        2. For 3D objects use ONLY these with EXACT parameters:
           - Cube(side_length=2)
           - Sphere(radius=1)
           - Prism(dimensions=[width, height, depth])
           - Torus(major_radius=1, minor_radius=0.25)
        3. For curves use ParametricFunction with 3D coordinates
        4. Always set camera orientation
        5. Include proper animations with self.play()
        6. Add wait times between animations
        Do not include any explanations, only generate the code."""
        
        # Generate code using OpenAI with lower temperature for more consistent output
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_template}
            ],
            temperature=0.3,  # Lower temperature for more consistent output
            max_tokens=1500
        )
        
        # Extract and clean the generated code
        generated_code = response.choices[0].message.content
        cleaned_code = clean_code(generated_code)
        
        # Log the generated code for debugging
        logger.debug(f"Generated code:\n{cleaned_code}")
        
        # Validate the generated code
        if not validate_manim_code(cleaned_code):
            logger.error("Code validation failed. Using fallback template...")
            # Use a simple fallback template that we know works
            return """from manim import *

class MainScene(ThreeDScene):
    def construct(self):
        # Set up the scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        
        # Create a 3D curve
        curve = ParametricFunction(
            lambda t: np.array([np.cos(t), np.sin(t), t/2]),
            t_range=[-2*PI, 2*PI],
            color=YELLOW
        )
        
        # Create 3D objects with correct parameters
        cube = Cube(side_length=2, fill_opacity=0.7, color=BLUE)
        cube.shift(RIGHT * 2)
        
        sphere = Sphere(radius=1, fill_opacity=0.7, color=RED)
        sphere.shift(LEFT * 2)
        
        torus = Torus(major_radius=1, minor_radius=0.25, fill_opacity=0.7, color=GREEN)
        
        # Animate
        self.play(Create(curve))
        self.wait(0.5)
        
        self.play(Create(cube))
        self.wait(0.5)
        
        self.play(Create(sphere))
        self.wait(0.5)
        
        self.play(Create(torus))
        self.wait(0.5)
        
        # Rotate the scene
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(2)
        self.stop_ambient_camera_rotation()
        
        # Move objects
        self.play(
            cube.animate.shift(LEFT * 4),
            sphere.animate.shift(RIGHT * 4),
            torus.animate.shift(UP * 2),
            run_time=2
        )
        self.wait(1)
        
        # Clean up
        self.play(
            *[FadeOut(obj) for obj in [curve, cube, sphere, torus]],
            run_time=1
        )
        self.wait(1)"""
        
        return cleaned_code
        
    except Exception as e:
        logger.error(f"Error generating Manim code: {str(e)}")
        logger.error(f"Full error details: {str(e.__class__.__name__)}: {str(e)}")
        # Return a basic working template as fallback
        return """from manim import *

class MainScene(ThreeDScene):
    def construct(self):
        # Set camera orientation for 3D
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        
        # Create a simple 3D curve
        curve = ParametricFunction(
            lambda t: np.array([np.cos(t), np.sin(t), t/2]),
            t_range=[-2*PI, 2*PI],
            color=YELLOW
        )
        
        # Create objects with correct parameters
        cube = Cube(side_length=2, fill_opacity=0.7, color=BLUE)
        sphere = Sphere(radius=1, fill_opacity=0.7, color=RED)
        torus = Torus(major_radius=1, minor_radius=0.25, fill_opacity=0.7, color=GREEN)
        
        # Position objects
        cube.shift(RIGHT * 2)
        sphere.shift(LEFT * 2)
        
        # Animate
        self.play(Create(curve))
        self.wait(0.5)
        self.play(Create(cube))
        self.wait(0.5)
        self.play(Create(sphere))
        self.wait(0.5)
        self.play(Create(torus))
        self.wait(0.5)
        
        # Move objects together
        self.play(
            cube.animate.shift(LEFT * 2),
            sphere.animate.shift(RIGHT * 2),
            torus.animate.shift(UP * 2),
            run_time=2
        )
        self.wait(1)
        
        # Clean up
        self.play(
            FadeOut(curve),
            FadeOut(cube),
            FadeOut(sphere),
            FadeOut(torus)
        )
        self.wait(1)"""

def validate_manim_code(code):
    """Validate the generated Manim code for common issues and structure."""
    try:
        # Clean the code before validation
        code = code.replace('\t', '    ').strip()
        
        # Fix common decimal literal issues before compilation
        code = re.sub(r'(\d+)\.([A-Z_]+)', r'\1 * \2', code)  # Fix decimals with constants
        code = re.sub(r'(\d+\.\d+)([A-Z_]+)', r'\1 * \2', code)  # Fix decimals with constants
        
        # Check for basic syntax errors by attempting to compile the code
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {str(e)}")
            # Log the problematic line
            lines = code.split('\n')
            if hasattr(e, 'lineno') and e.lineno <= len(lines):
                problematic_line = lines[e.lineno - 1]
                logger.error(f"Problematic line ({e.lineno}): {problematic_line}")
            return False
        
        # Required imports check
        required_imports = [
            "from manim import *"
        ]
        
        for imp in required_imports:
            if imp not in code:
                logger.error(f"Missing required import: {imp}")
                return False
        
        # Check class definition with more precise regex
        class_pattern = r'class\s+MainScene\s*\(\s*ThreeDScene\s*\):'
        if not re.search(class_pattern, code):
            logger.error("Invalid or missing MainScene class definition")
            return False
        
        # Check construct method with proper indentation
        construct_pattern = r'def\s+construct\s*\(\s*self\s*\)\s*:'
        if not re.search(construct_pattern, code):
            logger.error("Missing or invalid construct method")
            return False
            
        # Check for required structure elements
        required_elements = {
            "self.play(": "Animation command",
            "self.wait": "Wait command",
            "set_camera_orientation": "Camera orientation for 3D scene"
        }
        
        for element, description in required_elements.items():
            if element not in code:
                logger.error(f"Missing {description}: {element}")
                return False
        
        # Check for balanced code structure
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets.keys():
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    logger.error("Unmatched closing bracket detected")
                    return False
                if char != brackets[stack.pop()]:
                    logger.error("Mismatched brackets detected")
                    return False
                    
        if stack:
            logger.error("Unclosed brackets detected")
            return False
            
        # Check for proper animation structure
        if not re.search(r'self\.play\([^)]+\)', code):
            logger.error("No valid animation command found")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating code: {str(e)}")
        return False

def extract_scene_class(code):
    """Extract the scene class name from the generated code."""
    scene_class_match = re.search(r'class\s+(\w+)\s*\(\s*(?:Scene|ThreeDScene)\s*\):', code)
    if scene_class_match:
        return scene_class_match.group(1)
    else:
        raise ValueError("Could not find scene class definition in generated code")

def create_video(prompt, style='default'):
    """Create a video from the given prompt."""
    try:
        # Step 1: Generate the Manim code
        logger.info("Step 1: Generating Manim code...")
        generated_code = generate_manim_code(prompt, style)
        if not generated_code:
            raise ValueError("Failed to generate Manim code")
        
        # Step 2: Create a unique filename for the video
        video_filename = f"animation_{str(uuid.uuid4())}.mp4"
        output_path = os.path.join('static', 'videos', video_filename)
        
        # Step 3: Validate the generated code
        logger.info("Step 3: Validating generated code...")
        if not validate_manim_code(generated_code):
            raise ValueError("Generated code failed validation")
        
        # Step 4: Set up temporary directory
        logger.info("Step 4: Setting up temporary directory...")
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.debug(f"Created temp directory: {temp_dir}")
            
            # Create scene file
            scene_file = os.path.join(temp_dir, 'scene.py')
            logger.debug(f"Writing scene file to: {scene_file}")
            with open(scene_file, 'w') as f:
                f.write(generated_code)
            
            # Create media directory
            media_dir = os.path.join(temp_dir, 'media')
            os.makedirs(media_dir, exist_ok=True)
            logger.debug(f"Created media directory: {media_dir}")
            
            # Step 5: Run Manim render command
            logger.info("Step 5: Running Manim render command...")
            scene_class = extract_scene_class(generated_code)
            logger.debug(f"Found scene class name: {scene_class}")
            
            cmd = [
                'python',
                '-m',
                'manim',
                'render',
                scene_file,
                scene_class,
                '--quality',
                'h',
                '--format',
                'mp4',
                '--media_dir',
                media_dir
            ]
            
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            logger.debug(f"Command completed with return code: {result.returncode}")
            
            if result.returncode != 0:
                error_msg = f"Manim rendering failed:\nStdout: {result.stdout}\n\nStderr: {result.stderr}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Step 6: Locate the output video file
            logger.info("Step 6: Locating output video file...")
            video_path = os.path.join(media_dir, 'videos', 'scene', '1080p60', 'MainScene.mp4')
            logger.debug(f"Looking for video at: {video_path}")
            
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found at {video_path}")
            
            logger.debug(f"Found video at: {video_path}")
            
            # Step 7: Copy video to static directory
            logger.info("Step 7: Copying video to static directory...")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(video_path, output_path)
            
            logger.info("Successfully generated and saved video")
            return output_path, generated_code
            
    except Exception as e:
        logger.error(f"Error in create_video: {str(e)}")
        return None, str(e)

@app.route('/generate', methods=['POST'])
@require_auth
def generate():
    """Generate a Manim video based on the prompt."""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        style = data.get('style', 'default')
        
        # Get user ID from the authenticated session
        user_id = get_user_id()
        if not user_id:
            return jsonify({'error': 'User not authenticated'}), 401
        
        # Generate the video
        video_path, error = create_video(prompt, style)
        
        if error:
            return jsonify({'error': error}), 400
            
        if not video_path:
            return jsonify({'error': 'Failed to generate video'}), 500
            
        # Return the video URL
        video_url = url_for('static', filename=os.path.relpath(video_path, 'static'))
        return jsonify({'video_url': video_url})
        
    except Exception as e:
        logger.error(f"Error in generate: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/video/<filename>')
def serve_video(filename):
    return send_file(os.path.join('static', 'videos', filename))

if __name__ == '__main__':
    app.run(debug=True, port=5001)
