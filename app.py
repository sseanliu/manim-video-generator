from flask import Flask, render_template, request, jsonify, send_file
import os
import tempfile
import subprocess
import logging
import uuid
import shutil
from manim import *
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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

def generate_manim_code(concept):
    """Generate Manim code using GPT based on the given concept"""
    try:
        example_scene = '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create title
        title = Text("Quadratic Function", font_size=40)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))
        
        # Create axes without numbers (to avoid LaTeX)
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-2, 6, 1],
            tips=True,
            axis_config={"include_numbers": False}
        )
        self.play(Create(axes))
        
        # Create quadratic function
        parabola = axes.plot(
            lambda x: x**2,
            x_range=[-2, 2],
            color=BLUE
        )
        self.play(Create(parabola))
        
        # Add simple text labels instead of LaTeX
        x_label = Text("x", font_size=24).next_to(axes.x_axis, DOWN)
        y_label = Text("y", font_size=24).next_to(axes.y_axis, LEFT)
        labels = VGroup(x_label, y_label)
        self.play(Create(labels))
        
        # Add equation as text
        equation = Text("y = x²", font_size=36).next_to(title, DOWN)
        self.play(Write(equation))
        
        self.wait(2)'''

        prompt = f"""Create a Manim scene that explains and visualizes the following mathematical concept: {concept}

        Requirements:
        1. The scene class must be named 'MainScene' and inherit from Scene
        2. Follow this structure for the code:
           - Start with imports
           - Create a MainScene class
           - In construct method, build the visualization step by step
        3. Include these elements:
           - Title text introducing the concept
           - Clear visual elements (graphs, shapes, etc.)
           - Step-by-step animations with self.play()
           - Appropriate wait times with self.wait()
           - Use Text() for labels instead of MathTex() to avoid LaTeX dependencies
        4. For plotting graphs:
           - Create Axes with include_numbers=False to avoid LaTeX
           - Plot functions using: axes.plot(lambda x: f(x), x_range=[min, max])
           - Add labels with Text() class
        5. Use proper Manim syntax for:
           - Creating objects (Circle(), Square(), Axes(), etc.)
           - Animations (Create(), Write(), Transform(), etc.)
           - Positioning (shift(), next_to(), to_edge(), etc.)
        
        Here's an example of good Manim code:
        {example_scene}
        
        Return ONLY the Python code without any markdown formatting or explanation."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a Manim expert who creates educational mathematics animations. 
                    Generate only valid, runnable Manim Python code without any markdown formatting or explanations.
                    Always include proper imports, clear animations, and step-by-step construction of the scene.
                    Use Text() for labels instead of MathTex() to avoid LaTeX dependencies.
                    For plotting graphs, create Axes with include_numbers=False.
                    Include appropriate wait times between animations."""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        manim_code = response.choices[0].message.content.strip()
        
        # Clean up any markdown formatting that might be present
        manim_code = manim_code.replace("```python", "").replace("```", "").strip()
        
        # Add necessary imports if they're not present
        if "from manim import *" not in manim_code:
            manim_code = "from manim import *\n\n" + manim_code
            
        logger.info(f"Generated clean Manim code:\n{manim_code}")
        return manim_code
    except Exception as e:
        logger.error(f"Error generating Manim code: {str(e)}")
        raise

def validate_manim_code(code):
    """Validate the generated Manim code for syntax errors"""
    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error in generated code: {str(e)}"
    except Exception as e:
        return False, f"Error validating code: {str(e)}"

def create_manim_video(concept):
    """Create a Manim video based on the given concept"""
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(dir=app.config['TEMP_DIR'])
        logger.info(f"Created temporary directory: {temp_dir}")
        
        # Generate Manim code
        manim_code = generate_manim_code(concept)
        
        # Validate the generated code
        is_valid, error = validate_manim_code(manim_code)
        if not is_valid:
            raise Exception(f"Invalid Manim code generated: {error}")
        
        # Create scene file
        scene_file = os.path.join(temp_dir, 'scene.py')
        with open(scene_file, 'w') as f:
            f.write(manim_code)
        logger.info(f"Created scene file at: {scene_file}")
        
        # Create media directory inside tmp_dir
        media_dir = os.path.join(temp_dir, 'media')
        os.makedirs(media_dir, exist_ok=True)
        os.makedirs(os.path.join(media_dir, 'videos', 'scene', '720p30'), exist_ok=True)
        os.makedirs(os.path.join(media_dir, 'Tex'), exist_ok=True)
        
        # Set environment variables for Manim
        env = os.environ.copy()
        env['MEDIA_DIR'] = media_dir
        
        # Run Manim command
        cmd = [
            'manim',
            scene_file,
            'MainScene',
            '-pqm',  # preview quality, media file
            '--format=mp4'
        ]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(
            cmd,
            cwd=temp_dir,
            capture_output=True,
            text=True,
            env=env,
            timeout=120  # 2 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Manim command failed with return code {result.returncode}")
            logger.error(f"Stdout: {result.stdout}")
            logger.error(f"Stderr: {result.stderr}")
            raise Exception(f"Failed to generate video: {result.stderr}")
        
        # Find the generated video file
        output_dir = os.path.join(media_dir, 'videos', 'scene', '720p30')
        video_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
        
        if not video_files:
            logger.error(f"No video files found in {output_dir}")
            logger.error(f"Directory contents: {os.listdir(output_dir)}")
            raise Exception("No video file generated")
            
        video_path = os.path.join(output_dir, video_files[0])
        logger.info(f"Found video file: {video_path}")
        
        # Generate a unique filename for the video
        unique_filename = f"manim_video_{uuid.uuid4().hex[:8]}.mp4"
        target_path = os.path.join(app.static_folder, 'videos', unique_filename)
        
        # Move the video file to static directory
        shutil.move(video_path, target_path)
        logger.info(f"Moved video to: {target_path}")
        
        return f"videos/{unique_filename}"
        
    except Exception as e:
        logger.error(f"Error creating video: {str(e)}")
        raise
    
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_video():
    try:
        logger.info("Starting video generation")
        concept = request.form.get('concept', '')
        if not concept:
            return jsonify({'error': 'No concept provided'}), 400
            
        logger.info(f"Received concept: {concept}")
        
        # Generate video with the provided concept
        video_path = create_manim_video(concept)
        if not video_path:
            logger.error("Video generation returned empty path")
            return jsonify({'error': 'Failed to generate video'}), 500

        logger.info(f"Video generated successfully: {video_path}")
        return jsonify({'video_url': f'/static/{video_path}'})
        
    except Exception as e:
        logger.error(f"Error in generate_video route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
