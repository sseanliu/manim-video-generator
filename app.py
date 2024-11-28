from flask import Flask, render_template, request, jsonify, send_file
import os
import tempfile
import subprocess
import logging
import uuid
import shutil
from manim import *

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set media and temporary directories with fallback to local paths
if os.environ.get('DOCKER_ENV'):
    # In Docker container
    app.config['MEDIA_DIR'] = os.getenv('MEDIA_DIR', '/app/media')
    app.config['TEMP_DIR'] = os.getenv('TEMP_DIR', '/app/tmp')
else:
    # Local development
    app.config['MEDIA_DIR'] = os.path.join(os.path.dirname(__file__), 'media')
    app.config['TEMP_DIR'] = os.path.join(os.path.dirname(__file__), 'tmp')

# Ensure directories exist
os.makedirs(app.config['MEDIA_DIR'], exist_ok=True)
os.makedirs(app.config['TEMP_DIR'], exist_ok=True)
os.makedirs(os.path.join(app.config['MEDIA_DIR'], 'videos', 'scene', '720p30'), exist_ok=True)

# Ensure static/videos directory exists
os.makedirs(os.path.join(app.static_folder, 'videos'), exist_ok=True)

class MainScene(Scene):
    def construct(self):
        # Create a simple circle
        circle = Circle(radius=2, color=BLUE)
        
        # Show the circle being drawn
        self.play(Create(circle))
        self.wait(0.5)
        
        # Add a dot at the center
        dot = Dot(color=RED)
        self.play(Create(dot))
        
        # Move the dot around
        self.play(dot.animate.shift(RIGHT * 2), run_time=1)
        self.play(dot.animate.shift(UP * 2), run_time=1)
        self.play(dot.animate.shift(LEFT * 2), run_time=1)
        self.play(dot.animate.shift(DOWN * 2), run_time=1)
        
        # Cleanup
        self.play(
            FadeOut(circle),
            FadeOut(dot)
        )

def create_manim_video():
    """Create video using Manim"""
    tmp_dir = None
    try:
        # Create a temporary directory within our designated tmp directory
        tmp_dir = tempfile.mkdtemp(dir=app.config['TEMP_DIR'])
        logger.info(f"Created temporary directory: {tmp_dir}")
        
        # Write code to file
        script_path = os.path.join(tmp_dir, 'scene.py')
        with open(script_path, 'w') as f:
            f.write('''
from manim import *

class MainScene(Scene):
    def construct(self):
        # Create initial shapes
        circle = Circle(radius=2, color=BLUE)
        square = Square(side_length=4, color=GREEN)
        triangle = Triangle(color=RED).scale(2)
        
        # Position shapes
        circle.shift(LEFT * 4)
        triangle.shift(RIGHT * 4)
        
        # Create the shapes with animations
        self.play(Create(circle))
        self.wait(0.5)
        
        self.play(Create(square))
        self.wait(0.5)
        
        self.play(Create(triangle))
        self.wait(0.5)
        
        # Move shapes to center and create a composition
        self.play(
            circle.animate.shift(RIGHT * 4),
            square.animate.shift(LEFT * 0),
            triangle.animate.shift(LEFT * 4)
        )
        self.wait(1)
        
        # Rotate the composition
        self.play(
            Rotate(circle, angle=PI),
            Rotate(square, angle=PI/2),
            Rotate(triangle, angle=-PI)
        )
        self.wait(1)
        
        # Scale animation
        self.play(
            circle.animate.scale(0.5),
            square.animate.scale(0.75),
            triangle.animate.scale(0.5)
        )
        self.wait(1)
        
        # Final fade out
        self.play(
            FadeOut(circle),
            FadeOut(square),
            FadeOut(triangle)
        )
''')
        logger.info(f"Created scene file at: {script_path}")
        
        # Create media directory inside tmp_dir
        media_dir = os.path.join(tmp_dir, 'media', 'videos', 'scene', '720p30')
        os.makedirs(media_dir, exist_ok=True)
        logger.info(f"Created media directory: {media_dir}")
        
        # Set environment variables for Manim
        env = os.environ.copy()
        env['MEDIA_DIR'] = os.path.join(tmp_dir, 'media')
        
        # Run Manim command
        cmd = [
            'manim',
            '-qm',  # medium quality
            '--format=mp4',
            '--progress_bar=display',
            '--verbosity=DEBUG',
            script_path,
            'MainScene'
        ]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            env=env
        )
        
        if result.returncode != 0:
            logger.error(f"Manim command failed with return code {result.returncode}")
            logger.error(f"Stdout: {result.stdout}")
            logger.error(f"Stderr: {result.stderr}")
            raise Exception(f"Failed to generate video: {result.stderr}")
        else:
            logger.info(f"Manim stdout: {result.stdout}")

        # Find the generated video file
        output_dir = os.path.join(tmp_dir, 'media', 'videos', 'scene', '720p30')
        logger.info(f"Looking for video files in: {output_dir}")
        
        if not os.path.exists(output_dir):
            logger.error(f"Output directory does not exist: {output_dir}")
            raise Exception("Output directory not found")
            
        video_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
        logger.info(f"Found video files: {video_files}")
        
        if not video_files:
            raise Exception("No video file generated")
            
        video_path = os.path.join(output_dir, video_files[0])
        logger.info(f"Using video file: {video_path}")
        
        # Generate a unique filename for the video
        unique_filename = f"manim_video_{uuid.uuid4().hex[:8]}.mp4"
        target_path = os.path.join(app.static_folder, 'videos', unique_filename)
        logger.info(f"Moving video to: {target_path}")
        
        # Move the video file to static directory
        shutil.move(video_path, target_path)
        
        return f"videos/{unique_filename}"
        
    except Exception as e:
        logger.error(f"Error creating video: {str(e)}")
        raise
    
    finally:
        # Clean up temporary directory
        if tmp_dir and os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
                logger.info(f"Cleaned up temporary directory: {tmp_dir}")
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
        logger.info(f"Received concept: {concept}")
        
        # Check if static/videos directory exists
        if not os.path.exists(os.path.join(app.static_folder, 'videos')):
            os.makedirs(os.path.join(app.static_folder, 'videos'))
            logger.info("Created static/videos directory")
            
        # Generate video
        video_path = create_manim_video()
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
