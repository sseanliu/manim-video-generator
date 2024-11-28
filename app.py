from flask import Flask, render_template, request, jsonify
import os
import tempfile
import subprocess
import logging
import uuid
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Ensure static/videos directory exists
os.makedirs(os.path.join(app.static_folder, 'videos'), exist_ok=True)

def generate_manim_code(concept):
    """Generate specialized Manim code based on the concept type"""
    return '''from manim import *

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
'''

def create_manim_video(manim_code):
    """Create video using Manim"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write code to file
            script_path = os.path.join(temp_dir, 'scene.py')
            with open(script_path, 'w') as f:
                f.write(manim_code)
            
            # Run Manim command
            cmd = [
                'manim',
                '-pql',
                script_path,
                'MainScene'
            ]
            
            result = subprocess.run(
                cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Manim error: {result.stderr}")
                return None
            
            # Find the video file
            video_file = None
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.mp4'):
                        video_file = os.path.join(root, file)
                        break
                if video_file:
                    break
            
            if not video_file:
                logger.error("No video file found")
                return None
            
            # Copy to static directory with unique name
            output_filename = f'animation_{uuid.uuid4()}.mp4'
            output_path = os.path.join(app.static_folder, 'videos', output_filename)
            shutil.copy2(video_file, output_path)
            
            return f'/static/videos/{output_filename}'
            
    except Exception as e:
        logger.error(f"Error creating video: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_video():
    try:
        concept = request.form.get('concept', '')
        if not concept:
            return jsonify({'error': 'No concept provided'}), 400

        # Generate video
        video_path = create_manim_video(generate_manim_code(concept))
        if not video_path:
            return jsonify({'error': 'Failed to generate video'}), 500

        return jsonify({'video_url': video_path})
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5003)
