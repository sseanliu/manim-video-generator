# Manim Video Generator

This application allows you to generate mathematical animations using natural language descriptions. It uses OpenAI's GPT model to convert your descriptions into Manim code, which is then used to create beautiful mathematical animations.

## Prerequisites

- Python 3.7+
- FFmpeg
- LaTeX distribution (for rendering mathematical equations)
- OpenAI API key

## Installation

1. Clone this repository:
```bash
git clone https://github.com/rohitg00/manim-video-generator.git
cd manim-video-generator
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install Manim system dependencies:
```bash
# For macOS
brew install cairo ffmpeg
```

4. Set up your environment variables:
```bash
cp .env.example .env
```
Then edit `.env` and add your OpenAI API key.

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Enter a description of the mathematical animation you want to create in the text area

4. Click "Generate Video" and wait for your animation to be created

## Examples

Here are some example prompts you can try:

- "Create an animation showing a circle transforming into a square"
- "Show the unit circle with sine and cosine waves being traced out"
- "Demonstrate the Pythagorean theorem with animated squares on triangle sides"

## How it Works

1. The application takes your natural language description and sends it to OpenAI's GPT model
2. GPT generates the appropriate Manim code for your animation
3. The code is executed using Manim to create an MP4 video
4. The video is served back to your browser

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
