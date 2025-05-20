from flask import render_template
import os
import uuid
import json
import google.generativeai as genai # Import Google AI SDK

class HTMLGenerator:
    def __init__(self, app):
        self.app = app
        self.templates_dir = os.path.join(app.root_path, 'templates')
        os.makedirs(self.templates_dir, exist_ok=True)
        self.model = None
        try:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if google_api_key:
                genai.configure(api_key=google_api_key)
                # Using gemini-2.5-flash-preview-04-17 as a robust choice.
                # Replace with "gemini-2.5-flash-preview-04-17" if specifically needed and available this way.
                self.model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17') 
                self.app.logger.info("Google AI SDK configured for HTML generation with gemini-2.5-flash-preview-04-17.")
            else:
                self.app.logger.warning("GOOGLE_API_KEY not found. AI HTML generation will fall back to demo.")
        except Exception as e:
            self.app.logger.error(f"Failed to initialize Google AI SDK for HTML generation: {e}")

    def generate_visualization(self, concept):
        """Generate an HTML visualization based on the concept"""
        template_id = str(uuid.uuid4())
        template_name = f"visualization_{template_id}.html"
        template_path = os.path.join(self.templates_dir, template_name)
        
        # Select appropriate template based on concept
        html_content = self._generate_html_via_ai(concept)

        if html_content is None:
            return None

        # Save the template
        with open(template_path, 'w') as f:
            f.write(html_content)

        return template_name
    
    def _generate_html_via_ai(self, concept: str) -> str:
        """
        Generates HTML and JavaScript for an interactive canvas visualization
        using an AI model.
        """
        if not self.model:
            self.app.logger.error("Google AI model not initialized. Skipping HTML generation.")
            return None

        self.app.logger.info(f"Attempting to generate AI HTML Canvas with Gemini for concept: {concept}")
        # Prompt might need slight adjustments for Gemini if output is not as expected.
        prompt = f"""Create a complete, runnable HTML page that provides an interactive JavaScript canvas visualization for the concept: '{concept}'.

The HTML page MUST include:
1.  A `<canvas>` element with an `id` (e.g., `id="interactiveCanvas"`).
    IMPORTANT: The canvas and its direct parent container within this HTML page MUST be styled to be responsive. For example, the canvas could have `style="width: 100%; height: auto; max-width: 100%; display: block;"` and its parent container should allow it to scale.
    The canvas drawing logic should adapt to the canvas's current width and height.
2.  JavaScript for:
    a.  Getting canvas context.
    b.  Drawing the visualization for '{concept}'. This drawing logic MUST be responsive to changes in the canvas element's size.
    c.  Relevant interactivity.
    d.  VERBATIM `sendHeightToParent()` script (as previously defined) and ensure it's called on `window.load`, `window.resize`, and critically, AFTER any function that draws or redraws the canvas content or changes its size. Make sure this function accurately calculates the total height of the body content.
        ```javascript
        function sendHeightToParent() {{ 
            let pageHeight = 0;
            if (document.body) {{ pageHeight = Math.max(document.body.scrollHeight, document.body.offsetHeight); }}
            if (document.documentElement) {{ pageHeight = Math.max(pageHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight, document.documentElement.clientHeight); }}
            const finalHeight = pageHeight; // Use measured content height
            if (window.parent && typeof window.parent.postMessage === 'function') {{ window.parent.postMessage({{ type: 'resize-iframe', height: finalHeight }}, '*'); }}
        }}
        window.addEventListener('load', sendHeightToParent);
        window.addEventListener('resize', () => {{ setupCanvasAndDraw(); sendHeightToParent(); }}); // Assuming setupCanvasAndDraw handles responsive redraw
        // IMPORTANT: Also call sendHeightToParent() after main drawing/init and after dynamic content height changes.
        // For example, if you have a main draw function like `function drawMyVisualization() {{ ... ctx.stroke(); sendHeightToParent(); }}`
        ```
3.  Complete HTML structure: `<!DOCTYPE html><html><head><style>body {{ margin: 0; padding: 0; }} canvas {{ display: block; }}</style></head><body>...</body></html>`.
4.  Visualization as primary focus. The body should not have excessive margins or fixed large sizes that would prevent the `sendHeightToParent` function from working correctly.

Output ONLY the raw HTML code. The entire response must be a single block of HTML code, starting with `<!DOCTYPE html>`. Do not include markdown fences like ```html ... ```.
"""
        try:
            # Generation config for Gemini for more controlled output (less conversational)
            generation_config = genai.types.GenerationConfig(
                candidate_count=1,
                temperature=0.3 # For code generation, lower is often better
            )
            # Safety settings can be adjusted if needed
            # safety_settings=[...]

            response = self.model.generate_content(
                prompt, 
                generation_config=generation_config
                # safety_settings=safety_settings
            )
            
            if not response.candidates or not response.candidates[0].content.parts:
                self.app.logger.error(f"Gemini HTML generation for '{concept}' returned no content or parts.")
                raise ValueError("No content from Gemini model")

            generated_html = response.candidates[0].content.parts[0].text.strip()
            
            # Clean up potential markdown fences (though we asked Gemini not to use them)
            if generated_html.startswith("```html"):
                generated_html = generated_html[len("```html"):].strip()
            elif generated_html.startswith("```"):
                 generated_html = generated_html[len("```"):].strip()
            if generated_html.endswith("```"):
                generated_html = generated_html[:-len("```")].strip()

            if not (generated_html.lower().startswith("<!doctype html>") or generated_html.lower().startswith("<html>")):
                self.app.logger.error(f"Gemini HTML generation for '{concept}' did not start with <!DOCTYPE html> or <html> after cleanup. Fallback. Received: {generated_html[:300]}...")
                raise ValueError("Gemini AI response does not look like valid HTML starting tag after cleanup.")
            
            self.app.logger.info(f"Gemini AI HTML Canvas for '{concept}' generated successfully.")
            # Log the generated HTML for debugging purposes
            self.app.logger.debug(f"--- AI Generated HTML for '{concept}' ---")
            self.app.logger.debug(generated_html) # Print HTML on its own line
            self.app.logger.debug("-------------------------------------")
            return generated_html
        except Exception as e:
            self.app.logger.error(f"Error generating HTML canvas via Gemini for '{concept}': {str(e)}")
            self.app.logger.info("Skipping HTML visualization due to Gemini AI error.")
            return None
    
    def get_template_url(self, template_name):
        """Get the URL for the template"""
        return f"/visualization/{template_name}" 