import gradio as gr
from model import DocstringGenerator
import torch

# Initialize the model
generator = DocstringGenerator(model_name="saved_models/fine_tuned_model/",
                               device="cuda" if torch.cuda.is_available() else "cpu")


# Define the interface
def generate_output(code):
    try:
        docstring = generator.generate_docstring(code)
        comments = generator.generate_comments(code)
        explanation = generator.generate_explanation(code)
        return docstring, comments, explanation
    except Exception as e:
        return f"Error: {str(e)}", "Error: Failed to generate", "Error: Failed to generate"


interface = gr.Interface(
    fn=generate_output,
    inputs=gr.Textbox(lines=10, label="Enter your code here", placeholder="Enter Python code..."),
    outputs=[
        gr.Textbox(label="Generated Docstring"),
        gr.Textbox(label="Generated Inline Comments"),
        gr.Textbox(label="Explanation")
    ],
    title="AI Docstring Generator",
    description="Enter any Python code, and this AI will generate a docstring, inline comments, and an explanation!"
)

# Launch the app
interface.launch()
