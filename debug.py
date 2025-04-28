# debug_model.py
from model import DocstringGenerator

# Instantiate the class to trigger the debug print
generator = DocstringGenerator(model_name="google/flan-t5-base", device="cpu")  # Use CPU for debugging to avoid CUDA issues