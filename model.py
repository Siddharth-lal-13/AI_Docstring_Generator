# model.py
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from peft import PeftModel


class DocstringGenerator:
    def __init__(self, model_name="saved_models/fine_tuned_model/",
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model = PeftModel.from_pretrained(base_model, model_name).to(self.device)
        self.pipe = pipeline(
            "text2text-generation", model=self.model, tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

    def generate_docstring(self, code_snippet):
        prompt = f"Write a docstring for this Python function:\n{code_snippet}\nOutput:"
        output = self.pipe(prompt, max_new_tokens=200, do_sample=False)
        return output[0]["generated_text"].strip()

    def generate_comments(self, code_snippet):
        prompt = f"Add inline comments to this Python function:\n{code_snippet}\nOutput:"
        output = self.pipe(prompt, max_new_tokens=200, do_sample=False)
        return output[0]["generated_text"].strip()

    def generate_explanation(self, code_snippet):
        prompt = f"Explain this Python function briefly:\n{code_snippet}\nOutput:"
        output = self.pipe(prompt, max_new_tokens=200, do_sample=False)
        return output[0]["generated_text"].strip()
