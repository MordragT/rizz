import torch
import gradio as gr
import pandas as pd
import json
import intel_extension_for_pytorch as ipex
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer, LogitsProcessorList
from outlines.models.transformers import TransformerTokenizer
from outlines.processors import JSONLogitsProcessor

SCHEMA = """
{
    "type": "object",
    "properties": {
        "paragraphs": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": ["paragraphs"]
}
"""

class GeneratorEngine:
    def __init__(self, df, config):
        self.df = df
        self.device = config.device

        repo_id = "Qwen/Qwen2.5-3B-Instruct"

        self.tokenizer : Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(repo_id)
        self.generator : Qwen2ForCausalLM = Qwen2ForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=config.dtype,
        )

        footprint = self.generator.get_memory_footprint() / 1024 / 1024
        print(f"Generator Memory Footprint: {footprint}")

        if self.device == "xpu":
            ipex.llm.optimize(self.generator, inplace=True, device=self.device, dtype=config.dtype)

    def render(self):
        max_tokens = gr.Number(512, label="Max new Tokens")
        schema = gr.Code(SCHEMA, language='json', label="Structured Output")
        system = gr.Textbox(value="You are a paragraph segmentizer that writes text for audio synthesis", label="System Prompt")
        user = gr.Textbox(value="Write a motivational speech", label="User Prompt")
        submit = gr.Button("Generate")

        def on_submit(schema, system, user, max_tokens):
            processor = JSONLogitsProcessor(schema, TransformerTokenizer(self.tokenizer))

            self.generator.to(self.device)

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self.generator.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    logits_processor=LogitsProcessorList([processor])
                )
            outputs = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, outputs)]
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

            df = pd.DataFrame(json.loads(decoded))

            self.generator.cpu()
            if self.device == "xpu":
                torch.xpu.empty_cache()
            

            return df
        submit.click(on_submit, [schema, system, user, max_tokens], self.df)
 