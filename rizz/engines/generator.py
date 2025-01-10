import torch
import gradio as gr
import pandas as pd
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, BitsAndBytesConfig
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
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=config.dtype,
        #     bnb_4bit_quant_type="nf4",
        # )

        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.generator = AutoModelForCausalLM.from_pretrained(
            repo_id,
            # device_map="auto",
            torch_dtype=config.dtype,
            # quantization_config=quantization_config,
        )

        # # compilation
        # self.generator.generation_config.cache_implementation = "static"
        # self.generator.forward = torch.compile(self.generator.forward, mode="default", backend="inductor")

        # # warmup
        # print("Generator Warmup Start")
        # start = time.time()
        # self.generate(SCHEMA, "You write one paragraph", "this is a test", 512)
        # print(f"Generator Warmup Time: {time.time() - start}")

        footprint = self.generator.get_memory_footprint() / 1024 / 1024
        print(f"Generator Memory Footprint: {footprint}")

        # if self.device == "xpu":
        #     try:
        #         import intel_extension_for_pytorch as ipex
        #         ipex.llm.optimize(self.generator, inplace=True, device=self.device, dtype=config.dtype)
        #     except:
        #         pass
    
    def generate(self, schema, system, user, max_tokens):
        processor = JSONLogitsProcessor(schema, TransformerTokenizer(self.tokenizer))

        self.generator.to(self.device)
        self.generator.eval()

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

        self.generator.cpu()
        if self.device == "xpu":
            torch.xpu.empty_cache()

        return decoded

    def render(self):
        max_tokens = gr.Number(512, label="Max new Tokens")
        schema = gr.Code(SCHEMA, language='json', label="Structured Output")
        system = gr.Textbox(value="You are a paragraph segmentizer that writes text for audio synthesis", label="System Prompt")
        user = gr.Textbox(value="Write a motivational speech", label="User Prompt")
        submit = gr.Button("Generate")

        def on_submit(schema, system, user, max_tokens):
            print("Generator Processing Start")

            start = time.time()
            decoded = self.generate(schema, system, user, max_tokens)
            df = pd.DataFrame(json.loads(decoded))

            print(f"Generator Processing Time: {time.time() - start}")
            return df
        submit.click(on_submit, [schema, system, user, max_tokens], self.df)
 