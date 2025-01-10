import torch
import gradio as gr
import pandas as pd
import json
import time
# import intel_extension_for_pytorch as ipex
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, LogitsProcessorList
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

class OvGeneratorEngine:
    def __init__(self, df, config):
        self.df = df

        repo_id = "AIFunOver/Qwen2.5-1.5B-Instruct-openvino-fp16"

        self.tokenizer : AutoTokenizer = AutoTokenizer.from_pretrained(repo_id)
        # disable caching of gpu
        # ov_config = {"CACHE_DIR": "", "INFERENCE_PRECISION_HINT": "f16"}
        ov_config = {"CACHE_DIR": ""}
        self.generator : OVModelForCausalLM = OVModelForCausalLM.from_pretrained(
            repo_id,
            # compile=False,
            device=config.ov_device,
            ov_config=ov_config,
        )
    def generate(self, schema, system, user, max_tokens):

        processor = JSONLogitsProcessor(schema, TransformerTokenizer(self.tokenizer))

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer([text], return_tensors="pt")

        outputs = self.generator.generate(
            **inputs,
            max_new_tokens=max_tokens,
            # max_length=max_tokens,
            do_sample=True,
            logits_processor=LogitsProcessorList([processor])
        )
        # preds = torch.nn.functional.softmax(outputs["logits"], dim=-1).argmax(dim=-1)
        # print(outputs)
        outputs = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, outputs)]
        # print(outputs)
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        print(decoded)

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
 