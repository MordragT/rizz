import torch
import gradio as gr
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import pandas as pd

class ParaphraserEngine:
    def __init__(self, df, config):
        self.df = df
        self.device = config.device

        repo_id = "Vamsi/T5_Paraphrase_Paws"
        self.tokenizer = T5TokenizerFast.from_pretrained(repo_id, legacy=False)
        self.paraphraser = T5ForConditionalGeneration.from_pretrained(repo_id, torch_dtype=config.dtype)
        
        footprint = self.paraphraser.get_memory_footprint() / 1024 / 1024
        print(f"Paraphraser Memory Footprint: {footprint}")

    def render(self):
        source = gr.Dropdown([], label="Source Column")
        target = gr.Textbox("paraphrased", label="Target Column")
        max_length =  gr.Number(256, label="Max Length")
        submit = gr.Button("Paraphrase")

        def on_change(df: pd.DataFrame):
            choices = df.columns.array
            value = choices[0]

            return gr.update(choices=choices, value=value)
        self.df.change(on_change, inputs=self.df, outputs=source)

        def on_submit(df: pd.DataFrame, s, t, max_length):
            self.paraphraser.to(self.device)

            items = ["paraphrase: " + item + " </s>" for item in df[s]]
            inputs = self.tokenizer(items, return_tensors="pt", padding=True).to(self.device)
            
            with torch.inference_mode():
                outputs = self.paraphraser.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=True,
                )
            df[t] = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


            self.paraphraser.cpu()
            if self.device == "xpu":
                torch.xpu.empty_cache()

            return df
        submit.click(on_submit, [self.df, source, target, max_length], self.df)
 