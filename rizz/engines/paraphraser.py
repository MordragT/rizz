import gradio as gr
from transformers import pipeline, T5ForConditionalGeneration, T5TokenizerFast
import pandas as pd

class ParaphraserEngine:
    def __init__(self, df, config):
        self.df = df
        self.device = config.device

        repo_id = "Vamsi/T5_Paraphrase_Paws"
        self.tokenizer = T5TokenizerFast.from_pretrained(repo_id)
        self.paraphraser = T5ForConditionalGeneration.from_pretrained(repo_id, torch_dtype=config.dtype).to(config.device)

        # self.paraphraser = pipeline("text2text-generation", model=, device=config.device)
        
        footprint = self.paraphraser.get_memory_footprint() / 1024 / 1024
        print(f"Paraphraser Memory Footprint: {footprint}")

    def render(self):
        max_length =  gr.Number(256, label="Max Length")
        submit = gr.Button("Paraphrase")

        def on_submit(df: pd.DataFrame, max_length):
            items = ["paraphrase: " + item + " </s>" for item in df["body"]]
            inputs = self.tokenizer(items, return_tensors="pt", padding=True).to(self.device)
            outputs = self.paraphraser.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
            )
            df["body"] = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


            # f = lambda body : self.paraphraser(f"paraphrase: {body}", max_length=100)[0]['generated_text']
            # df["body"] = df["body"].apply(f)
            return df
        submit.click(on_submit, [self.df, max_length], self.df)
 