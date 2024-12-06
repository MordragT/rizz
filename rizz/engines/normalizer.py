import gradio as gr
from tokenizers.normalizers import BertNormalizer, Sequence, Strip
import pandas as pd

class NormalizerEngine:
    def __init__(self, df):
        self.df = df
        self.normalizer = Sequence([BertNormalizer(), Strip()])

    def render(self):
        submit = gr.Button("Normalize")

        def on_submit(df: pd.DataFrame):
            df["body"] = df["body"].apply(self.normalizer.normalize_str)
            return df
        submit.click(on_submit, self.df, self.df)
