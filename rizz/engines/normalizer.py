import gradio as gr
from tokenizers.normalizers import BertNormalizer, Sequence, Strip
import pandas as pd

class NormalizerEngine:
    def __init__(self, df):
        self.df = df
        self.normalizer = Sequence([BertNormalizer(), Strip()])

    def render(self):
        source = gr.Dropdown([], label="Source Column")
        target = gr.Textbox("normalized", label="Target Column")
        submit = gr.Button("Normalize")

        def on_change(df: pd.DataFrame):
            choices = df.columns.array
            value = choices[0]

            return gr.update(choices=choices, value=value)
        self.df.change(on_change, inputs=self.df, outputs=source)

        def on_submit(df: pd.DataFrame, s, t):
            df[t] = df[s].apply(self.normalizer.normalize_str)
            return df
        submit.click(on_submit, [self.df, source, target], self.df)
