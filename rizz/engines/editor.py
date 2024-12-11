import gradio as gr
import pandas as pd
import json

class EditorEngine:
    def __init__(self, df):
        self.df = df

    def render(self):
        content = gr.Code(language="json", label="Content")
        with gr.Row():
            copy = gr.Button("Copy from Dataframe")
            submit = gr.Button("Save")

        def on_copy(df: pd.DataFrame):
            return gr.update(value=df.to_json(indent=2))
        copy.click(on_copy, inputs=self.df, outputs=content)

        def on_submit(content):
            df = pd.DataFrame.from_dict(json.loads(content))
            return df
        submit.click(on_submit, content, self.df)
