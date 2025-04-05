import torch
import gradio as gr
import pandas as pd

# try:
#     import intel_extension_for_pytorch as ipex
# except:
#     pass

from .engines import EditorEngine, BarkEngine, ConcatenatorEngine, GeneratorEngine, RedditEngine, NormalizerEngine, ParaphraserEngine, TikTokEngine, MovieEngine, VitsEngine, ParlerEngine, YoutubeEngine

class RizzApp:
    def __init__(self, config):
        self.config = config

    def launch(self):
        df = gr.Dataframe(wrap=False, label="Output")
        source = gr.Dropdown([], label="Source Column")
        audio = gr.Audio(type="filepath")
        video = gr.Video(interactive=False, height=640)

        # generator = OvGeneratorEngine(df, self.config)
        generator = GeneratorEngine(df, self.config)
        reddit = RedditEngine(df)
        normalizer = NormalizerEngine(df)
        paraphraser = ParaphraserEngine(df, self.config)
        editor = EditorEngine(df)
        parler = ParlerEngine(df, source, audio, self.config)
        bark = BarkEngine(df, source, audio, self.config)
        vits = VitsEngine(df, source, audio, self.config)
        movie = MovieEngine(df, source, audio, video, self.config)
        concatenator = ConcatenatorEngine(self.config)
        youtube = YoutubeEngine()
        tiktok = TikTokEngine()

        def on_tick():
            if self.config.device == "xpu":
                free, total = torch.xpu.mem_get_info()
                return f"Free Memory: {free}\nTotal Memory: {total}"
            elif self.config.device == "cuda":
                free, total = torch.cuda.mem_get_info()
                return f"Free Memory: {free}\nTotal Memory: {total}"
            else:
                return "No memory information"

        with gr.Blocks() as demo:
            gr.Markdown("# Rizz")

            gr.Textbox(on_tick, every=1)

            with gr.Tab("Generator"):
                generator.render()

            with gr.Tab("Reddit"):
                reddit.render()

            with gr.Tab("Normalizer"):
                normalizer.render()

            with gr.Tab("Paraphraser"):
                paraphraser.render()

            with gr.Tab("Editor"):
                editor.render()

            df.change(on_change, inputs=df, outputs=source)
            df.render()
            source.render()

            with gr.Tab("Parler TTS"):
                parler.render()

            with gr.Tab("Bark TTS"):
                bark.render()

            with gr.Tab("Vits TTS"):
                vits.render()


            audio.render()

            with gr.Tab("Movie Maker"):
                movie.render()

            with gr.Tab("Concatenator"):
                concatenator.render()

            video.render()

            with gr.Tab("Youtube"):
                youtube.render()

            with gr.Tab("TikTok"):
                tiktok.render()

        demo.launch()


def on_change(df: pd.DataFrame):
    choices = df.columns.array
    value = choices[0]

    return gr.update(choices=choices, value=value)
