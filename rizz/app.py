import gradio as gr

from .engines import RedditEngine, NormalizerEngine, ParaphraserEngine, TikTokEngine, MovieEngine, VitsEngine, ParlerEngine, YoutubeEngine

class RizzApp:
    def __init__(self, config):
        self.config = config

    def launch(self):
        df = gr.Dataframe(headers=["author", "score", "body"], wrap=False, label="Output")
        audio = gr.Audio(type="filepath")
        video = gr.Video(interactive=False, height=640)

        reddit = RedditEngine(df)
        normalizer = NormalizerEngine(df)
        paraphraser = ParaphraserEngine(df, self.config)
        parler = ParlerEngine(df, audio, self.config)
        vits = VitsEngine(df, audio, self.config)
        movie = MovieEngine(df, audio, video, self.config)
        youtube = YoutubeEngine()
        tiktok = TikTokEngine()

        with gr.Blocks() as demo:
            gr.Markdown("# Rizz")

            with gr.Tab("Reddit"):
                reddit.render()

            with gr.Tab("Normalizer"):
                normalizer.render()

            with gr.Tab("Paraphraser"):
                paraphraser.render()

            df.render()

            with gr.Tab("Parler TTS"):
                parler.render()

            with gr.Tab("Vits TTS"):
                vits.render()

            
            audio.render()

            with gr.Tab("Movie Maker"):
                movie.render()

            video.render()

            with gr.Tab("Youtube"):
                youtube.render()

            with gr.Tab("TikTok"):
                tiktok.render()

        demo.launch()
