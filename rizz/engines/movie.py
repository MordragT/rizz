import gradio as gr
import pandas as pd

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.fx.Loop import Loop
from moviepy.video.fx.MakeLoopable import MakeLoopable
from moviepy.video.fx.HeadBlur import HeadBlur
from moviepy.audio.io.AudioFileClip import AudioFileClip

class MovieEngine:
    def __init__(self, df, source, audio, video, config):
        self.df = df
        self.source = source
        self.audio = audio
        self.video = video
        self.resources = config.resources

    def render(self):
        with gr.Row():
            with gr.Column():
                text_width = gr.Number(900, label="Text Width")
                text_height = gr.Number(1600, label="Text Height")
                text_margin_x = gr.Number(90, label="Horizontal Text Margin")
                text_margin_y = gr.Number(160, label="Vertical Text Margin")
                text_align = gr.Dropdown(["center", "left", "right"], value="center", allow_custom_value=False, label="Text Alignment")
                font = gr.FileExplorer(file_count="single", root_dir=self.resources / "fonts", value="Thernaly.ttf", label="Font")
                font_color = gr.ColorPicker(value="#000", label="Font Color")
                font_size = gr.Number(64, label="Font Size")
                blur = gr.Checkbox(label="Blur")
                blur_x = gr.Number(540, label="Blur Horizontal Position")
                blur_y = gr.Number(960, label="Blur Vertical Position")
                blur_radius = gr.Number(480, label="Blur Radius")
                blur_intensity = gr.Number(4.0, label="Blur Intensity")

            with gr.Column():
                video_width = gr.Number(1080, label="Target Video Width")
                video_height = gr.Number(1920, label="Target Video Height")
                video_select = gr.FileExplorer(file_count="single", root_dir=self.resources / "videos", value="default.mp4", label="Video Selection")
                in_video = gr.Video(value=lambda f : self.resources / "videos" / f, inputs=video_select, label="Input Video")

        submit = gr.Button("Render Video")

        def on_submit(
                df: pd.DataFrame,
                s,
                in_video,
                video_width,
                video_height,
                text_width,
                text_height,
                text_margin_x,
                text_margin_y,
                text_align,
                font,
                font_color,
                font_size,
                blur,
                blur_x,
                blur_y,
                blur_radius,
                blur_intensity,
            ):
            video_duration = df["duration"].sum()

            progress = 0
            text_clips = []
            for text, duration in zip(df[s].values, df["duration"].values):
                text_clip = TextClip(
                    text=text,
                    text_align=text_align,
                    font=self.resources / "fonts" / font,
                    font_size=font_size,
                    color=font_color,
                    # bg_color='#fff',
                    method="caption",
                    size=(text_width, text_height),
                    margin=(text_margin_x, text_margin_y),
                    duration=duration,
                ).with_start(progress)
                text_clips.append(text_clip)
                progress += duration

            effects = [Loop(duration=video_duration)] # MakeLoopable(0.1),
            if blur:
                effects.append(HeadBlur(fx=lambda _ : blur_x, fy=lambda _ : blur_y, radius=blur_radius, intensity=blur_intensity))
        
            video_clip = VideoFileClip(in_video, target_resolution=(video_width, video_height)).with_effects(effects)

            audio_clip = AudioFileClip("audio.wav")
            video = CompositeVideoClip([video_clip] + text_clips).with_audio(audio_clip)

            video_path = "video.mp4"
            video.write_videofile(
                video_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile="audio.mp4",
                remove_temp=True,
            )

            return gr.update(value=video_path)
        
        submit.click(on_submit, [
            self.df,
            self.source,
            in_video,
            video_width,
            video_height,
            text_width,
            text_height,
            text_margin_x,
            text_margin_y,
            text_align,
            font,
            font_color,
            font_size,
            blur,
            blur_x,
            blur_y,
            blur_radius,
            blur_intensity,
        ], self.video)