import gradio as gr

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips

class ConcatenatorEngine:
    def __init__(self, config):
        self.resources = config.resources

    def render(self):
        in_videos = gr.Gallery(format="mp4", label="Input Videos")
        width = gr.Number(1080, label="Target Video Width")
        height = gr.Number(1920, label="Target Video Height")
        filename = gr.Textbox("output.mp4", label="Filename")
        submit = gr.Button("Concatenate Videos")
        out_video = gr.Video(interactive=False, label="Concatenated Video")

        def on_submit(in_videos, width, height, filename):
            clips = [VideoFileClip(v, target_resolution=(width, height)) for v, _ in in_videos]
            out_video = concatenate_videoclips(clips)
            path = self.resources / "videos" / filename
            out_video.write_videofile(
                path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile="audio.mp4",
                remove_temp=True,
            )

            return gr.update(value=path)
        
        submit.click(on_submit, [in_videos, width, height, filename], out_video)