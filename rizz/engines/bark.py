import io
import torch
import gradio as gr
import pandas as pd
from torchaudio.io import StreamWriter
from transformers import BarkProcessor, BarkModel, set_seed

class BarkEngine:
    def __init__(self, df, source, audio, config):
        self.df = df
        self.source = source
        self.audio = audio
        self.device = config.device

        repo_id = "suno/bark-small"
        self.processor = BarkProcessor.from_pretrained(repo_id)
        self.model = BarkModel.from_pretrained(repo_id, torch_dtype=config.dtype)
        # self.model = self.model.to_bettertransformer()
        self.sampling_rate = self.model.generation_config.sample_rate
        
        footprint = self.model.get_memory_footprint() / 1024 / 1024
        print(f"Bark Memory Footprint: {footprint}")

        # requires cuda
        # self.model.to(self.device)
        # self.model.enable_cpu_offload()

    def render(self):
        voices = [f"v2/{l}_speaker_{i}" for l in ["de", "en"] for i in range(10)]

        voice = gr.Dropdown(voices, value="v2/en_speaker_6", label="Voice Preset")
        do_sample = gr.Checkbox(True, label="Enable Sampling")
        fine_temperature = gr.Slider(value=0.4, minimum=0.0, maximum=2.0, step=0.1, label="Fine Temperature")
        coarse_temperature = gr.Slider(value=0.5, minimum=0.0, maximum=2.0, step=0.1, label="Coarse Temperature")
        max_length = gr.Number(256, label="Max Length")
        batch_size = gr.Number(1, label="Batch Size")
        submit = gr.Button("Synthesize")

        def on_submit(
                df: pd.DataFrame,
                s,
                voice,
                do_sample,
                fine_temp,
                coarse_temp,
                max_length,
                batch_size,
                progress=gr.Progress()
            ):
            self.model.to(self.device)

            audio = io.FileIO("audio.wav", "w+")
            stream = StreamWriter(audio, "wav")
            stream.add_audio_stream(self.sampling_rate, 1)
            durations = []

            text_inputs = df[s].to_list()

            with stream.open() as writer:
                for i in progress.tqdm(range(0, len(text_inputs), batch_size)):
                    batch = text_inputs[i:i + batch_size]

                    inputs = self.processor(
                        batch,
                        return_tensors="pt",
                        max_length=max_length,
                        voice_preset=voice,
                    ).to(self.device)

                    set_seed(0)
                    with torch.inference_mode():
                        waveform, sequence_lengths = self.model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            history_prompt=inputs["history_prompt"].to(self.device),
                            return_output_lengths=True,
                            do_sample=do_sample,
                            fine_temperature=fine_temp,
                            coarse_temperature=coarse_temp,
                        )
                    
                    for i, length in enumerate(sequence_lengths):
                        chunk = waveform[i, :length].unsqueeze(1).float().cpu()
                        writer.write_audio_chunk(0, chunk)

                    durations = durations + [length / self.sampling_rate for length in sequence_lengths]

            df["duration"] = durations

            self.model.cpu()
            if self.device == "xpu":
                torch.xpu.empty_cache()

            return df, gr.update(value="audio.wav")

        submit.click(on_submit, [
            self.df,
            self.source,
            voice,
            do_sample,
            fine_temperature,
            coarse_temperature,
            max_length,
            batch_size
        ], [self.df, self.audio])
