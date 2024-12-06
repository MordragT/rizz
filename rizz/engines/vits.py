import io
import torch
import gradio as gr
import pandas as pd
from torchaudio.io import StreamWriter
from transformers import VitsTokenizer, VitsModel, set_seed

class VitsEngine:
    def __init__(self, df, audio, config):
        self.df = df
        self.audio = audio
        self.device = config.device

        repo_id = "facebook/mms-tts-eng"
        self.tokenizer = VitsTokenizer.from_pretrained(repo_id)
        self.model = VitsModel.from_pretrained(repo_id,).to(config.device)
        self.sampling_rate = self.model.config.sampling_rate
        
        footprint = self.model.get_memory_footprint() / 1024 / 1024
        print(f"Vits Memory Footprint: {footprint}")

    def render(self):
        max_length = gr.Number(200, label="Max Length")
        batch_size = gr.Number(8, label="Batch Size")
        submit = gr.Button("Synthesize")

        def on_submit(df: pd.DataFrame, max_length, batch_size):
            audio = io.FileIO("audio.wav", "w+")
            stream = StreamWriter(audio, "wav")
            stream.add_audio_stream(self.sampling_rate, 1)
            durations = []

            text_inputs = df["body"].to_list()

            with stream.open() as writer:
                for i in range(0, len(text_inputs), batch_size):
                    batch = text_inputs[i:i + batch_size]

                    # TODO set max_length as minimum of the max length of the actual batch and the user supplied max_length

                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                    ).to(self.device)

                    set_seed(0)
                    with torch.inference_mode():
                        outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                    
                    sequence_lengths = outputs.sequence_lengths.cpu()
                    for i, length in enumerate(sequence_lengths):
                        chunk = outputs.waveform[i, :length].unsqueeze(1).float().cpu()
                        writer.write_audio_chunk(0, chunk)

                    durations = durations + [length.item() / self.sampling_rate for length in sequence_lengths]

            df["duration"] = durations
            return df, gr.update(value="audio.wav")

        submit.click(on_submit, [self.df, max_length, batch_size], [self.df, self.audio])
