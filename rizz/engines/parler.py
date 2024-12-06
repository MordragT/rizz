import io
import torch
import gradio as gr
import pandas as pd
from transformers import AutoTokenizer, set_seed
from torchaudio.io import StreamWriter
from parler_tts import ParlerTTSForConditionalGeneration

class ParlerEngine:
    def __init__(self, df, audio, config):
        self.df = df
        self.audio = audio
        self.device = config.device

        repo_id = "parler-tts/parler-tts-mini-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            repo_id,
            torch_dtype=config.dtype,
        ).to(config.device)
        self.sampling_rate = self.model.audio_encoder.config.sampling_rate

        
        footprint = self.model.get_memory_footprint() / 1024 / 1024
        print(f"Parler Memory Footprint: {footprint}")

        # TODO maybe works on Leon's machine
        # # compile the forward pass
        # self.model.generation_config.cache_implementation = "static"
        # self.model.forward = torch.compile(self.model.forward, mode="default")

        # # warmup
        # inputs = self.tokenizer(
        #     "Warmup",
        #     return_tensors="pt",
        #     padding="max_length",
        #     max_length=256
        # ).to(self.config.device)

        # _ = self.model.generate(
        #     input_ids=inputs.input_ids,
        #     attention_mask=inputs.attention_mask,
        #     prompt_input_ids=inputs.input_ids,
        #     prompt_attention_mask=inputs.attention_mask,
        # )


    def render(self):
        speaker_prompt = gr.Textbox("Jon's voice delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.", label="Speaker Prompt")
        max_length = gr.Number(100, label="Max Length")
        batch_size = gr.Number(1, label="Batch Size")
        submit = gr.Button("Synthesize")

        def on_submit(df: pd.DataFrame, speaker_prompt, max_length, batch_size):
            speaker_inputs = self.tokenizer(speaker_prompt, return_tensors="pt").to(self.device)

            audio = io.FileIO("audio.wav", "w+")
            stream = StreamWriter(audio, "wav")
            stream.add_audio_stream(self.sampling_rate, 1)
            durations = []

            text_inputs = df["body"].to_list()

            with stream.open() as writer:
                for i in range(0, len(text_inputs), batch_size):
                    batch = text_inputs[i:i + batch_size]

                    # TODO set max_length as minimum of the max length of the actual batch and the user supplied max_length

                    prompt_inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                    ).to(self.device)

                    input_ids = speaker_inputs.input_ids.expand(prompt_inputs.input_ids.size(0), -1)
                    attention_mask = speaker_inputs.attention_mask.expand(prompt_inputs.attention_mask.size(0), -1)

                    set_seed(0)
                    with torch.inference_mode():
                        outputs = self.model.generate(
                            prompt_input_ids=prompt_inputs.input_ids,
                            prompt_attention_mask=prompt_inputs.attention_mask,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict_in_generate=True,
                        )
                    
                    sequence_lengths = outputs.audios_length
                    for i, length in enumerate(sequence_lengths):
                        chunk = outputs.sequences[i, :length].unsqueeze(1).float().cpu()
                        writer.write_audio_chunk(0, chunk)

                    durations = durations + [length / self.sampling_rate for length in sequence_lengths]

            df["duration"] = durations
            return df, gr.update(value="audio.wav")

        submit.click(on_submit, [self.df, speaker_prompt, max_length, batch_size], [self.df, self.audio])

