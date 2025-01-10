import io
import torch
import gradio as gr
import pandas as pd
# import intel_extension_for_pytorch as ipex
from transformers import AutoTokenizer, set_seed
from torchaudio.io import StreamWriter
from parler_tts import ParlerTTSForConditionalGeneration

class ParlerEngine:
    def __init__(self, df, source, audio, config):
        self.df = df
        self.source = source
        self.audio = audio
        self.device = config.device
        self.dtype = config.dtype

        # repo_id = "parler-tts/parler-mini-v1-jenny"
        repo_id = "parler-tts/parler-tts-mini-v1.1"
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            repo_id,
            torch_dtype=config.dtype,
            # attn_implementation="sdpa",
            attn_implementation="eager",
        )
        self.sampling_rate = self.model.audio_encoder.config.sampling_rate

        footprint = self.model.get_memory_footprint() / 1024 / 1024
        print(f"Parler Memory Footprint: {footprint}")

        # print(torch._dynamo.list_backends())

        # # # TODO maybe works on Leon's machine
        # # # compile the forward pass
        # self.model.generation_config.cache_implementation = "static"
        # self.model.to(self.device)
        # self.model.forward = torch.compile(self.model.forward)

        # # warmup
        # inputs = self.tokenizer("Warmup", return_tensors="pt").to(self.device)

        # _ = self.model.generate(
        #     input_ids=inputs.input_ids,
        #     attention_mask=inputs.attention_mask,
        #     prompt_input_ids=inputs.input_ids,
        #     prompt_attention_mask=inputs.attention_mask,
        # )
        # self.model.cpu()


    def render(self):
        speaker_prompt = gr.Textbox("Gary's voice delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.", label="Speaker Prompt")
        do_sample = gr.Checkbox(True, label="Enable Sampling")
        temperature = gr.Slider(value=1.0, minimum=0.0, maximum=2.0, step=0.1, label="Temperature")
        max_length = gr.Number(96, label="Max Length")
        batch_size = gr.Number(1, label="Batch Size")
        submit = gr.Button("Synthesize")

        def on_submit(
                df: pd.DataFrame,
                s,
                speaker_prompt,
                do_sample,
                temperature,
                max_length,
                batch_size,
                progress=gr.Progress()
            ):
            self.model.to(self.device)
            self.model.eval()

            # if self.device == "xpu":
            #     self.model = ipex.optimize(self.model, dtype=self.dtype)

            speaker_inputs = self.tokenizer(speaker_prompt, return_tensors="pt").to(self.device)

            audio = io.FileIO("audio.wav", "w+")
            stream = StreamWriter(audio, "wav")
            stream.add_audio_stream(self.sampling_rate, 1)
            durations = []

            text_inputs = df[s].to_list()

            with stream.open() as writer:
                for i in progress.tqdm(range(0, len(text_inputs), batch_size)):
                    batch = text_inputs[i:i + batch_size]

                    prompt_inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        max_length=max_length,
                        padding="longest",
                        truncation=True,
                    ).to(self.device)

                    print(prompt_inputs)

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
                            do_sample=do_sample,
                            temperature=temperature,
                        )
                    
                    sequence_lengths = outputs.audios_length
                    for i, length in enumerate(sequence_lengths):
                        chunk = outputs.sequences[i, :length].unsqueeze(1).float().cpu()
                        writer.write_audio_chunk(0, chunk)

                    durations = durations + [length / self.sampling_rate for length in sequence_lengths]

            df["duration"] = durations

            self.model.cpu()
            if self.device == "xpu":
                # TODO synchronize first ?
                torch.xpu.empty_cache()

            return df, gr.update(value="audio.wav")

        submit.click(on_submit, [
            self.df,
            self.source,
            speaker_prompt,
            do_sample,
            temperature,
            max_length,
            batch_size,
        ], [self.df, self.audio])

