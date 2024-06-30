import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using:",device)


# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)




tts.tts_to_file(text="Hi this is Ronaldo ... i know about unysis and the inovation programme",# sample text for cloning the voice
                 speaker_wav="ronaldo.wav", 
                 language="en",
                 file_path="outputronaldo.wav")