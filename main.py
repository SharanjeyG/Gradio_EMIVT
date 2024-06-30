import gradio as gr
import os
import threading
import time
import wave
from datetime import datetime
# import boto3
import pyaudio
from botocore.exceptions import NoCredentialsError
from google.cloud import speech
from google.cloud import texttospeech as tts
from google.cloud import translate_v2 as translate
#from transformers import pipeline
import google.generativeai as genai
import html
import torch
from TTS.api import TTS
import requests

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "tribal-booth-414608-dbc61449795a.json"
RATE = 44100 

genai.configure(api_key="AIzaSyB-f-KRov7a3KxB7Dqon3kpoTWGwEUld9E")
model = genai.GenerativeModel('gemini-pro')

class MicrophoneRecorder:
    #emotions = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
    def __init__(self):
        self.stream = None
        self.p = pyaudio.PyAudio()
        self.frames = []
        # self.s3_client = boto3.client('s3')
        self.is_recording = False  # Flag to indicate if recording is in progress

    def start_recording(self):
        self.is_recording = True
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=1024)
        self.frames = []
        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    def _record(self):
        while self.is_recording:
            data = self.stream.read(1024)
            self.frames.append(data)

    def stop_recording(self, fromlang, tolang):
        self.is_recording = False
        self.stream.stop_stream()
        self.stream.close()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"input_{timestamp}.wav"

        # Save the recording locally first
        file_path = self.save_recording(filename)

        progress = gr.Progress()
        progress(0, desc="Starting")
        time.sleep(1)
        progress(0.05)

        fromlangu = language_mapping.get(fromlang)
        tolangu = language_mapping.get(tolang)
        # use_local_file = False
        # s3_input_path = f"{filename}"

        # Try to upload to S3
        # try:
        #     self.s3_client.upload_file(filename, S3_BUCKET_NAME, s3_input_path)
        #     print(f"File {filename} uploaded to {s3_input_path} successfully.")
        # except Exception as e:
        #     print(f"Failed to upload {filename} to S3: {e}. Using local file for further processing.")
        #     use_local_file = True

        # if not use_local_file:
            # If upload succeeds, remove the local file
            # os.remove(filename)

        # Process the recording, using S3 path if upload succeeded, or local path if failed
        # file_path_for_processing = s3_input_path if not use_local_file else filename
        transcribed_text = self.transcribe_audio(file_path, fromlangu) #, use_local_file
        translated_text= self.trans_text(transcribed_text, tolangu)
        # audio_emotion = self.audio_emo(file_path_for_processing)
        # emotion = self.emotionlabel(emotext)
        # audio_url = self.tts(translated_text, tolang, emotion)

        # return transcribed_text, translated_text, emotion, audio_url, audio_emotion
        return transcribed_text, translated_text

    def save_recording(self, filename):
        directory = "audios/inputs"
        os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
        file_path = os.path.join(directory, filename)

        wf = wave.open(file_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        return file_path
    
    def audiofile(self, audiof, fromlang, tolang):
        print(type(audiof))
        tolangu = language_mapping.get(tolang)
        fromlangu = language_mapping.get(fromlang)
        print(fromlangu)

        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=audiof)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=fromlangu,
        )
        try:
            response = client.recognize(config=config, audio=audio)
            transcribed_text = ''.join([result.alternatives[0].transcript for result in response.results])
            print("HI", transcribed_text)
        except Exception as e:
            print(f"Error during transcription: {e}")
            transcribed_text = ""

        # Delete the local file if it was downloaded from S3
        # if not use_local_file:
        #     os.remove(local_file_path)

        print("Transcribed Text:", transcribed_text)
        translated_text = self.trans_text(transcribed_text, tolangu)
        return transcribed_text, translated_text


    def transcribe_audio(self, file_path, fromlang):#, use_local_file=False
        # if use_local_file:
        #     local_file_path = file_path
        # else:
            # Assuming file_path is the S3 object key
        # filename = file_path.split('/')[-1]
        # local_file_path = f"./{filename}"
        print(file_path)
            # try:
            #     self.s3_client.download_file(S3_BUCKET_NAME, file_path, local_file_path)
            #     print(f"Downloaded {file_path} from S3 for processing.")
            # except Exception as e:
            #     print(f"Failed to download {file_path} from S3: {e}")
            #     return ""

        # File transcription process remains the same
        with open(file_path, 'rb') as audio_file:
            content = audio_file.read()
            print("Type: ",type(content))

        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=fromlang,
        )
        try:
            response = client.recognize(config=config, audio=audio)
            transcribed_text = ''.join([result.alternatives[0].transcript for result in response.results])
        except Exception as e:
            print(f"Error during transcription: {e}")
            transcribed_text = ""

        # Delete the local file if it was downloaded from S3
        # if not use_local_file:
        #     os.remove(local_file_path)

        print("Transcribed Text:", transcribed_text)
        return transcribed_text
    
    def trans_text(self, text, tolang):

        translate_client = translate.Client()

    # Translates the text into the target language
        translated_text = translate_client.translate(text, target_language=tolang)
        trans_text = html.unescape(translated_text['translatedText'])

        # emotion_text = translate_client.translate(text, target_language='en-US')
        # emo_text = html.unescape(emotion_text['translatedText'])
        
        print("translated text:",translated_text)

        return trans_text #, emo_text
    
recorder = MicrophoneRecorder()

def context(text,tolang):
    to_lang = language_mapping.get(tolang)

    """
    Extracts meaning from the provided text and attempts to generate a sentence in the target language.

    Args:
        text: The text to understand and potentially translate/rephrase.

    Returns:
        A string containing the extracted meaning or an informative message if no content is generated.
    """

    prompts = [
        f"For the sentence: {text}, you want to understand the meaning of the sentence and generate a similar sentence in {to_lang} note that the complete response should be in  {to_lang}?"
    ]

    for prompt in prompts:
        response = model.generate_content(prompt)

        # Check if the response contains any candidates
        if response._result.candidates:
            # Extract the meaning from the response (assuming first candidate)
            meaning = response._result.candidates[0].content.parts[0].text
            return meaning
        else:
            meaning = "Model did not generate any response for this sentence."

def voiceclone(audio, text, tolang):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using:",device)

    print(tolang)

    ttolang = maping.get(tolang)
    print(ttolang)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    op_path = f"/input_audios/output{timestamp}.wav"

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    tts.tts_to_file(text=text,# sample text for cloning the voice
                 speaker_wav=audio,
                 language=ttolang,
                 file_path=op_path)

    return op_path

def audio_emo(file_emo):
    api_url = "http://127.0.0.1:8000/classify-emotion"
    files = {'file': open(file_emo, 'rb')}
    try:
        response = requests.post(api_url, files=files)
                    
        if response.status_code == 200:
            emotion_label = response.json().get('emotion')
            emot = str(emotion_label)
            print(emot)
            print(f"Classified Emotion: {emotion_label}")
            return emot
        else:
            print("Failed to classify emotion")
            return "error"
    except Exception as ex:
        print(f"API request failed: {ex}")

def savevc(dir, audio):
    with open(dir, "wb") as f:
        f.write(audio)
    return f

maping = {
    "English": "en",
    "French": "fr", # French
    "Japenese": "ja", # Japanese
    "Hindi": "hi"
}

languages = ["Tamil", "English", "French", "Japenese", "Hindi", "Malayalam"]
vlang = ["English", "French", "Japenese", "Hindi"]

language_mapping = {
    "" : "",
    'Tamil': 'ta-IN',  # Tamil
    'English': 'en-US',  # English
    'French': 'fr-FR',  # French
    'Japenese': 'ja-JP',# Japanese
    'Hindi': 'hi-IN',  # hindi
    'Malayalam': 'ml-IN'
}

with gr.Blocks() as demo:
    with gr.Tab("Speech To Text"):
        with gr.Row():
            audiof = gr.Audio(label="Input Audio",  type="filepath",format="wav")
            print("Type: ", type(audiof))

        with gr.Row():
            submit = gr.Button(value = "Submit")

        with gr.Row():
            with gr.Column():
                fromlang = gr.Dropdown(choices=languages, label="Source Language")
            with gr.Column():
                tolang = gr.Dropdown(choices=languages, label="Target Language")

        with gr.Row():
            with gr.Column():
                start = gr.Button(value="Start Recording")
            with gr.Column():
                stop = gr.Button(value="Stop Recording")

        transcribed_text = gr.Textbox(label="Transcribed Text")
        translated_text = gr.Textbox(label="Translate Text")

        start.click(fn=recorder.start_recording)
        stop.click(fn=recorder.stop_recording, inputs=[fromlang, tolang], outputs=[transcribed_text, translated_text])
        submit.click(fn = recorder.audiofile, inputs=[audiof,fromlang,  tolang], outputs=[transcribed_text, translated_text])
    
    with gr.Tab("Text To Text"):
        with gr. Row():
            # with gr.Column():
            #     fromlang = gr.Dropdown(choices=language_mapping.values(), label="Source Language")
            # with gr.Column():
                tolang = gr.Dropdown(choices=languages, label="Target Language")

        with gr.Row():
            text = gr.TextArea(label="Text Prompt")#, description="Enter the text here that you want to Translate")

        with gr.Row():
            trans = gr.TextArea(label="Translated Text")#, description="The text you entered is Translated here")

        text.change(fn= recorder.trans_text, inputs=[text, tolang], outputs=trans)

    with gr.Tab("Contextual Analysis"):
        with gr. Row():
            # with gr.Column():
            #     fromlang = gr.Dropdown(choices=language_mapping.values(), label="Source Language")
            # with gr.Column():
                tolang = gr.Dropdown(choices=languages, label="Target Language")

        with gr.Row():
            text = gr.TextArea(label="Text Prompt")#, description="Enter the text here that you want to Translate")

        with gr.Row():
            trans = gr.TextArea(label="Translated Text")#, description="The text you entered is Translated here")


        text.change(fn=context, inputs=[text, tolang], outputs=trans)
   
    with gr.Tab("Voice Cloning"):
        with gr.Row():
            tolang = gr.Dropdown(choices=vlang, label="Target Language")

        with gr.Row():
            text = gr.TextArea(label="Text Prompt")#, description="One or Two sentences to here from the cloned voice")


        with gr.Row():
            audio = gr.Audio(label="Reference Audio", type="filepath",format="wav")
            print("Type: ", type(audiof))
        
        send = gr.Button(value="Generate")

        dir = "audios/vcinputs"

        file = savevc(dir, audiof)
        print(file)
        
        with gr.Row():
            syn = gr.Audio(label="Synthesised Audio", type="filepath", format="wav")
            # print("Type: ", type(audiof))
        
        send.click(fn=voiceclone, inputs=[file, text, tolang], outputs=syn)

    with gr.Tab("Voice Emotion"):
        with gr.Row():
            file_emo = gr.Audio(label="Reference Audio", type="filepath",format="wav")
            
            emo = gr.Button(value='Analyse')
        
        with gr.Row():
            emotions = gr.Label(label="Emotions: ")
            
        emo.click(fn=audio_emo, inputs=[file_emo], outputs=emotions)


demo.launch(share=True)