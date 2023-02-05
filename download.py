# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import torch
import whisper
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

def download_model():
    #medium, large-v1, large-v2
    model_name = "large-v2"
    model = whisper.load_model(model_name)
    embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")

if __name__ == "__main__":
    download_model()