import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
import torch
import os
import tempfile
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import torchaudio
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Loading the HuggingFace token from .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Settings
SAMPLE_RATE = 16000  # 16kHz is what Whisper expects
CHUNK_DURATION = 5   # could set to lower time if needed later

# Load Whisper model
print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")
print("Whisper ready!")

# Load Pyannote speaker diarization model
print("Loading Pyannote model...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)
print("Pyannote ready!")

print("Loading T5 Speech Sanitizer model...")
stutter_id = "./t5-speech-sanitizer" 
stutter_tokenizer = AutoTokenizer.from_pretrained(stutter_id)
stutter_model = AutoModelForSeq2SeqLM.from_pretrained(stutter_id)
print("Sanitizer ready!")

# Record audio from microphone for a set duration
def record_audio():
    print(f"\nRecording for {CHUNK_DURATION} seconds... Speak now!")
    
    # Recordung
    audio_data = sd.rec(
        int(CHUNK_DURATION * SAMPLE_RATE),  # Total number of samples
        samplerate=SAMPLE_RATE,
        channels=1,                       
        dtype=np.int16                     
    )
    sd.wait()
    print("Recording done!")
    return audio_data

# Save audio chunk to a temporary file so Pyannote can read it
def save_temp_wav(audio_data):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(temp_file.name, SAMPLE_RATE, audio_data)
    return temp_file.name

# Run speaker diarization on the audio file
def detect_speakers(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    audio_dict = {"waveform": waveform, "sample_rate": sample_rate}
    diarization = diarization_pipeline(audio_dict)
    
    # Pyannote 4.x uses .serialize() to get results
    result = diarization.serialize()
    segments = []
    for item in result["diarization"]:
        segments.append({
            "start": round(item["start"], 2),
            "end": round(item["end"], 2),
            "speaker": item["speaker"]
        })
    
    return segments

## whisper ##
# Transcribe audio using Whisper
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["segments"]

# Combine speaker segments with transcription
def merge_speaker_transcript(diarization_segments, whisper_segments):
    if not diarization_segments:
        return [{"speaker": "SPEAKER_00", "text": seg["text"]} for seg in whisper_segments]
    
    output = []
    for w_seg in whisper_segments:
        # 4. Find the midpoint of the spoken text to map it to the correct speaker
        midpoint = w_seg["start"] + ((w_seg["end"] - w_seg["start"]) / 2)
        assigned_speaker = "UNKNOWN"
        
        for d_seg in diarization_segments:
            if d_seg["start"] <= midpoint <= d_seg["end"]:
                assigned_speaker = d_seg["speaker"]
                break
                
        output.append({
            "speaker": assigned_speaker,
            "text": w_seg["text"].strip()
        })
        
    return output

def sanitize_text(text):
    """Passes raw text through the T5 model to remove disfluencies."""
    if not text.strip():
        return ""
    
    stutter_input = stutter_tokenizer("sanitize speech: " + text, return_tensors="pt")
    stutter_output = stutter_model.generate(
        **stutter_input, 
        max_length=128, 
        num_beams=4, 
        no_repeat_ngram_size=2
    )
    return stutter_tokenizer.decode(stutter_output[0], skip_special_tokens=True)

# Main
def main():
    print("\n=== Speech to Text with Speaker Detection ===")
    print("CntrlC to quit\n")
    
    while True:
        try:
            # Step 1: Record audio
            audio_data = record_audio()
            
            # Step 2: Save to temp file
            temp_path = save_temp_wav(audio_data)
            
            # Step 3: Detect speakers
            print("Detecting speakers...")
            segments = detect_speakers(temp_path)
            
            # Step 4: Transcribe with Whisper
            print("Transcribing...")
            transcript = transcribe_audio(temp_path)
            
            # Step 5: Merge and print output
            results = merge_speaker_transcript(segments, transcript)
            
            print("\n--- Output ---")
            for r in results:
                raw_text = r['text']
                clean_text = sanitize_text(raw_text)
                
                print(f"[{r['speaker']}] (Raw)  : {raw_text}")
                print(f"[{r['speaker']}] (Clean): {clean_text}")
                print("-")
            print("--------------\n")
            
            # Step 6: Clean up temp file
            os.remove(temp_path)
            
        except KeyboardInterrupt:
            print("\nStopped.")
            break

if __name__ == "__main__":
    main()