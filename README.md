# Live Speech-to-Text Transcription with Speaker Diarization and Disfluency Removal

## Overview
This project presents a real-time, end-to-end Natural Language Processing (NLP) pipeline designed to address the challenges of live multi-speaker speech transcription and disfluency removal. 

In everyday conversational settings, spoken audio is inherently messy, containing hesitation fillers (e.g., "um", "uh"), stutters, and grammatical irregularities. This system captures live microphone audio continuously, detects and labels individual speakers, transcribes the speech, and sanitizes the output by removing conversational disfluencies in real time.

## System Architecture
The pipeline is modular, consisting of an Automatic Speech Recognition and Speaker Diarization module feeding into a two-stage NLP text sanitization module.

### 1. ASR and Speaker Diarization
* **Audio Capture:** Captures 5 seconds of raw microphone input at 16kHz using the `sounddevice` library. This 5-second window was found to be the optimal balance between responsiveness and accuracy.
* **Speaker Diarization:** Utilizes `pyannote.audio 4.x` to extract time-stamped speaker segments using unique vocal frequency voice embeddings.
* **Speech-to-Text:** Employs OpenAI's `Whisper` (tiny variant) to process the mel-spectrogram and transcribe the audio into raw text. The tiny variant was selected to ensure the model runs efficiently on CPU.
* **Output:** Merges the speaker segments and transcript to produce structured, speaker-labelled text (e.g., `[SPEAKER_00]: transcript text`).

### 2. NLP Text Sanitization
To resolve task interference and memory capacity limitations, the text sanitization process utilizes a modular, two-stage sequential pipeline.
* **Stage 1: Acoustic Sanitization**
  * A custom fine-tuned `T5-small` sequence-to-sequence model. 
  * Dedicated to identifying and erasing conversational fillers and stuttering patterns without altering the surrounding text.
  * Utilizes Beam Search Decoding (`num_beams=4`, `no_repeat_ngram_size=2`) to mathematically eliminate hallucination loops.
* **Stage 2: Grammatical Polish**
  * The sanitized output is passed into a secondary, larger pre-trained model (`grammarly/coedit-large`).
  * Acts as a final validation layer, correcting casing, punctuation, and residual grammatical errors.

## Example Output
The system successfully bridges the gap between raw, messy acoustic transcription and clean, presentation-ready text.

**Raw ASR Input:**
> "I- I went to the town mekka after that and um, forgave the people whose abused him"

**Stage 1 Output (Stutters/Fillers Removed):**
> "I went to the town mekka after that and forgave the people whose abused him."

**Stage 2 Output (Grammar Fixed):**
> "I went to the town Mekka after that and forgave the people who had abused him."

## Training Details & Hyperparameters (NLP Module)
The T5 model was fine-tuned using a supervised learning approach on a cloud-based accelerator instance.
* **Epochs:** 3 (flattened by the third epoch, indicating optimal convergence).
* **Optimization:** Mixed Precision (`fp16=True`) alongside a batch size of 16 to maximize GPU VRAM efficiency.
* **Learning Rate:** 2e-5 with a weight decay of 0.01.

## Team Members
* **Muneef Azwad** - NLP Module Lead
* **Mohamed Sahdhin Mohamed Ali** - ASR and Speaker Diarization Lead
* **Ayush** - Team Member

*Subject: Natural Language Processing | University of Technology Sydney 2026*
