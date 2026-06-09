# [cite_start]Live Speech-to-Text Transcription with Speaker Diarization and Disfluency Removal [cite: 1]

## Overview
[cite_start]This project presents a real-time, end-to-end Natural Language Processing (NLP) pipeline designed to address the challenges of live multi-speaker speech transcription and disfluency removal[cite: 9]. 

[cite_start]In everyday conversational settings, spoken audio is inherently messy, containing hesitation fillers (e.g., "um", "uh"), stutters, and grammatical irregularities[cite: 16]. [cite_start]This system captures live microphone audio continuously, detects and labels individual speakers, transcribes the speech, and sanitizes the output by removing conversational disfluencies in real time[cite: 21].

## System Architecture
[cite_start]The pipeline is modular, consisting of an Automatic Speech Recognition and Speaker Diarization module feeding into a two-stage NLP text sanitization module[cite: 22].

### 1. ASR and Speaker Diarization
* [cite_start]**Audio Capture:** Captures 5 seconds of raw microphone input at 16kHz using the `sounddevice` library[cite: 50]. [cite_start]This 5-second window was found to be the optimal balance between responsiveness and accuracy[cite: 93].
* [cite_start]**Speaker Diarization:** Utilizes `pyannote.audio 4.x` to extract time-stamped speaker segments using unique vocal frequency voice embeddings[cite: 47, 48].
* [cite_start]**Speech-to-Text:** Employs OpenAI's `Whisper` (tiny variant) to process the mel-spectrogram and transcribe the audio into raw text[cite: 43, 79]. [cite_start]The tiny variant was selected to ensure the model runs efficiently on CPU[cite: 45].
* [cite_start]**Output:** Merges the speaker segments and transcript to produce structured, speaker-labelled text (e.g., `[SPEAKER_00]: transcript text`)[cite: 53].

### 2. NLP Text Sanitization
[cite_start]To resolve task interference and memory capacity limitations, the text sanitization process utilizes a modular, two-stage sequential pipeline[cite: 65].
* **Stage 1: Acoustic Sanitization**
  * [cite_start]A custom fine-tuned `T5-small` sequence-to-sequence model[cite: 57, 66]. 
  * [cite_start]Dedicated to identifying and erasing conversational fillers and stuttering patterns without altering the surrounding text[cite: 66].
  * [cite_start]Utilizes Beam Search Decoding (`num_beams=4`, `no_repeat_ngram_size=2`) to mathematically eliminate hallucination loops[cite: 102, 103].
* **Stage 2: Grammatical Polish**
  * [cite_start]The sanitized output is passed into a secondary, larger pre-trained model (`grammarly/coedit-large`)[cite: 67].
  * [cite_start]Acts as a final validation layer, correcting casing, punctuation, and residual grammatical errors[cite: 68].

## Example Output
[cite_start]The system successfully bridges the gap between raw, messy acoustic transcription and clean, presentation-ready text[cite: 113].

**Raw ASR Input:**
> [cite_start]"I- I went to the town mekka after that and um, forgave the people whose abused him" [cite: 117]

**Stage 1 Output (Stutters/Fillers Removed):**
> [cite_start]"I went to the town mekka after that and forgave the people whose abused him." [cite: 118]

**Stage 2 Output (Grammar Fixed):**
> [cite_start]"I went to the town Mekka after that and forgave the people who had abused him." [cite: 119]

## Training Details & Hyperparameters (NLP Module)
[cite_start]The T5 model was fine-tuned using a supervised learning approach on a cloud-based accelerator instance[cite: 95].
* [cite_start]**Epochs:** 3 (flattened by the third epoch, indicating optimal convergence)[cite: 97].
* [cite_start]**Optimization:** Mixed Precision (`fp16=True`) alongside a batch size of 16 to maximize GPU VRAM efficiency[cite: 99].
* [cite_start]**Learning Rate:** 2e-5 with a weight decay of 0.01[cite: 100].

## Team Members
* [cite_start]**Muneef Azwad** - NLP Module Lead [cite: 5, 129]
* [cite_start]**Mohamed Sahdhin Mohamed Ali** - ASR and Speaker Diarization Lead [cite: 4, 121]
* [cite_start]**Ayush** - Team Member [cite: 3]

*Subject: Natural Language Processing | [cite_start]University of Technology Sydney 2026* [cite: 6, 7]
