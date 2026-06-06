from faster_whisper import WhisperModel

audio_file = "40.mp3"  # change to your file

model = WhisperModel(
    "medium",
    device="cuda",
    compute_type="float16"
)

segments, info = model.transcribe(
    audio_file,
    beam_size=5
)

print(f"Detected language: {info.language}")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")