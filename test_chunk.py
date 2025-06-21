# check_chunking.py

import json
from transformers import GPT2Tokenizer

# Load data
with open("dsp/processed_meetings.json", "r", encoding="utf-8") as f:
    processed_meetings = json.load(f)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def create_transcript_chunk(dialogues):
    return [f"{d['speaker']} ({d['timestamp']}): {d['content']}" for d in dialogues]

def chunk_dialogues(dialogues, chunk_size=30):
    chunks = []
    for i in range(0, len(dialogues), chunk_size):
        chunk = dialogues[i:i + chunk_size]
        transcript = "\n".join(create_transcript_chunk(chunk))
        token_count = len(tokenizer.encode(transcript))
        chunks.append((transcript, token_count))
    return chunks

# Kiểm tra hội thoại đầu tiên
sample_meeting = processed_meetings[0]["dialogue"]
chunked_transcripts = chunk_dialogues(sample_meeting, chunk_size=30)

for idx, (transcript, token_count) in enumerate(chunked_transcripts):
    print(f"\n--- Chunk {idx+1} ---")
    print(f"Số tokens: {token_count}")
    print(transcript[:300] + "...\n")  # In preview
