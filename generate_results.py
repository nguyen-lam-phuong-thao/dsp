import time
import json
import os
import random
from typing import Any, Optional, Dict, List
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from deepeval.models.base_model import DeepEvalBaseLLM
import litellm
from transformers import GPT2Tokenizer

load_dotenv()

# --- Reusable Components from evaluate_meeting.py ---

class CustomGeminiLLM(DeepEvalBaseLLM):
    def __init__(self):
        from instructor import from_litellm
        from litellm import completion
        self.client = from_litellm(completion)
        self.raw_model = completion

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: BaseModel = None, **kwargs) -> Any:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if schema is not None:
                    response = self.client.chat.completions.create(
                        model="gemini/gemini-2.0-flash",
                        messages=[{"role": "user", "content": prompt}],
                        response_model=schema,
                        **kwargs,
                    )
                    return response
                else:
                    response = self.raw_model(
                        model="gemini/gemini-2.0-flash",
                        messages=[{"role": "user", "content": prompt}],
                        **kwargs,
                    )
                    content = response.choices[0].message.content
                    if isinstance(content, str):
                        content = content.replace("\n", " ").replace("\r", " ").replace("\\", "/")
                    return content
            except litellm.RateLimitError as e:
                wait_time = 60 * (attempt + 1)
                print(f"[Gemini] Rate limit hit. Waiting {wait_time}s before retrying... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            except Exception as e:
                print(f"[Gemini] Error generating response: {e}")
                # Return a specific error structure if generation fails
                return {"error": str(e)}
        print(f"[Gemini] Failed to generate response after {max_retries} retries.")
        return {"error": "Max retries exceeded"}


    async def a_generate(self, prompt: str, schema: BaseModel = None, **kwargs) -> str:
        return self.generate(prompt, schema, **kwargs)

    def get_model_name(self):
        return "Custom Gemini LLM"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def chunk_dialogues(dialogues, max_tokens=2800):
    chunks = []
    current_chunk = []
    current_tokens = 0
    for d in dialogues:
        line = f"{d['speaker']} ({d['timestamp']}): {d['content']}"
        line_tokens = len(line.split())
        if current_tokens + line_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [d]
            current_tokens = len(tokenizer.encode("\n".join(create_transcript_chunk(current_chunk))))
        else:
            current_chunk.append(d)
            current_tokens += line_tokens
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def create_transcript_chunk(dialogues):
    return [f"{d['speaker']} ({d['timestamp']}): {d['content']}" for d in dialogues]

def create_transcript(dialogues):
    return "\n".join(create_transcript_chunk(dialogues))

def get_first_timestamp(dialogues):
    return dialogues[0]["timestamp"] if dialogues else "Không xác định"

# --- Phase 1: Generation Logic ---

def generate_model_summaries(model, transcript, first_timestamp):
    prompt = f"""Dựa trên đoạn hội thoại:\n\n{transcript}\n\nHãy trả lời:\n1. Bao nhiêu người tham gia?\n2. Tóm tắt?\n3. Ý chính?\n4. Thời gian? (gợi ý: {first_timestamp})\n\nTrả lời theo JSON: {{"num_participants": [số], "summary": "[...]", "main_point": "[...]", "time": "[...]"}}"""
    start_time = time.time()
    try:
        response = model.invoke(prompt)
        content = response.content
    except Exception as e:
        print(f"Error invoking model: {e}")
        content = f'{{"error": "Failed to generate summary: {e}"}}' 
        
    duration = time.time() - start_time
    return {"summary": content, "duration": duration}

def generate_ground_truth(llm, transcript, first_timestamp):
    prompt = f"""Dựa trên đoạn hội thoại:\n\n{transcript}\n\nHãy trả lời:\n1. Bao nhiêu người tham gia?\n2. Tóm tắt?\n3. Ý chính?\n4. Thời gian? (gợi ý: {first_timestamp})\n\nTrả lời theo JSON: {{"num_participants": [số], "summary": "[...]", "main_point": "[...]", "time": "[...]"}}"""
    start_time = time.time()
    response = llm.generate(prompt)
    duration = time.time() - start_time
    return {"summary": response, "duration": duration}

def main():
    # Load data
    with open("dsp/processed_meetings.json", "r", encoding="utf-8") as f:
        processed_meetings = json.load(f)

    if not processed_meetings:
        print("Không có dữ liệu để xử lý.")
        return

    # Initialize models
    gemini_llm = CustomGeminiLLM()
    ollama_models = {
        "llama2:latest": ChatOllama(model="llama2:latest", temperature=0.8),
        "llama3:8b": ChatOllama(model="llama3:8b", temperature=0.8),
        "mistral:latest": ChatOllama(model="mistral:latest", temperature=0.8),
    }

    # Sample and process dialogues
    num_samples = min(200, len(processed_meetings))
    print(f"Sẽ xử lý {num_samples} hội thoại ngẫu nhiên từ tập dữ liệu.")
    sampled_meetings = random.sample(processed_meetings, num_samples)
    
    all_results = []
    
    for i, meeting in enumerate(sampled_meetings):
        dialogue_id = meeting.get('id', f'dialogue_{i}')
        print(f"\n{'='*20} Xử lý hội thoại {i+1}/{num_samples} (ID: {dialogue_id}) {'='*20}")
        dialogues = meeting["dialogue"]

        if not dialogues:
            print("Hội thoại rỗng, bỏ qua.")
            continue

        dialogue_chunks = chunk_dialogues(dialogues, max_tokens=2200)
        print(f"Hội thoại được chia thành {len(dialogue_chunks)} chunks.")

        for j, chunk in enumerate(dialogue_chunks):
            print(f"\n--- Xử lý Chunk {j+1}/{len(dialogue_chunks)} ---")
            transcript = create_transcript(chunk)
            first_timestamp = get_first_timestamp(chunk)

            # Generate ground truth with Gemini
            print("Generating ground truth with Gemini...")
            ground_truth_result = generate_ground_truth(gemini_llm, transcript, first_timestamp)
            
            # Generate summaries with Ollama models
            model_summaries = {}
            for model_name, model in ollama_models.items():
                print(f"Generating summary with {model_name}...")
                model_summaries[model_name] = generate_model_summaries(model, transcript, first_timestamp)

            chunk_result = {
                "dialogue_id": dialogue_id,
                "chunk_index": j,
                "transcript": transcript,
                "ground_truth": ground_truth_result,
                "models": model_summaries
            }
            all_results.append(chunk_result)
            
            # Rate limiting before next chunk
            if j < len(dialogue_chunks) - 1:
                print("--- Chờ 10 giây trước khi xử lý chunk tiếp theo ---")
                time.sleep(10)

        # Rate limiting before next dialogue
        if i < len(sampled_meetings) - 1:
            print(f"\n{'='*20} Chờ 20 giây trước khi xử lý hội thoại tiếp theo {'='*20}\n")
            time.sleep(20)

    # Save all results to a file
    output_path = "dsp/results/generated_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    print(f"\nĐã lưu thành công {len(all_results)} kết quả chunks vào {output_path}")

if __name__ == "__main__":
    if not os.path.exists(".deepeval"):
        with open(".deepeval", "w") as f:
            f.write('{"USE_AZURE_OPENAI": false}')
    main() 