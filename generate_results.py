import time
import json
import os
import random
from typing import Any, Dict, List
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from langchain_ollama import ChatOllama
from deepeval.models.base_model import DeepEvalBaseLLM
import litellm
from transformers import GPT2Tokenizer

load_dotenv()

class OptimizedGeminiLLM(DeepEvalBaseLLM):
    def __init__(self):
        self._client = None  # Khởi tạo trước khi gọi super()
        self.last_call = 0
        self.delay = 25
        super().__init__()  # Gọi super() sau khi đã khởi tạo client

    def load_model(self):
        if self._client is None:
            from litellm import completion
            self._client = completion
        return self._client

    def get_model_name(self):
        return "gemini/gemini-2.0-flash"

    async def a_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    def generate(self, prompt: str, **kwargs) -> Dict:
        current_time = time.time()
        if current_time - self.last_call < self.delay:
            time.sleep(self.delay - (current_time - self.last_call))
        
        try:
            self.last_call = time.time()
            response = self.load_model()(  
                model=self.get_model_name(),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                **kwargs
            )
            return {"summary": response.choices[0].message.content[:1000]}
        except Exception as e:
            print(f"Gemini Error: {str(e)[:200]}")
            return {"error": str(e)}

def chunk_dialogues(dialogues: List[Dict], max_tokens: int = 3000) -> List[List[Dict]]:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for d in dialogues:
        line = f"{d['speaker']}: {d['content'][:300]}"
        line_tokens = len(tokenizer.tokenize(line))
        
        if current_tokens + line_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [d]
            current_tokens = line_tokens
        else:
            current_chunk.append(d)
            current_tokens += line_tokens
    
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def filter_dialogues_by_chunks(dialogues: List[Dict], max_chunks: int = 9) -> List[Dict]:
    suitable_dialogues = []
    for dialogue in dialogues:
        chunks = chunk_dialogues(dialogue["dialogue"])
        if len(chunks) <= max_chunks:
            suitable_dialogues.append(dialogue)
            if len(suitable_dialogues) >= 100:
                break
    return suitable_dialogues

def process_models_parallel(models: Dict, transcript: str) -> Dict:
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                lambda m: (name, m.invoke(f"Phân tích ngắn:\n{transcript}")),
                model
            ): name for name, model in models.items()
        }
        for future in futures:
            try:
                name, response = future.result()
                results[name] = {"summary": response.content[:1000]}
            except Exception as e:
                results[futures[future]] = {"error": str(e)[:200]}
    return results

def main():
    with open("dsp/processed_meetings.json", "r", encoding="utf-8") as f:
        all_meetings = json.load(f)
    
    print("Đang lọc 100 hội thoại có ≤9 chunks...")
    selected_meetings = filter_dialogues_by_chunks(all_meetings)
    
    gemini = OptimizedGeminiLLM()
    ollama_models = {
        "llama2": ChatOllama(model="llama2:latest", temperature=0.7),
        "llama3": ChatOllama(model="llama3:8b", temperature=0.7),
        "mistral": ChatOllama(model="mistral:latest", temperature=0.7)
    }
    
    results = []
    
    for idx, meeting in enumerate(selected_meetings):
        print(f"Processing {idx+1}/{len(selected_meetings)} - ID: {meeting.get('id', '')}")
        
        chunks = chunk_dialogues(meeting["dialogue"])
        
        for chunk in chunks:
            try:
                transcript = "\n".join(f"{d['speaker']}: {d['content'][:300]}" for d in chunk)
                ground_truth = gemini.generate(f"Tóm tắt ngắn:\n{transcript}")
                model_results = process_models_parallel(ollama_models, transcript)
                
                results.append({
                    "dialogue_id": meeting.get("id", f"dialogue_{idx}"),
                    "chunk_size": len(chunk),
                    "transcript": transcript[:1500],
                    "ground_truth": ground_truth,
                    "models": model_results
                })
                
                time.sleep(5)
            except Exception as e:
                print(f"Lỗi xử lý chunk: {str(e)[:200]}")
        
        if idx % 10 == 0:
            time.sleep(15)
    
    os.makedirs("dsp/results", exist_ok=True)
    with open("dsp/results/compare_3models.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nHoàn thành! Đã xử lý {len(results)} chunks từ {len(selected_meetings)} meetings")

if __name__ == "__main__":
    if not os.path.exists(".deepeval"):
        with open(".deepeval", "w") as f:
            f.write('{"USE_AZURE_OPENAI": false}')
    main()
