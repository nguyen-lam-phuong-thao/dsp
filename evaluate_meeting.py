import json
from langchain_ollama import ChatOllama
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM
import litellm
import time
import os

class CustomGeminiLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.model = litellm.completion

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model(
            model="gemini-2.0-flash",  
            messages=[{"role": "user", "content": prompt}],
            api_key="AIzaSyBLPqdI2JSWLidxSRMtlRmrQUKFqWZXXYc"  
        )
        return response['choices'][0]['message']['content']

    def get_model_name(self):
        return "Custom Gemini LLM"

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)  

with open("processed_meetings.json", "r", encoding="utf-8") as f:
    processed_meetings = json.load(f)

def create_transcript(dialogues):
    return "\n".join([f"{d['speaker']} ({d['timestamp']}): {d['content']}" for d in dialogues])

def get_first_timestamp(dialogues):
    return dialogues[0]["timestamp"] if dialogues else "Không xác định"

def evaluate_models(dialogues):
    transcript = create_transcript(dialogues)
    first_timestamp = get_first_timestamp(dialogues)
    
    ollama_models = {
        "llama2:latest": ChatOllama(model="llama2:latest", temperature=0.8),
        "llama3:8b": ChatOllama(model="llama3:8b", temperature=0.8),
        "mistral:latest": ChatOllama(model="mistral:latest", temperature=0.8),
    }

    def generate_summary(model, transcript, first_timestamp):
        prompt = f"""Dựa trên đoạn hội thoại cuộc họp sau:\n\n{transcript}\n\nHãy trả lời các câu hỏi sau một cách ngắn gọn, rõ ràng:\n1. Đoạn hội thoại này có mấy người tham gia?\n2. Tóm tắt nội dung đoạn hội thoại.\n3. Ý chính/trọng điểm của đoạn hội thoại là gì?\n4. Đoạn hội thoại diễn ra lúc nào? (dựa trên thông tin có sẵn, ví dụ: {first_timestamp})\n\nTrả lời theo định dạng JSON: {{"num_participants": [số], "summary": "[nội dung]", "main_point": "[nội dung]", "time": "[thời gian]"}}"""
        start_time = time.time()
        response = model.invoke(prompt)
        duration = time.time() - start_time
        return response.content, duration

    results = {}
    for model_name, model in ollama_models.items():
        summary, duration = generate_summary(model, transcript, first_timestamp)
        results[model_name] = {"summary": summary, "duration": duration}
    return results

def evaluate_with_ground_truth(dialogues, model_results):
    transcript = create_transcript(dialogues)
    first_timestamp = get_first_timestamp(dialogues)
    
    custom_llm = CustomGeminiLLM()

    def generate_ground_truth(transcript, first_timestamp):
        prompt = f"""Dựa trên đoạn hội thoại cuộc họp sau:\n\n{transcript}\n\nHãy trả lời các câu hỏi sau một cách ngắn gọn, rõ ràng:\n1. Đoạn hội thoại này có mấy người tham gia?\n2. Tóm tắt nội dung đoạn hội thoại.\n3. Ý chính/trọng điểm của đoạn hội thoại là gì?\n4. Đoạn hội thoại diễn ra lúc nào? (dựa trên thông tin có sẵn, ví dụ: {first_timestamp})\n\nTrả lời theo định dạng JSON: {{"num_participants": [số], "summary": "[nội dung]", "main_point": "[nội dung]", "time": "[thời gian]"}}"""
        start_time = time.time()
        response = custom_llm.generate(prompt)
        duration = time.time() - start_time
        return response, duration

    ground_truth, _ = generate_ground_truth(transcript, first_timestamp)

    test_cases = []
    for model_name, result in model_results.items():
        test_case = LLMTestCase(
            input=transcript,
            actual_output=result["summary"],
            expected_output=ground_truth,
            retrieval_context=[transcript]  # Cung cấp ngữ cảnh cho metric contextual
        )
        test_cases.append(test_case)

    # Đánh giá với các metric contextual
    metrics = [
        ContextualPrecisionMetric(model=custom_llm, threshold=0.5),
        ContextualRecallMetric(model=custom_llm, threshold=0.5),
        ContextualRelevancyMetric(model=custom_llm, threshold=0.5)
    ]
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        print_results=True
    )

    evaluations = {}
    for i, (model_name, result) in enumerate(model_results.items()):
        evaluations[model_name] = {
            "precision": test_cases[i].metrics[0].score,
            "recall": test_cases[i].metrics[1].score,
            "relevancy": test_cases[i].metrics[2].score,
            "duration": result["duration"]
        }

    return ground_truth, evaluations

def main():
    if not processed_meetings:
        print("Không có dữ liệu để xử lý.")
        return
    
    first_meeting = processed_meetings[0]
    dialogues = first_meeting["dialogue"]

    print("\n=== Kết quả đánh giá mô hình (Outcome 2) ===")
    model_results = evaluate_models(dialogues)
    for model_name, result in model_results.items():
        print(f"Mô hình: {model_name}")
        print(f"Thời gian thực thi: {result['duration']:.2f} giây")
        print(f"Kết quả:\n{result['summary']}\n")

    print("\n=== Kết quả so sánh với ground truth (Outcome 3) ===")
    ground_truth, evaluations = evaluate_with_ground_truth(dialogues, model_results)
    print(f"Ground Truth:\n{ground_truth}\n")
    for model_name, eval in evaluations.items():
        print(f"Mô hình: {model_name}")
        print(f"Thời gian thực thi: {eval['duration']:.2f} giây")
        print(f"Contextual Precision: {eval['precision']:.2f}")
        print(f"Contextual Recall: {eval['recall']:.2f}")
        print(f"Contextual Relevancy: {eval['relevancy']:.2f}\n")

if __name__ == "__main__":
    # Tạo file .deepeval nếu chưa tồn tại
    if not os.path.exists(".deepeval"):
        with open(".deepeval", "w") as f:
            f.write('{"USE_AZURE_OPENAI": false}')
    main()