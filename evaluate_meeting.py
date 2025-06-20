import time
import json
import os
from typing import Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
import instructor
import litellm

load_dotenv()

class CustomGeminiLLM(DeepEvalBaseLLM):
    def __init__(self):
        from instructor import from_litellm
        from litellm import completion
        self.client = from_litellm(completion)
        self.raw_model = completion

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: BaseModel = None, **kwargs) -> str:
        max_retries = 3
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
                wait_time = 60
                print(f"[Gemini] Rate limit hit. Waiting {wait_time}s before retrying... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            except Exception as e:
                print(f"[Gemini] Error: {e}")
                if schema is not None:
                    try:
                        # Với schema có trường 'verdicts', khởi tạo đúng dạng
                        if "verdicts" in schema.model_fields:
                            return schema(verdicts=[])
                        # Schema có trường 'reason'
                        if "reason" in schema.model_fields:
                            return schema(reason=f"Fallback after error: {e}")
                        # Schema khác
                        return schema()
                    except Exception as fallback_e:
                        print(f"⚠️ Fallback schema failed: {fallback_e}")
                        # Bắt buộc khởi tạo đúng schema type
                        if "verdicts" in schema.model_fields:
                            from deepeval.metrics.contextual_relevancy import ContextualRelevancyVerdicts
                            return ContextualRelevancyVerdicts(verdicts=[])
                        return schema()
                return None


    async def a_generate(self, prompt: str, schema: BaseModel = None, **kwargs) -> str:
        return self.generate(prompt, schema, **kwargs)

    def get_model_name(self):
        return "Custom Gemini LLM"

# Load data
with open("dsp/processed_meetings.json", "r", encoding="utf-8") as f:
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
        prompt = f"""Dựa trên đoạn hội thoại:\n\n{transcript}\n\nHãy trả lời:\n1. Bao nhiêu người tham gia?\n2. Tóm tắt?\n3. Ý chính?\n4. Thời gian? (gợi ý: {first_timestamp})\n\nTrả lời theo JSON: {{"num_participants": [số], "summary": "[...]", "main_point": "[...]", "time": "[...]"}}"""
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

    def generate_ground_truth():
        prompt = f"""Dựa trên đoạn hội thoại:\n\n{transcript}\n\nHãy trả lời:\n1. Bao nhiêu người tham gia?\n2. Tóm tắt?\n3. Ý chính?\n4. Thời gian? (gợi ý: {first_timestamp})\n\nTrả lời theo JSON: {{"num_participants": [số], "summary": "[...]", "main_point": "[...]", "time": "[...]"}}"""
        start_time = time.time()
        response = custom_llm.generate(prompt)
        duration = time.time() - start_time
        return response, duration

    ground_truth, _ = generate_ground_truth()

    test_cases = [
        LLMTestCase(
            input=transcript,
            actual_output=result["summary"],
            expected_output=ground_truth,
            retrieval_context=[transcript],
        )
        for result in model_results.values()
    ]

    metrics = [
        ContextualPrecisionMetric(model=custom_llm, threshold=0.5),
        ContextualRecallMetric(model=custom_llm, threshold=0.5),
        ContextualRelevancyMetric(model=custom_llm, threshold=0.5),
    ]

    results = evaluate(test_cases=test_cases, metrics=metrics, run_async=False)

    evaluations = {}
    for i, (model_name, model_result) in enumerate(model_results.items()):
        test_result = results.test_results[i]

        print(f"\nDEBUG - model_name: {model_name}")
        print("Attributes of test_result:", dir(test_result))

        evaluation = {
            "precision": 0.0,
            "recall": 0.0,
            "relevancy": 0.0,
            "duration": model_result["duration"],
        }

        for metric_result in test_result.metrics_data:
            print(f"Metric name: {metric_result.name}, Score: {metric_result.score}")
            if metric_result.name == "Contextual Precision":
                evaluation["precision"] = metric_result.score
            elif metric_result.name == "Contextual Recall":
                evaluation["recall"] = metric_result.score
            elif metric_result.name == "Contextual Relevancy":
                evaluation["relevancy"] = metric_result.score

        evaluations[model_name] = evaluation

    return ground_truth, evaluations


def main():
    if not processed_meetings:
        print("Không có dữ liệu để xử lý.")
        return
    dialogues = processed_meetings[0]["dialogue"]
    print("\n=== Kết quả đánh giá mô hình (Outcome 2) ===")
    model_results = evaluate_models(dialogues)
    for model_name, result in model_results.items():
        print(f"Mô hình: {model_name}")
        print(f"Thời gian thực thi: {result['duration']:.2f}s")
        print(f"Kết quả:\n{result['summary']}\n")
    print("\n=== Kết quả so sánh với ground truth (Outcome 3) ===")
    ground_truth, evaluations = evaluate_with_ground_truth(dialogues, model_results)
    print(f"Ground Truth:\n{ground_truth}\n")
    for model_name, eval in evaluations.items():
        print(f"Mô hình: {model_name}")
        print(f"Thời gian: {eval['duration']:.2f}s")
        print(f"Precision: {eval['precision']:.2f} | Recall: {eval['recall']:.2f} | Relevancy: {eval['relevancy']:.2f}\n")

if __name__ == "__main__":
    if not os.path.exists(".deepeval"):
        with open(".deepeval", "w") as f:
            f.write('{"USE_AZURE_OPENAI": false}')
    main()
