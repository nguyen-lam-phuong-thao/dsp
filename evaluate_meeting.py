import json
from langchain_ollama import ChatOllama
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM
import litellm
import time
import os
from pydantic import BaseModel
import instructor
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Set the Gemini API key as environment variable (best practice)



class CustomGeminiLLM(DeepEvalBaseLLM):
    """Custom LLM implementation for Gemini with structured output support using instructor."""

    def __init__(self):
        # Create instructor client for structured responses
        from instructor import from_litellm
        from litellm import completion

        self.client = from_litellm(completion)
        self.raw_model = completion

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: BaseModel = None, **kwargs) -> str:
        """Generate response with optional Pydantic schema validation."""
        try:
            if schema is not None:
                # Use instructor client for structured output
                response = self.client.chat.completions.create(
                    model="gemini/gemini-2.0-flash",
                    messages=[{"role": "user", "content": prompt}],
                    response_model=schema,
                    **kwargs,
                )
                # Return the Pydantic model instance
                return response
            else:
                # Regular text generation using raw litellm
                response = self.raw_model(
                    model="gemini/gemini-2.0-flash",
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
                content = response.choices[0].message.content
                # Clean up content for better JSON parsing
                if isinstance(content, str):
                    content = content.replace("\n", " ").replace("\r", " ")
                    # Try to clean up problematic escape sequences
                    content = content.replace("\\", "/")
                return content
        except Exception as e:
            print(f"Error in generate: {e}")
            if schema is not None:
                # Return a minimal valid response for the schema
                try:
                    return schema()
                except:
                    # If schema can't be instantiated, create a fallback
                    return (
                        schema(error="Failed to generate structured response")
                        if hasattr(schema, "error")
                        else schema()
                    )
            else:
                return '{"error": "Failed to generate response"}'

    async def a_generate(self, prompt: str, schema: BaseModel = None, **kwargs) -> str:
        """Async version of generate with schema support."""
        return self.generate(prompt, schema, **kwargs)

    def get_model_name(self):
        return "Custom Gemini LLM"


with open("processed_meetings.json", "r", encoding="utf-8") as f:
    processed_meetings = json.load(f)


def create_transcript(dialogues):
    return "\n".join(
        [f"{d['speaker']} ({d['timestamp']}): {d['content']}" for d in dialogues]
    )


def get_first_timestamp(dialogues):
    return dialogues[0]["timestamp"] if dialogues else "Không xác định"


def evaluate_models(dialogues):
    transcript = create_transcript(dialogues)
    first_timestamp = get_first_timestamp(dialogues)

    ollama_models = {
        "qwen3:8b": ChatOllama(model="qwen3:8b", temperature=0.8),
        "llama3.2:1b": ChatOllama(model="llama3.2:1b", temperature=0.8),
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
            retrieval_context=[transcript],  # Cung cấp ngữ cảnh cho metric contextual
        )
        test_cases.append(test_case)

    # Đánh giá với các metric contextual
    metrics = [
        ContextualPrecisionMetric(model=custom_llm, threshold=0.5),
        ContextualRecallMetric(model=custom_llm, threshold=0.5),
        ContextualRelevancyMetric(model=custom_llm, threshold=0.5),
    ]
    result = evaluate(test_cases=test_cases, metrics=metrics)

    evaluations = {}
    for i, (model_name, model_result) in enumerate(model_results.items()):
        # Truy cập metrics từ test_case sau khi evaluate
        test_case = test_cases[i]

        # Chạy metrics trực tiếp trên test case để lấy scores với error handling
        evaluations[model_name] = {
            "precision": 0.0,
            "recall": 0.0,
            "relevancy": 0.0,
            "duration": model_result["duration"],
        }

        try:
            precision_metric = ContextualPrecisionMetric(
                model=custom_llm, threshold=0.5
            )
            precision_metric.measure(test_case)
            evaluations[model_name]["precision"] = precision_metric.score
            print(f"✓ Precision metric completed for {model_name}")
        except Exception as e:
            print(f"✗ Error measuring precision for {model_name}: {e}")

        try:
            recall_metric = ContextualRecallMetric(model=custom_llm, threshold=0.5)
            recall_metric.measure(test_case)
            evaluations[model_name]["recall"] = recall_metric.score
            print(f"✓ Recall metric completed for {model_name}")
        except Exception as e:
            print(f"✗ Error measuring recall for {model_name}: {e}")

        try:
            relevancy_metric = ContextualRelevancyMetric(
                model=custom_llm, threshold=0.5
            )
            relevancy_metric.measure(test_case)
            evaluations[model_name]["relevancy"] = relevancy_metric.score
            print(f"✓ Relevancy metric completed for {model_name}")
        except Exception as e:
            print(f"✗ Error measuring relevancy for {model_name}: {e}")

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
