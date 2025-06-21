import json
import os
import time
from collections import defaultdict
from typing import Any, List
from dotenv import load_dotenv
from pydantic import BaseModel
import litellm
from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

load_dotenv()

# --- Reusable Gemini LLM for Evaluation ---

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
                if schema:
                    # Specific handling for deepeval's schema expectations
                    if "verdicts" in schema.model_fields:
                        from deepeval.metrics.utils import (
                            construct_verbose_logs,
                            trim_string,
                        )
                        from deepeval.metrics.contextual_relevancy import ContextualRelevancyVerdicts
                        # This part is complex and specific to deepeval's inner workings
                        # A simplified approach is taken here. For full fidelity, one might need
                        # to replicate the metric's internal logic.
                        # For now, we rely on a more direct generation and fallback.
                        pass # Fall through to the generic schema generation
                    
                    response = self.client.chat.completions.create(
                        model="gemini/gemini-2.0-flash",
                        messages=[{"role": "user", "content": prompt}],
                        response_model=schema, **kwargs
                    )
                    return response
                else:
                    response = self.raw_model(
                        model="gemini/gemini-2.0-flash",
                        messages=[{"role": "user", "content": prompt}],
                        **kwargs,
                    )
                    return response.choices[0].message.content
            except litellm.RateLimitError:
                wait_time = 60 * (attempt + 1)
                print(f"Rate limit hit. Waiting {wait_time}s... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            except Exception as e:
                print(f"Error during generation: {e}")
                # Fallback for specific deepeval schemas
                if schema and "verdicts" in schema.model_fields:
                    from deepeval.metrics.contextual_relevancy import ContextualRelevancyVerdicts
                    return ContextualRelevancyVerdicts(verdicts=[])
                if schema and "reason" in schema.model_fields:
                     return schema(reason=f"Fallback due to error: {e}", score=0)
                return None
        raise ConnectionError("Failed to generate response from Gemini after multiple retries.")

    async def a_generate(self, prompt: str, schema: BaseModel = None, **kwargs) -> Any:
        return self.generate(prompt, schema, **kwargs)

    def get_model_name(self):
        return "Custom Gemini LLM"

# --- Phase 2: Evaluation Logic ---

def main():
    results_path = "dsp/results/generated_results.json"
    if not os.path.exists(results_path):
        print(f"Không tìm thấy file kết quả tại '{results_path}'.")
        print("Vui lòng chạy 'generate_results.py' trước.")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    print(f"Đã tải {len(all_results)} kết quả chunks để đánh giá.")

    # Initialize evaluator and metrics
    gemini_evaluator = CustomGeminiLLM()
    metrics = [
        ContextualPrecisionMetric(model=gemini_evaluator, threshold=0.5),
        ContextualRecallMetric(model=gemini_evaluator, threshold=0.5),
        ContextualRelevancyMetric(model=gemini_evaluator, threshold=0.5),
    ]
    
    # Store aggregated results
    evaluation_scores = defaultdict(lambda: defaultdict(list))
    model_durations = defaultdict(list)

    for i, result in enumerate(all_results):
        print(f"\n--- Đánh giá Chunk {i+1}/{len(all_results)} (Dialogue: {result['dialogue_id']}, Index: {result['chunk_index']}) ---")
        
        transcript = result["transcript"]
        ground_truth = result["ground_truth"]["summary"]
        
        # Check if ground truth is valid
        if isinstance(ground_truth, dict) and "error" in ground_truth:
            print(f"Bỏ qua chunk vì ground truth có lỗi: {ground_truth['error']}")
            continue

        test_cases = []
        model_names_in_chunk = []
        for model_name, model_data in result["models"].items():
            model_summary = model_data["summary"]
            # Check if model summary is valid
            if "error" in model_summary:
                 print(f"Bỏ qua model '{model_name}' vì có lỗi: {model_summary}")
                 continue

            test_cases.append(LLMTestCase(
                input=transcript,
                actual_output=model_summary,
                expected_output=ground_truth,
                retrieval_context=[transcript]
            ))
            model_names_in_chunk.append(model_name)
            model_durations[model_name].append(model_data["duration"])

        if not test_cases:
            print("Không có model hợp lệ nào trong chunk này để đánh giá.")
            continue
            
        # Run evaluation
        print(f"Đánh giá các models: {model_names_in_chunk}")
        try:
            eval_results = evaluate(test_cases=test_cases, metrics=metrics, run_async=False)
            
            for i, test_result in enumerate(eval_results.test_results):
                model_name = model_names_in_chunk[i]
                for metric_data in test_result.metrics_data:
                    score_key = metric_data.name.lower().replace(" ", "_")
                    evaluation_scores[model_name][score_key].append(metric_data.score)

        except Exception as e:
            print(f"Lỗi trong quá trình đánh giá của deepeval: {e}")
            continue

    # --- Final Report ---
    print("\n\n" + "="*20 + " BÁO CÁO TỔNG KẾT " + "="*20)
    
    for model_name in model_durations.keys():
        scores = evaluation_scores[model_name]
        avg_duration = sum(model_durations[model_name]) / len(model_durations[model_name]) if model_durations[model_name] else 0
        
        print(f"\n--- Mô hình: {model_name} ---")
        print(f"  - Tổng số chunks được đánh giá: {len(scores.get('contextual_precision', []))}")
        print(f"  - Thời gian tạo tóm tắt trung bình: {avg_duration:.2f}s")

        if scores:
            for metric, values in scores.items():
                avg_score = sum(values) / len(values) if values else 0
                print(f"  - {metric.replace('_', ' ').title()} trung bình: {avg_score:.3f}")
        else:
            print("  - Không có điểm đánh giá nào được ghi nhận.")

if __name__ == "__main__":
    if not os.path.exists(".deepeval"):
        with open(".deepeval", "w") as f:
            f.write('{"USE_AZURE_OPENAI": false}')
    main() 