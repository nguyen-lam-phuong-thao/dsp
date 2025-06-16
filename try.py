
from typing import Dict
from langchain_core.messages import AIMessage, HumanMessage
import time

# Measure time for Google LLM
start_time = time.time()
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

GOOGLE_API_KEY = "Key của m"
google_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.8,
)
ollama_llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.8,
)

response_goolge = google_llm.invoke(
    "What is the capital of France?",
)
google_execution_time = time.time() - start_time

# Measure time for Ollama LLM
start_time = time.time()
response_ollama = ollama_llm.invoke(
    "What is the capital of France?",
)
ollama_execution_time = time.time() - start_time

print("Google LLM Response:", response_goolge.content)
print(f"Google LLM Execution Time: {google_execution_time:.2f} seconds")

print("Ollama LLM Response:", response_ollama.content.split("</think>")[-1].strip())
print(f"Ollama LLM Execution Time: {ollama_execution_time:.2f} seconds")



def generate( state: Dict, llm) -> Dict:
		"""
		Generate a simple meeting note based on the transcript and meeting type.
		Uses the type-specific prompt to format the note appropriately.
		"""
		print(f'Generating simple meeting note for meeting type: {state.get("meeting_type", "general")}')

		meeting_type = state.get('meeting_type', 'general')
		transcript = state.get('transcript', '')
		custom_prompt = state.get('custom_prompt')
		if custom_prompt:
			print(f"Using custom prompt for meeting note generation of type '{meeting_type} custom prompt: {custom_prompt}'")
		else:
			print('Using default prompt for meeting note generation')
		if not transcript or len(transcript) < 50:
			print('Transcript too short to generate meaningful meeting note')
			return {
				**state,
				'meeting_note': 'Không đủ thông tin để tạo ghi chú cuộc họp.',
				'messages': state['messages'] + [HumanMessage(content='Tạo ghi chú cuộc họp'), AIMessage(content='Không đủ thông tin để tạo ghi chú cuộc họp.')],
			}

		meeting_note_prompt = """Tóm tắt nội dung cuộc họp dưới dạng các ý chính, dễ hiểu và đầy đủ. Ghi rõ các chủ đề được thảo luận, quyết định được đưa ra, và các việc cần làm sau cuộc họp (bao gồm deadline từng task nếu có đề cập).

Hãy đảm bảo nội dung tóm tắt:
1. Có cấu trúc rõ ràng, chia thành các phần nhỏ dễ đọc
2. Ngắn gọn nhưng đầy đủ thông tin quan trọng
3. Sử dụng văn phong chuyên nghiệp, dễ hiểu
4. Bỏ qua các nội dung không liên quan, tập trung vào các thông tin có giá trị, không tạo ra thông tin sai lệch, chỉ tóm tắt những gì đã được thảo luận
5. Nêu bật được các ý chính, quyết định quan trọng và nhiệm vụ cần thực hiện
6. Không sử dụng từ ngữ phức tạp hoặc thuật ngữ chuyên ngành mà không giải thích

CHÚ Ý: KHÔNG TẠO THÔNG TIN MỚI, CHỈ TÓM TẮT CÁC NỘI DUNG ĐÃ ĐƯỢC THẢO LUẬN TRONG CUỘC HỌP.

ĐỊNH DẠNG YÊU CẦU:
```
# [TÊN CUỘC HỌP]
## Thông tin chung
- **Ngày họp**: [ngày họp nếu có trong transcript]
- **Chủ đề**: [chủ đề chính của cuộc họp]
- **Người tham gia**: [danh sách người tham gia nếu có thể xác định]

## Tóm tắt nội dung
[Tóm tắt ngắn gọn 3-5 câu về nội dung chính của cuộc họp]

## Các chủ đề được thảo luận
1. [Chủ đề 1]
   - [Điểm chính]
   - [Điểm chính]
2. [Chủ đề 2]
   - [Điểm chính]
   - [Điểm chính]
...

## Quyết định quan trọng
- [Quyết định 1]
- [Quyết định 2]
...

## Công việc cần thực hiện
- [Công việc 1] - Người phụ trách: [Tên], Deadline: [Thời hạn nếu có]
- [Công việc 2] - Người phụ trách: [Tên], Deadline: [Thời hạn nếu có]
...
```
"""


		# Create the full prompt with transcript context
		# Prioritize custom prompt if available, otherwise use the default
		base_prompt = meeting_note_prompt
		if custom_prompt:
			base_prompt = f'{meeting_note_prompt}\n\nYêu cầu đặc biệt từ người dùng (ưu tiên cao nhất):\n{custom_prompt}'

		prompt = f"""{base_prompt}

    Dưới đây là transcript cuộc họp để bạn tham khảo:

    {transcript}

    Hãy tạo một ghi chú cuộc họp ngắn gọn, rõ ràng dựa trên nội dung transcript trên.
    Lưu ý: Nếu có yêu cầu đặc biệt từ người dùng, hãy ưu tiên tuân theo các yêu cầu đó trước tiên.
    Cố gắng xác định tên người nói dựa vào nội dung cuộc họp không nên dựa vào SPEAKER ID, và danh sách người tham gia là những người đang nói trong cuộc họp, khác với những người được đề cập trong nội dung cuộc họp.
    Nếu không thể xác định tên người nói, hãy sử dụng 'Người nói' làm tên mặc định.
    """

		try:
			# Track token usage

			# Start timing
			start_time = time.time()

			# Call LLM to generate meeting note
			response = llm.invoke(prompt)
			meeting_note = response.content

			# Track output tokens

			# Log performance
			duration = time.time() - start_time
			print(f'Generated meeting note in {duration:.2f} seconds')
			print(f'Meeting note length: {len(meeting_note)} characters')

			# Update state with new meeting note
			return {
				**state,
				'meeting_note': meeting_note.replace('```', '').strip(),
				'messages': state['messages'] + [HumanMessage(content='Tạo ghi chú cuộc họp'), AIMessage(content=f"Đã tạo ghi chú cuộc họp cho loại '{meeting_type}'")],
			}

		except Exception as e:
			print(f'Error generating meeting note: {str(e)}')
			return {
				**state,
				'meeting_note': 'Đã xảy ra lỗi khi tạo ghi chú cuộc họp.',
				'messages': state['messages'] + [HumanMessage(content='Tạo ghi chú cuộc họp'), AIMessage(content=f'Lỗi: {str(e)}')],
			}
   
# Example usage
state = {
	'meeting_type': 'general',
	'transcript': 'Hôm nay chúng ta sẽ thảo luận về các dự án hiện tại và các nhiệm vụ cần thực hiện trong tuần tới.',
	'custom_prompt': None,
	'messages': []
}
result = generate(state, ollama_llm)
print("Generated Meeting Note:")
print(result['meeting_note'])