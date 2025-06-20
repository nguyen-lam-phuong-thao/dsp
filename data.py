import re
import json
from datetime import datetime
from typing import List, Dict
import pandas as pd


class MeetingProcessor:
    def __init__(self):
        # Pattern cải tiến để bắt nhiều định dạng timestamp khác nhau
        self.speaker_pattern = re.compile(
            r"([A-Za-zÀ-ỹ\s]+|SPEAKER_\d+)\s*[\[\(](\d{2}[/-]\d{2}[/-]\d{4}(?:,|\s|,?\s)\d{1,2}:\d{2}(?: ?(?:AM|PM|CH|SA))?|(?:AM|PM|CH|SA)? ?\d{1,2}:\d{2} (?:AM|PM|CH|SA) \d{2}/\d{2}/\d{4})[\]\)]"
        )
        self.vn_time_replace = {"CH": "PM", "SA": "AM"}

    def _normalize_timestamp(self, ts: str) -> str:
        # Chuẩn hóa các định dạng thời gian khác nhau
        ts = ts.replace(",", "").strip()
        for k, v in self.vn_time_replace.items():
            ts = ts.replace(k, v)
        
        # Xử lý các định dạng khác nhau
        try:
            if re.match(r"\d{2}/\d{2}/\d{4} \d{1,2}:\d{2} [AP]M", ts):
                return datetime.strptime(ts, "%d/%m/%Y %I:%M %p").isoformat()
            elif re.match(r"\d{2}/\d{2}/\d{4} \d{1,2}:\d{2}", ts):
                return datetime.strptime(ts, "%d/%m/%Y %H:%M").isoformat()
            elif re.match(r"[AP]M \d{1,2}:\d{2} \d{2}/\d{2}/\d{4}", ts):
                return datetime.strptime(ts, "%p %I:%M %d/%m/%Y").isoformat()
            else:
                return ts
        except Exception:
            return ts

    def extract_dialogues(self, text: str) -> List[Dict]:
        dialogues = []
        # Tách nội dung trước khi xử lý để tránh phân tích nhầm
        main_content = re.split(r"### [A-Z]. THÔNG TIN|### [A-Z]. CÁC ĐIỂM CHÍNH|### [A-Z]. QUYẾT ĐỊNH & HÀNH ĐỘNG|### [A-Z]. THEO DÕI", text)[0]
        
        # Tìm tất cả các đoạn hội thoại
        matches = list(self.speaker_pattern.finditer(main_content))
        
        for i, match in enumerate(matches):
            speaker = match.group(1).strip()
            timestamp = self._normalize_timestamp(match.group(2))
            
            # Xác định nội dung giữa các speaker
            if i < len(matches) - 1:
                content = main_content[match.end():matches[i+1].start()].strip()
            else:
                content = main_content[match.end():].strip()
            
            # Loại bỏ các dòng trống và ký tự không cần thiết
            content = re.sub(r"\n+", " ", content).strip()
            if content:
                dialogues.append({
                    "speaker": speaker,
                    "timestamp": timestamp,
                    "content": content
                })
                
        return dialogues

    def extract_sections(self, text: str, start_marker: str) -> str:
        match = re.search(rf"{start_marker}(.+?)(?=###|##|\Z)", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_bullet_lines(self, text: str, marker: str) -> List[str]:
        section = self.extract_sections(text, marker)
        return [line.strip("-*+ ").strip() for line in section.splitlines() if line.strip().startswith(("-", "*", "+"))]

    def process_meeting_text(self, meeting_id: int, raw_text: str) -> Dict:
        # Xử lý phần metadata
        meta = {
            "date": self.extract_sections(raw_text, r"#### 1\. (?:Ngày|Date):"),
            "attendees": re.findall(r"- (.+)", self.extract_sections(raw_text, r"#### 2\. (?:Người tham dự|Attendance):")),
            "agenda": self.extract_sections(raw_text, r"#### 3\. (?:Nội dung chính|Agenda Outline):"),
            "tone": self.extract_sections(raw_text, r"#### 4\. (?:Không khí/Tình hình cuộc họp|Meeting Tone/Atmosphere):")
        }

        # Xử lý phần summary với các trường hợp tiếng Việt
        summary = {
            "facts": self._extract_bullet_lines(raw_text, r"#### 6\. (?:Tóm tắt|Summary):\n\s*- (?:Sự kiện|Facts):"),
            "problems": self._extract_bullet_lines(raw_text, r"- (?:Vấn đề|Problems):"),
            "solutions": self._extract_bullet_lines(raw_text, r"- (?:Giải pháp được đề xuất|Solutions Proposed):")
        }

        # Xử lý các phần khác với cả tiếng Anh và tiếng Việt
        decisions = self._extract_bullet_lines(raw_text, r"#### 7\. (?:Danh sách các quyết định đã đưa ra|List of Decisions Made):")
        action_items = self._extract_bullet_lines(raw_text, r"#### 8\. (?:Các hành động theo từng người tham dự|Action Items by Attendee):")
        questions = self._extract_bullet_lines(raw_text, r"#### 9\. (?:Các câu hỏi được đặt ra|Questions Raised):")
        next_steps = self._extract_bullet_lines(raw_text, r"#### 10\. (?:Các bước tiếp theo|Next Steps):")

        dialogue = self.extract_dialogues(raw_text)

        return {
            "meeting_id": meeting_id,
            "meta": meta,
            "dialogue": dialogue,
            "summary": summary,
            "decisions": decisions,
            "action_items": action_items,
            "questions": questions,
            "next_steps": next_steps
        }

    def process_meeting_texts(self, records: List[Dict], text_key: str = "text_content", meeting_id_key: str = "meeting_id") -> List[Dict]:
        results = []
        for i, row in enumerate(records):
            meeting_id = row.get(meeting_id_key, i + 1)
            text = str(row.get(text_key, ""))
            results.append(self.process_meeting_text(meeting_id, text))
        return results


def main():
    input_csv = "dsp/sql.csv"  # Đường dẫn file input
    output_json = "dsp/processed_meetings.json"  # Đường dẫn file output

    df = pd.read_csv(input_csv, sep=";")
    processor = MeetingProcessor()

    df = df.reset_index().rename(columns={"index": "meeting_id"})

    processed = processor.process_meeting_texts(
        df.to_dict(orient="records"),
        text_key="text_content",
        meeting_id_key="meeting_id"
    )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"Đã xử lý {len(processed)} cuộc họp và lưu vào: {output_json}")


if __name__ == "__main__":
    main()
