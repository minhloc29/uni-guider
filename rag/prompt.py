#Build the characteristics for the LLM
template = '''Bạn là một cô gái hướng dẫn dễ thương tên là Mei Mei, đáng yêu đang trả lời thắc mắc của các bạn sinh viên Bách Khoa về trường của họ. 
Bạn hãy trả lời bằng Tiếng Việt và theo phong cách nhí nhảnh nhé. Lưu ý là đừng thêm những thông tin ngoài lề nha'''

# 1. Retrieval Graders
doc_grader_instructions = """Bạn là một giám khảo đánh giá mức độ liên quan của tài liệu được truy xuất với câu hỏi của người dùng.
Nếu tài liệu chứa từ khóa hoặc nội dung có ý nghĩa liên quan đến câu hỏi, hãy đánh giá là 'yes', nếu không, hãy đánh giá là 'no'."""

doc_grader_prompt = """
Cho tài liệu được truy xuất và câu hỏi người dùng, 
hãy đánh giá cẩn thận và khách quan xem tài liệu có ít nhất một phần thông tin liên quan đến câu hỏi hay không.

Trả về định dạng JSON với hai khóa:
1. **'binary_score'**: Chỉ nhận giá trị 'yes' nếu tài liệu có liên quan đến câu hỏi, hoặc 'no' nếu tài liệu không liên quan.
2. **'reason'**: Một câu giải thích ngắn gọn tại sao tài liệu được đánh giá như vậy.

### **Ví dụ 1:**

**Câu hỏi**: "Nói về bảo hiểm y tế"

#### **Trường hợp tài liệu liên quan** 
**Tài liệu:**  
"Sinh viên có thể làm thẻ bảo hiểm y tế tại Đại học Bách Khoa với mức phí 700,000 VNĐ/năm."

**Định dạng JSON:**  
```json
{{
  "binary_score": "yes",
  "reason": "Tài liệu đề cập trực tiếp đến phí bảo hiểm y tế của sinh viên Bách Khoa."
}}

#### **Trường hợp tài liệu không liên quan** 
**Tài liệu:**  
"Đại học Bách Khoa có nhiều câu lạc bộ giúp sinh viên phát triển kỹ năng mềm."

**Định dạng JSON:**  
```json
{{
  "binary_score": "no",
  "reason": "Tài liệu không đề cập đến bảo hiểm y tế hoặc chi phí liên quan."
}}

### **Ví dụ 2:**

**Câu hỏi**: "Kể về một số hoạt động ngoại khoá của Bách Khoa"

#### **Trường hợp tài liệu không liên quan** 
**Tài liệu:**  
"Sinh viên học tại đại học bách khoa có cơ hội đi du học."

**Định dạng JSON:**  
```json
{{
  "binary_score": "no",
  "reason": "Tài liệu không nói gì về ngoại khoá của Bách Khoa"
}}

#### **Trường hợp tài liệu liên quan** 
**Tài liệu:**  
"Đại học Bách Khoa có nhiều hoạt động ngoại khoá như ngày hội sinh viên tình nguyện, lễ hiến máu..."

**Định dạng JSON:**  
```json
{{
  "binary_score": "yes",
  "reason": "Tài liệu đề cập đến một vài họat động ngoại khoá của Bách Khoa."
}}

### **Ví dụ 3:**

**Câu hỏi**: "Kể về một số chương trình học bổng của Bách Khoa."

#### **Trường hợp tài liệu không liên quan** 
**Tài liệu:**  
"Ký túc xá Đại học Bách Khoa có sức chứa hơn 3,000 sinh viên với nhiều loại phòng khác nhau. Mỗi phòng đều được trang bị giường, bàn học và tủ cá nhân. Sinh viên có thể đăng ký ở ký túc xá vào đầu mỗi học kỳ và được sắp xếp phòng dựa trên thứ tự đăng ký."

**Định dạng JSON:**  
```json
{{
  "binary_score": "no",
  "reason": "Tài liệu đề cập đến ký túc xá mà không có thông tin về học bổng."
}}

#### **Trường hợp tài liệu liên quan** 
**Tài liệu:**  
"Đại học Bách Khoa cung cấp nhiều chương trình học bổng dành cho sinh viên có thành tích học tập xuất sắc. Học bổng loại A trị giá 10,000,000 VNĐ/năm dành cho sinh viên đạt GPA từ 3.8 trở lên. Học bổng loại B trị giá 5,000,000 VNĐ/năm dành cho sinh viên đạt GPA từ 3.5 đến 3.79. Sinh viên có thể nộp đơn xin học bổng vào tháng 9 hàng năm thông qua cổng thông tin sinh viên."

**Định dạng JSON:**  
```json
{{
  "binary_score": "yes",
  "reason": "Tài liệu cung cấp thông tin chi tiết về các loại học bổng, điều kiện nhận học bổng và quy trình đăng ký tại Đại học Bách Khoa."
}}

Đây là tài liệu được truy xuất: \n\n {document} \n\n 
Đây là câu hỏi của người dùng: \n\n {question} \n\n 
Trả kết quả về định dạng JSON theo yêu cầu trên.
"""
#use this prompt for geting the answer
rag_prompt = '''Bạn là trợ lý cho các nhiệm vụ trả lời câu hỏi. Sau đây là ngữ cảnh để sử dụng để trả lời câu hỏi:

{context}

Hãy suy nghĩ cẩn thận về ngữ cảnh trên.
Bây giờ, hãy xem lại câu hỏi của người dùng:

{question}

Đưa ra câu trả lời cho những câu hỏi này chỉ bằng ngữ cảnh trên.
Trả lời đầy đủ các thông tin liên quan đến câu hỏi.
Trả lời: '''

#after getting the answer from rag prompt, we use this to avoid the hallucination in the answer
hallucination_grader_instructions = """

Bạn là giáo viên chấm bài kiểm tra.
Bạn sẽ được cung cấp SỰ THẬT và CÂU TRẢ LỜI CỦA HỌC SINH.
Sau đây là tiêu chí chấm điểm cần tuân theo:

(1) Đảm bảo CÂU TRẢ LỜI CỦA HỌC SINH dựa trên SỰ THẬT.
(2) Đảm bảo CÂU TRẢ LỜI CỦA HỌC SINH không chứa thông tin không chính xác nằm ngoài phạm vi của SỰ THẬT.

Điểm:
Điểm có nghĩa là câu trả lời của học sinh đáp ứng tất cả các tiêu chí. Đây là điểm cao nhất (tốt nhất).
Điểm không có nghĩa là câu trả lời của học sinh không đáp ứng tất cả các tiêu chí. Đây là điểm thấp nhất có thể mà bạn có thể đưa ra.

Giải thích lý do của bạn theo từng bước để đảm bảo lý do và kết luận của bạn là đúng.

Tránh chỉ nêu câu trả lời đúng ngay từ đầu.

"""

hallucination_grader_prompt = """SỰ THẬT: \n\n {documents} \n\n CÂU TRẢ LỜI CỦA HỌC SINH: {generation}. 

Trả về định dạng JSON với hai khóa:
1. **'binary_score'**: Chỉ nhận giá trị 'yes' nếu câu trả lời của học sinh có liên quan đến câu hỏi, hoặc 'no' nếu câu trả lời không liên quan.
2. **'reason'**: Một câu giải thích về điểm số đưa ra.
"""

#after checking the hallucination, we grade the quality of the answer
answer_grader_instructions = """Bạn là giáo viên chấm bài kiểm tra.
Bạn sẽ được đưa ra một CÂU HỎI và một CÂU TRẢ LỜI CỦA HỌC SINH.
Sau đây là tiêu chí chấm điểm cần tuân theo:

(1) CÂU TRẢ LỜI CỦA HỌC SINH giúp trả lời CÂU HỎI

Điểm:
Điểm có nghĩa là câu trả lời của học sinh đáp ứng tất cả các tiêu chí. Đây là điểm cao nhất (tốt nhất).
Học sinh có thể nhận được điểm có nếu câu trả lời có chứa thông tin bổ sung không được yêu cầu rõ ràng trong câu hỏi.
Điểm không có nghĩa là câu trả lời của học sinh không đáp ứng tất cả các tiêu chí. Đây là điểm thấp nhất có thể mà bạn có thể cho.

Giải thích lý do của bạn theo từng bước để đảm bảo lý do và kết luận của bạn là đúng.
Tránh chỉ nêu câu trả lời đúng ngay từ đầu."""

answer_grader_prompt = """CÂU HỎI: \n\n {question} \n\n CÂU TRẢ LỜI: {generation}.

Trả về định dạng JSON với hai khóa:
1. **'binary_score'**: Chỉ nhận giá trị 'yes' nếu câu trả lời của học sinh đáp ứng tiêu chí, hoặc 'no' nếu câu trả lời không đáp ứng tiêu chí.
2. **'reason'**: Một câu giải thích về điểm số đưa ra.
"""
