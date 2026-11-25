# Theory and calculations

## 1. Extractive QA

Input của mô hình gồm question và context. Output là hai phân phối xác suất trên các vị trí token trong input. Mô hình chọn một span liên tục trong context có start không lớn hơn end.

| Task | Input | Representation | Output |
|---|---|---|---|
| Classification | Text | CLS | Một nhãn cho văn bản |
| NER | Text | Token representations | Một nhãn cho mỗi token |
| Extractive QA | Question và context | Token representations | Start và end position |

## 2. Calculation Exercise 1

| Token index | Token |
|---:|---|
| 0 | The |
| 1 | capital |
| 2 | of |
| 3 | France |
| 4 | is |
| 5 | Paris |
| 6 | . |

Answer: Paris

- Start = 5
- End = 5

Answer: capital of France

- Start = 1
- End = 3

## 3. Calculation Exercise 2

Xác suất tại start position đúng là 0.60.

L_start = −ln 0.60 = 0.510826

L_end = −ln p_end

Đề bài không cung cấp xác suất tại end position đúng nên chưa thể tính giá trị cụ thể của L_end. Nếu p_end = 0.60 thì L_end = 0.510826.

## 4. QA Loss

L = L_start / 2 + L_end / 2

Nếu cả hai xác suất đúng đều bằng 0.60 thì L = 0.510826.

Cần hai prediction vì đáp án có thể gồm nhiều token. Start xác định điểm bắt đầu và end xác định điểm kết thúc. Một prediction duy nhất không đủ để xác định cả hai biên của span có độ dài thay đổi.

## 5. Calculation Exercise 3

Context: The capital of France is Paris.

| Thành phần | Giá trị |
|---|---|
| Ký tự đầu của Paris | 25 |
| Ký tự cuối của Paris | 29 |
| Character span | [25, 30) |
| Token | Paris |
| Start token trong context | 5 |
| End token trong context | 5 |

Character span có điểm cuối không bao gồm ký tự tại index 30. Token span sử dụng cả hai đầu. Với input ghép question và context theo DistilBERT, Paris nằm tại token index 14 vì có thêm CLS, question và SEP ở phía trước.

## 6. Exact Match và F1

Gold: Paris France

Prediction: France

Sau normalization, gold có 2 token và prediction có 1 token. Số token trùng nhau là 1.

Precision = 1/1 = 1

Recall = 1/2 = 0.5

F1 = 2 × 1 × 0.5 / 1.5 = 0.666667

EM = 0

F1 phản ánh mức độ trùng lặp từng phần, còn EM yêu cầu toàn bộ đáp án sau normalization trùng nhau.

## 7. Tokenization và sliding window

Input IDs biểu diễn token dưới dạng chỉ số trong vocabulary. Attention mask phân biệt token thực với padding. Token type IDs phân biệt question và context ở các mô hình có sử dụng segment embedding. DistilBERT không sử dụng token type IDs.

Offset mapping lưu vị trí ký tự đầu và cuối của mỗi token trong chuỗi gốc. Overflow to sample mapping xác định window thuộc sample nào. Sequence IDs phân biệt token của question và context, tránh căn chỉnh answer vào question.

Khi context dài hơn giới hạn input, sliding window chia context thành nhiều đoạn có phần chồng lấn. Stride là số token chồng lấn. Window không chứa đầy đủ answer được gán start và end tại CLS trong quá trình huấn luyện.

Tăng max sequence length không luôn cải thiện QA. Context dài giúp giữ thông tin liên quan nhưng có thể thêm distractor và tăng chi phí tính toán. Self-attention thông thường có độ phức tạp bậc hai theo chiều dài chuỗi.
