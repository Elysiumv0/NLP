# Prediction before experiment

## Prediction 1

Model có thể chọn sai span khi context chứa nhiều từ trùng với question. Lexical overlap cao chưa đảm bảo span đó thực sự trả lời câu hỏi. Model cần học quan hệ ngữ nghĩa giữa question và context.

## Prediction 2

Answer dài có thể làm boundary prediction khó hơn vì model phải xác định đầy đủ các thành phần của cụm từ. EM được dự đoán giảm ở các nhóm answer dài, trong khi F1 có thể vẫn cao nếu phần lớn token được dự đoán đúng.

## Prediction 3

Thêm thông tin không liên quan hoặc distractor có thể làm giảm EM và F1, đặc biệt khi distractor chứa từ khóa tương tự question. Context dài hơn cũng có thể thay đổi cách answer nằm trong các window.

## Prediction 4

Đảo thứ tự câu có thể ảnh hưởng prediction dù answer text không đổi. Vị trí token, quan hệ giữa các câu và thông tin tham chiếu có thể thay đổi. Model được dự đoán ít bị ảnh hưởng hơn nếu câu chứa answer vẫn độc lập về ngữ nghĩa.

## Kỳ vọng chung

Fine-tuned Transformer được kỳ vọng vượt baseline keyword overlap. Context length 256 hoặc 384 có thể giữ nhiều thông tin hơn 128 nhưng chi phí huấn luyện lớn hơn. Các giả thuyết cần được kiểm chứng bằng kết quả trên cùng test split.


# Đối chiếu kết quả

## Prediction 1

Nhóm lexical overlap cao đạt F1 78.01, cao hơn nhóm thấp với 65.18. Khi thêm distractor chứa question, EM giảm từ 62.13 xuống 47.33 và F1 giảm từ 73.73 xuống 55.40. Kết quả phù hợp với dự đoán rằng model có thể nhầm giữa những span chứa từ khóa tương tự.

## Prediction 2

Nhóm answer dài 2–3 token đạt EM 68.63, trong khi nhóm 7+ token chỉ đạt 42.11. F1 của nhóm 7+ vẫn đạt 65.29. Dự đoán phù hợp với xu hướng ở các answer dài nhưng EM không giảm đều từ nhóm 1 token.

## Prediction 3

Thêm thông tin không liên quan làm EM tăng nhẹ 0.33 điểm và F1 tăng 0.16 điểm. Dự đoán giảm điểm không được xác nhận trong trường hợp này. Distractor làm EM giảm 14.80 điểm và F1 giảm 18.33 điểm, phù hợp với dự đoán.

## Prediction 4

Đảo thứ tự câu làm EM giảm từ 62.22 xuống 60.90 và F1 giảm từ 73.90 xuống 72.14 trên 1440 mẫu. Kết quả phù hợp với dự đoán rằng model vẫn chịu ảnh hưởng của thứ tự context.

## Kết luận

DistilBERT vượt baseline rõ rệt. Max length 384 đạt F1 cao nhất nhưng max length 256 đạt EM cao nhất. Tăng context length không đảm bảo mọi metric đều cải thiện.
