# Predictions Before Experiment

Các dự đoán được ghi trước khi chạy notebook. Kết quả thực tế sẽ được đối chiếu trong `reflection.md` sau khi notebook hoàn tất.

| ID | Prediction | Reason | Confidence |
|---|---|---|---|
| P1 | TF-IDF + Logistic Regression sẽ là baseline mạnh cho IMDb. | Sentiment reviews thường có từ và cụm từ trực tiếp biểu thị cảm xúc; TF-IDF giữ được tín hiệu lexical và Logistic Regression phù hợp với feature space sparse. | High |
| P2 | Average Word2Vec không chắc chắn tốt hơn TF-IDF. | Vector trung bình giữ một phần quan hệ ngữ nghĩa nhưng làm mất thứ tự từ và có thể làm loãng các từ phủ định hoặc cụm từ quan trọng. | High |
| P3 | Tăng vocabulary size không bảo đảm accuracy tăng. | Feature mới có thể thêm tín hiệu nhưng cũng có thể thêm từ hiếm, nhiễu và tăng chi phí tính toán. | High |
| P4 | Bỏ stopwords không bảo đảm performance tăng. | Một số từ thường bị xem là stopwords, đặc biệt từ phủ định như `not`, có thể mang tín hiệu sentiment. | High |
| P5 | Accuracy và F1 có thể dẫn đến kết luận khác nhau. | Accuracy đo tỷ lệ dự đoán đúng tổng thể; F1 cân bằng precision và recall của positive class. | High |

## Quy ước so sánh

- Giữ nguyên training, validation và test split giữa các mô hình.
- Chỉ fit TF-IDF trên training data.
- Chỉ học Word2Vec từ training text.
- Không chọn mô hình chỉ dựa trên accuracy.
- Dùng confusion matrix và error analysis để kiểm tra dạng lỗi của baseline.
