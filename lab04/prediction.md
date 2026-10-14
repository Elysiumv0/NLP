
| ID | Prediction | Reason | Confidence |
|---|---|---|---|
| P1 | TF-IDF + Logistic Regression sẽ là baseline mạnh cho IMDb. | Sentiment reviews thường có từ và cụm từ trực tiếp biểu thị cảm xúc; TF-IDF giữ được tín hiệu lexical và Logistic Regression phù hợp với feature space sparse. | 90% |
| P2 | Average Word2Vec không chắc chắn tốt hơn TF-IDF. | Vector trung bình giữ một phần quan hệ ngữ nghĩa nhưng làm mất thứ tự từ và có thể làm loãng các từ phủ định hoặc cụm từ quan trọng. | 80% |
| P3 | Tăng vocabulary size không bảo đảm accuracy tăng. | Feature mới có thể thêm tín hiệu nhưng cũng có thể thêm từ hiếm, nhiễu và tăng chi phí tính toán. | 99% |
| P4 | Bỏ stopwords không bảo đảm performance tăng. | Một số từ thường bị xem là stopwords, đặc biệt từ phủ định như `not`, có thể mang tín hiệu sentiment. | 90% |
| P5 | Accuracy và F1 có thể dẫn đến kết luận khác nhau. | Accuracy đo tỷ lệ dự đoán đúng tổng thể; F1 cân bằng precision và recall của positive class. | 99% |

