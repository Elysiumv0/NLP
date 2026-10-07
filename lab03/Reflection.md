# Reflection — Từ Word2Vec đến Transformer (LAB 03)

## 1. Bảng so sánh các biểu diễn

| Representation | Context-dependent? | Sparse/Dense | Một từ có nhiều vector? |
|---|---|---|---|
| TF-IDF | Không. Trọng số tính từ thống kê toàn corpus (TF trong văn bản × IDF), không phụ thuộc cách từ được dùng trong câu cụ thể. Vector là của văn bản, không mã hoá nghĩa của từ. | Sparse (|V| chiều, hầu hết bằng 0) | Không (mỗi từ là một cột cố định) |
| Co-occurrence | Không. Context chỉ được dùng một lần để xây vector; vector tạo xong là cố định cho mọi câu. | Sparse (|V|×|V|); có thể nén thành dense bằng PPMI+SVD | Không |
| Word2Vec | Không. Một từ → đúng một vector, dùng lại ở mọi câu. | Dense (vd. 100–300 chiều) | Không — mỗi từ đúng một vector |
| Contextual embedding) | Có. Vector của từ được tính lại cho từng câu từ toàn bộ ngữ cảnh. | Dense | Có — cùng một từ, vector khác nhau theo từng câu |


## 2. Tại sao `bank` cần contextual representation?

Từ `bank` có (ít nhất) hai nghĩa không liên quan — *ngân hàng* và *bờ sông* — như trong *I deposited money in the bank* và *We sat on the river bank*. Word2Vec tĩnh chỉ có một vector cho `bank`, nên vector đó buộc phải là một dạng trung bình có trọng số theo tần suất của mọi ngữ cảnh mà từ xuất hiện. Hệ quả: vector không đại diện đúng cho nghĩa nào trong một câu cụ thể.
