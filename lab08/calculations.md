# Calculations

## Calculation Exercise 1 — MLM Loss

Loss cho một token bị mask, log là logarit tự nhiên:

L = −ln p(token đúng)

Với p(eats) = 0.70:

L = −ln(0.70) = 0.3567

| p(eats) | L = −ln p | Thay đổi so với p = 0.70 |
|---|---|---|
| 0.95 | 0.0513 | giảm 0.3054, khoảng 86% |
| 0.70 | 0.3567 | mốc so sánh |
| 0.01 | 4.6052 | tăng 4.2485, khoảng 12.9 lần |

**1. p(eats) = 0.95.** Loss giảm từ 0.3567 xuống 0.0513. Khi xác suất của token đúng tiến gần 1, loss tiến về 0 và gradient trên logit nhỏ dần, nên mô hình gần như không còn bị điều chỉnh ở mẫu này.

**2. p(eats) = 0.01.** Loss tăng từ 0.3567 lên 4.6052. Loss tăng rất nhanh khi xác suất gán cho token đúng tiến về 0, nên mẫu này tạo gradient lớn và kéo mô hình mạnh về phía token đúng.

**3. Vì sao cross-entropy phù hợp với MLM.**
- MLM là bài toán phân loại nhiều lớp trên toàn bộ từ vựng, đầu ra softmax là một phân phối xác suất trên các token.
- Với nhãn one-hot, cross-entropy bằng −log xác suất của token đúng, tức negative log-likelihood. Tối thiểu hóa nó là ước lượng hợp lý cực đại cho mô hình xác suất của token bị che.
- Gradient theo logit có dạng p − y, lớn khi dự đoán sai và nhỏ khi dự đoán đúng, không bị bão hòa như MSE đặt trên đầu ra softmax.
- Dự đoán sai nhưng tự tin bị phạt nặng hơn dự đoán sai nhưng không chắc chắn.
- Loss chỉ được tính tại các vị trí bị mask, và exp của loss trung bình cho perplexity.

## Calculation Exercise 2 — Tokenization

Câu: "The cats are playing."

Tokenizer là tokenizer của pretrained model, chạy trong `tokenization.ipynb`. Notebook cho ra:
- token sequence;
- token IDs;
- special tokens;
- attention mask;
- bảng so sánh số words, tokens và subword tokens.
