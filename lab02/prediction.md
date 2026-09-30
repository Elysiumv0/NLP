1.

Prediction: Không
Reason: Vocab phụ thuộc vào corpus và cách tokenize, chọn n chỉ ảnh hưởng đến cách model nhìn câu.
Confidence: 100%

2.

Prediction: Số n gram khác nhau tăng.
Reason: Mỗi vị trí trong văn bản sinh ra 1 n gram, và số tổ hợp có thể tăng theo (V^n) trong khi corpus có hữu hạn token nên phần lớn tổ hợp dài chỉ gặp 1 lần.
Confidence: 70%

3.

Prediction: Trigram
Reason: Vì context length càng dài càng khó xuất hiện những cụm 3 trong train.
Confidence: 100%

4.

Prediction: Trigram
Reason: Perplexity cao hơn đồng nghĩa với xác suất thấp hơn, tức có nghĩa là trigram.
Confidence: 80%

5.

Prediction: Không
Reason: Vì nhiều trigram không xuất hiện trong train, và nếu nhiều context hơn nhưng ít dữ liệu cho mỗi context thì ước lượng cũng không đáng tin.
Confidence: 90%
