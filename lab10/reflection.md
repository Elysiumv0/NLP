# Reflection

LAB 08, LAB 09 và LAB 10 đều sử dụng pretrained Transformer để khai thác biểu diễn ngữ cảnh đã học từ dữ liệu lớn. Điểm khác biệt chính nằm ở đơn vị dự đoán, prediction head và cách xây dựng target cho từng bài toán.

```text
                    Pretrained Transformer
                              │
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
 [CLS] representation     Token reps            Token reps
        ↓                     ↓                     ↓
 Classification             NER                Start / End
        ↓                     ↓                     ↓
      LAB 08                LAB 09                LAB 10
```

Trong LAB 08, classification dự đoán một nhãn cho toàn bộ văn bản. Biểu diễn CLS tổng hợp thông tin của input và được đưa qua một lớp tuyến tính để tạo logits tương ứng với các lớp. Target là nhãn của từng văn bản. Mô hình cần hiểu nội dung tổng thể để phân biệt các nhóm dữ liệu.

Trong LAB 09, NER xác định thực thể tại từng vị trí token. Prediction head nhận biểu diễn của tất cả token và tạo phân phối xác suất trên tập nhãn NER. Target là chuỗi nhãn tương ứng với token. Việc căn chỉnh nhãn từ word sang subword rất quan trọng vì tokenizer có thể chia một từ thành nhiều phần. Mô hình không chỉ nhận biết loại thực thể mà còn cần xác định biên của thực thể.

Trong LAB 10, extractive QA nhận đồng thời question và context. Mỗi token có hai logits dùng để xác định khả năng bắt đầu và kết thúc đáp án. Target gồm start position và end position. Vì dataset lưu đáp án theo vị trí ký tự, offset mapping được sử dụng để chuyển character span sang token span. Sliding window giúp xử lý context dài và giữ lại thông tin có khả năng chứa đáp án. EM đánh giá sự trùng khớp hoàn toàn, còn F1 phản ánh mức độ trùng lặp giữa prediction và gold.

Qua ba bài lab, có thể thấy backbone Transformer cung cấp biểu diễn ngữ cảnh dùng chung nhưng prediction head phải phù hợp với cấu trúc output. Classification dự đoán ở cấp văn bản, NER dự đoán ở cấp token, còn QA dự đoán hai biên của một span. Sự khác biệt này dẫn đến cách chuẩn bị dữ liệu, tính loss và lựa chọn metric khác nhau. Fine-tuning giúp điều chỉnh cả backbone và prediction head theo mục tiêu cụ thể. Extractive QA cũng cho thấy giới hạn của việc trích xuất đáp án khi context không chứa thông tin cần thiết. Đây là cơ sở để tiếp tục nghiên cứu retrieval và RAG.


Kết quả LAB 10 cho thấy DistilBERT đạt EM 62.13 và F1 73.73, vượt xa baseline keyword overlap. Tuy nhiên, answer dài và distractor vẫn gây khó khăn. Nhóm answer từ 7 token trở lên chỉ đạt EM 42.11. Khi thêm distractor, F1 giảm còn 55.40. Sự khác biệt giữa EM và F1 cũng giúp đánh giá các trường hợp model dự đoán đúng một phần đáp án nhưng chưa xác định chính xác toàn bộ span.
