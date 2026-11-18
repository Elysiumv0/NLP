# Reflection — Sequence Classification và Token Classification

Một pretrained Transformer có thể phục vụ cả sequence classification và token classification vì phần quan trọng nhất của kiến trúc là bộ mã hóa ngữ cảnh, không phải lớp dự đoán cuối cùng. Khi nhận một câu, Transformer chuyển các token thành chuỗi vector có cùng kích thước ẩn. Các vector được cập nhật thông qua self-attention nên nội dung của mỗi vector phụ thuộc vào những token khác trong câu. Kết quả đó là một biểu diễn chung, giàu thông tin, có thể được khai thác cho nhiều mục tiêu dự đoán.

Ở sequence classification, nhiệm vụ yêu cầu một nhãn cho toàn bộ đầu vào. Với những mô hình có token đặc biệt CLS, vector của CLS có thể được đưa qua một classification head để tạo logits cho các lớp. Dù lựa chọn cách nào, mô hình chỉ cần tạo một quyết định cho cả câu. Loss được tính dựa trên nhãn của chuỗi.

Ở token classification, nhiệm vụ yêu cầu một nhãn cho từng token. Named Entity Recognition là ví dụ điển hình. Mỗi vector ngữ cảnh của token được đưa qua cùng một lớp tuyến tính để tạo logits cho các nhãn BIO. Vì vậy, model có thể dự đoán B-PER cho tên riêng, I-PER cho phần tiếp theo và O cho token không thuộc thực thể. Loss được tính tại nhiều vị trí trong chuỗi thay vì chỉ một vị trí đại diện.

Khác biệt giữa hai bài toán chủ yếu nằm ở cách lấy biểu diễn để dự đoán và cách tổ chức nhãn mục tiêu. Với cùng hidden dimension, sequence classification dùng một biểu diễn cấp câu, còn token classification dùng toàn bộ chuỗi biểu diễn. Phần encoder đã học cú pháp, ngữ nghĩa và quan hệ ngữ cảnh từ quá trình tiền huấn luyện vẫn hữu ích trong cả hai trường hợp. Khi fine-tune, encoder tiếp tục điều chỉnh theo yêu cầu của nhiệm vụ mới.

Token classification còn có một thách thức kỹ thuật riêng: nhãn của dataset NER thường được gắn ở mức word, trong khi tokenizer có thể chia một word thành nhiều subword. Nếu dùng chiến lược gán nhãn cho subword đầu tiên, các vị trí còn lại nhận ignore index để không tham gia trực tiếp vào loss. Tuy nhiên, chúng vẫn góp phần tạo biểu diễn ngữ cảnh thông qua self-attention. Điều này cho thấy khả năng dùng chung encoder không có nghĩa là mọi nhiệm vụ dùng chung hoàn toàn một pipeline huấn luyện.

Kết luận, Transformer đóng vai trò bộ tạo biểu diễn linh hoạt. Chỉ bằng việc thay đổi prediction head, lựa chọn vector cần phân loại và định nghĩa loss phù hợp, ta có thể chuyển từ một nhãn cho câu sang nhiều nhãn cho các token mà không phải xây dựng lại mô hình ngôn ngữ từ đầu.
