# Reflection

## So sánh LAB 05 và LAB 06

RNN, LSTM và GRU tạo hidden state theo thứ tự thời gian. Representation cuối phải mang thông tin đã tích lũy qua các bước trước đó, nên thông tin xa có thể khó duy trì. Attention tạo trọng số trên các key rồi kết hợp value thành context vector. Trong self-attention, mỗi token có thể tương tác trực tiếp với các token khác trong cùng sequence, và các vị trí có thể được tính song song.

## Điều attention cải thiện

- Cung cấp cơ chế lựa chọn thông tin theo query thay vì chỉ dùng hidden state cuối.
- Tạo kết nối trực tiếp giữa các token trong sequence.
- Cho phép tính các vị trí song song trong bước attention.
- Tạo phân phối attention weights và context representation có thể kiểm tra bằng visualization.

## Giới hạn mới

Full self-attention tạo ma trận `T × T`, nên bộ nhớ và phép tính attention tăng theo `O(T²)` theo chiều dài sequence. Attention weights cũng không tự động là giải thích đầy đủ cho dự đoán, reasoning hay linguistic understanding. Chất lượng representation vẫn phụ thuộc vào embedding, dữ liệu, objective và cách huấn luyện.

## Reflection về thí nghiệm sentiment

Thí nghiệm dùng embedding đặc trưng cố định, một cửa sổ negation đơn giản, self-attention và Logistic Regression. Bộ huấn luyện template giúp chạy được end-to-end nhưng có độ đa dạng thấp. Tập thử thách chứa sarcasm, negation, từ tiếng lóng và các câu có nhiều mệnh đề; các hiện tượng này cho thấy model có thể dự đoán sai ngay cả khi pipeline attention hoạt động đúng về mặt tính toán.

Không nên kết luận attention luôn tốt hơn recurrent model. Cần kiểm tra cùng dữ liệu, cách chia train/test, số tham số, metric và thời gian huấn luyện để so sánh công bằng. Hướng tiếp theo là dùng embedding học được hoặc pretrained, tăng kích thước dataset, kiểm tra ablation không dùng attention và đánh giá riêng các hiện tượng negation/long-range dependency.
