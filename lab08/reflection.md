# Reflection

Sau LAB 07, chúng ta đã có thể tự xây dựng một Transformer hoàn chỉnh gồm self-attention, feed-forward, residual connection và layer normalization. Tuy vậy, trong thực tế gần như không ai bắt đầu một bài toán mới bằng việc khởi tạo ngẫu nhiên toàn bộ các tham số đó. LAB 08 cho thấy các lý do cụ thể.

Lý do đầu tiên là dữ liệu. Một Transformer khởi tạo ngẫu nhiên phải học mọi thứ từ nhãn của tác vụ đích, từ nghĩa của từ, cú pháp cho đến quan hệ giữa các câu. Dữ liệu có nhãn cho một tác vụ như phân loại cảm xúc thường chỉ có từ vài nghìn đến vài chục nghìn mẫu. Pretraining dùng lượng văn bản không nhãn lớn hơn nhiều thông qua mục tiêu như masked language modeling, nên khi dữ liệu downstream giảm xuống còn 10 phần trăm, mô hình pretrained thường mất ít hơn mô hình học từ đầu.

Lý do thứ hai là pretrained representation. Sau pretraining, mỗi token có một biểu diễn theo ngữ cảnh đã mã hóa nhiều thông tin về ngôn ngữ. Đó là lý do một encoder đóng băng kèm một lớp tuyến tính vẫn phân loại được. Fine-tuning toàn bộ tiếp tục điều chỉnh biểu diễn theo tác vụ đích, và khoảng cách giữa hai chế độ cho biết biểu diễn có sẵn phù hợp với tác vụ đến mức nào.

Lý do thứ ba là computational cost. Pretraining một mô hình lớn cần hàng nghìn giờ GPU, mức chi phí một nhóm nhỏ hoặc một sinh viên không thể gánh. Dùng lại mô hình đã pretrain biến chi phí đó thành chi phí dùng chung, còn phần của người dùng chỉ là vài epoch fine-tuning cho một downstream task cụ thể. Chi phí này còn phụ thuộc vào độ dài chuỗi, vì attention tăng theo bình phương T, nên chọn max length cũng là một quyết định về chi phí.

Cuối cùng là limitation và domain shift. Pretrained không có nghĩa là hiểu mọi domain. Khi chuyển từ review phim sang review sản phẩm, từ vựng, phong cách và cách bộc lộ cảm xúc đổi khác, nên hiệu năng có thể giảm. Confidence cao cũng không bảo đảm dự đoán đúng, vì mô hình vẫn có thể sai một cách tự tin khi gặp phủ định, mỉa mai hoặc biểu thức đặc thù của domain.

Vì vậy, quy trình hợp lý là bắt đầu từ một mô hình pretrained, fine-tune trên dữ liệu của tác vụ, rồi kiểm tra bằng nhiều chỉ số, phân tích lỗi và đánh giá ngoài domain. Một kết quả NLP chỉ có ý nghĩa khi đi kèm dữ liệu, cấu hình, bằng chứng và phần diễn giải, chứ không chỉ là một con số accuracy.
