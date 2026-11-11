# Prediction Before Experiment

## Prediction 1

**Prediction:** Có. Transformer pretrained được fine-tune trên IMDb đạt accuracy và F1 cao hơn Transformer nhỏ train from scratch trên cùng dữ liệu.

**Reason:** Mô hình from scratch phải học từ vựng, nghĩa của từ, cú pháp và tín hiệu cảm xúc chỉ từ vài nghìn review có nhãn. Mô hình pretrained đã học các biểu diễn theo ngữ cảnh từ lượng văn bản không nhãn rất lớn bằng masked language modeling, nên fine-tuning chủ yếu điều chỉnh biểu diễn có sẵn cho tác vụ phân loại.

**Expected evidence:** Khoảng cách accuracy và F1 giữa hai mô hình lớn hơn mức dao động giữa các seed, và mô hình pretrained hội tụ nhanh hơn trên validation. Chi phí huấn luyện và số tham số của mô hình pretrained cao hơn.

## Prediction 2

**Prediction:** C. Frozen encoder cho kết quả kém hơn fine-tuning toàn bộ.

**Reason:** Với encoder đóng băng, chỉ lớp tuyến tính trên biểu diễn [CLS] được huấn luyện. Biểu diễn này được tối ưu cho mục tiêu pretraining chứ không cho phân loại cảm xúc, nên thông tin về cảm xúc có thể có mặt nhưng không được sắp xếp để một ranh giới tuyến tính tách tốt. Fine-tuning cho phép toàn bộ tầng attention và feed-forward thích nghi, nên có thêm khả năng biểu diễn dành cho tác vụ. Frozen vẫn phải khá hơn mức ngẫu nhiên nhiều, vì biểu diễn pretrained chứa sẵn thông tin ngữ nghĩa.

**Expected evidence:** Accuracy và F1 của frozen thấp hơn fine-tuning, thời gian huấn luyện mỗi epoch và bộ nhớ của frozen thấp hơn, số tham số trainable chỉ gồm lớp tuyến tính.

## Prediction 3

**Prediction:** Mô hình train from scratch giảm performance mạnh hơn khi dữ liệu giảm từ 100% xuống 10%.

**Reason:** Pretraining chuyển phần lớn việc học ngôn ngữ sang giai đoạn không cần nhãn. Khi dữ liệu downstream còn ít, mô hình pretrained chỉ cần học ánh xạ từ biểu diễn có sẵn sang nhãn, một bài toán có ít tham số cần xác định hơn nhiều. Mô hình from scratch phải học đồng thời biểu diễn và bộ phân loại từ ít mẫu, nên dễ overfit và thiếu bao phủ từ vựng.

**Expected evidence:** Đường data-efficiency của mô hình pretrained phẳng hơn từ 100% xuống 10%, còn đường của mô hình from scratch dốc hơn và khoảng cách giữa hai đường rộng ra ở 10%.
