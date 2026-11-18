# Dự đoán trước khi chạy thí nghiệm

## Prediction 1. Baseline và pretrained Transformer

**Giả thuyết:** Transformer đạt entity-level F1 cao hơn Logistic Regression với đặc trưng token thủ công trên cùng test set CoNLL-2003.

**Lý do:** Embedding của Transformer thay đổi theo ngữ cảnh, giúp phân biệt từ có nhiều cách hiểu và khai thác quan hệ xa hơn cửa sổ token gần kề. Baseline có thể học cách viết hoa, hình dạng chữ và các từ lân cận nhưng gặp giới hạn khi tên ít gặp hoặc khi ngữ cảnh quyết định loại thực thể. Mức cải thiện thực tế cần được đo qua precision, recall và F1, không được giả định trước.

## Prediction 2. Độ dài entity

**Giả thuyết:** Entity một token thường dễ nhận diện toàn vẹn hơn entity nhiều token, đặc biệt nhóm từ bốn token trở lên.

**Lý do:** Entity dài yêu cầu mô hình dự đoán chính xác cả điểm bắt đầu, điểm kết thúc và toàn bộ nhãn trong span. Một token sai có thể biến một entity đúng thành một false positive và một false negative. United Nations Security Council vì vậy có nhiều cơ hội xảy ra boundary error hơn Microsoft. Trường hợp ngoại lệ có thể xuất hiện khi entity dài có mẫu cấu trúc rất đặc trưng.

## Prediction 3. Entity hiếm

**Giả thuyết:** Entity hiếm trong train có tỷ lệ false negative và sai loại cao hơn entity thường gặp.

**Lý do:** Số lần xuất hiện thấp hạn chế việc học quan hệ giữa chuỗi từ và nhãn, đặc biệt với tên riêng không có quy luật hình thái nổi bật. Transformer có thể khái quát nhờ tiền huấn luyện và ngữ cảnh, nhưng không loại bỏ hoàn toàn lỗi với tên ít xuất hiện.

## Kiểm chứng

- Experiment 2 sử dụng cùng tập test để đo entity-level precision, recall và F1 của baseline và Transformer.
- Experiment 4 kiểm tra khác biệt giữa PER, ORG, LOC và MISC.
- Experiment 5 kiểm tra F1 theo độ dài 1, 2, 3 và 4+ token.
- Experiment 6 kiểm tra entity hiếm và boundary errors bằng ví dụ thực tế.

Dự đoán trên được viết trước thực nghiệm, không phải kết luận về kết quả đo.
