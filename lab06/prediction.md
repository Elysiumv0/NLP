# Predictions before experiments

## Prediction 1

**Prediction:** Key có dot product lớn hơn sẽ nhận attention weight lớn hơn nếu giữ các score còn lại cố định.

**Reason:** Softmax là hàm tăng theo từng score; score lớn hơn tạo exponential lớn hơn và weight lớn hơn.

**Confidence:** 0.99

## Prediction 2

**Prediction:** Với `s = [1, 1, 1, 1]`, weights là `[0.25, 0.25, 0.25, 0.25]`.

**Reason:** Các exponential bằng nhau nên tổng xác suất được chia đều cho bốn phần tử.

**Confidence:** 1.00

## Prediction 3

**Prediction:** Với `s = [1, 1, 10]`, attention gần như tập trung hoàn toàn vào phần tử thứ ba.

**Reason:** Chênh lệch score lớn làm exponential của score thứ ba lớn vượt trội. Softmax xấp xỉ `[0.000123, 0.000123, 0.999753]`.

**Confidence:** 0.99

## Prediction 4

**Prediction:** Chi phí tạo full attention matrix tăng theo bình phương chiều dài sequence. Tăng từ 10 lên 1,000 tokens làm số phần tử tăng từ 100 lên 1,000,000.

**Reason:** Ma trận có kích thước `T × T`. Tăng sequence length 100 lần làm số phần tử tăng 10,000 lần.

**Confidence:** 1.00

## Prediction 5

**Prediction:** Attention không luôn chọn đúng token mà con người đánh giá là quan trọng nhất.

**Reason:** Attention weights phụ thuộc vào representation, query, key và quá trình huấn luyện. Trọng số biểu diễn phân phối attention, không tự động là giải thích đầy đủ cho quyết định hoặc reasoning của model.

**Confidence:** 0.98
