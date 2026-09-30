1.

Nếu tăng n, ta sẽ nhận được thêm thông tin về ngữ cảnh. Nhờ đó phân phối của từ tiếp theo nhọn và khớp dữ liệu hơn.

2.

Số tổ hợp tăng theo V^n trong khi số token cố định.

3.

Maximum likelihood gán giá trị 0 cho n gram chưa thấy và làm xác suất bằng 0. Ta không muốn điều đó xảy ra vì đó chỉ là từ chưa xuất hiện chứ không phải không xảy ra.

4.

Perplexity đo mức độ đánh giá cao đến mức nào của model đối với dữ liệu, tức số lựa chọn tương đương mà model phải phân vân ở mỗi bước.

5.

Không, perplexity đo xác suất trên dữ liệu đánh giá, không đo mức độ mạch lạc, sự thật.

6.

+, Chỉ nhìn được n-1 từ nên không nắm bắt được phụ thuộc xa.
+, Không có khái niệm ngữ nghĩa hay tương đồng.
+, Sparsity
+, Bộ nhớ phình to.

7.

Không, trigram sử dụng 2 từ liền trước để đoán ngữ cảnh nên 97 từ đầu của context bị bỏ qua hoàn toàn.
