## 1 

Prediction: Cặp gần nhất là `doctor–physician`; tiếp theo là `doctor–hospital` và
`physician–hospital`. `banana` và `car` nằm xa cả nhóm y tế và xa nhau.
Thứ tự dự đoán: doctor–physician > doctor–hospital ≈ physician–hospital >> banana–car ≈ car–doctor.

Reason: Hai từ `doctor` và `physician` đứng trong cùng các khung câu (`the ___ treated the patient`,
`the ___ prescribed ...`) nên theo distributional hypothesis chúng có context gần như giống nhau.
`hospital` cùng chủ đề nhưng đóng vai trò khác, nên chỉ giống ở mức
"cùng chủ đề". `banana` và `car` thuộc hai chủ đề không giao nhau với y tế.

Confidence: 85%

---

## 2 

Prediction: Có. Câu trong corpus rất ngắn nên window=5 gần như phủ cả câu: mọi từ
cùng câu đều thành context. Dự đoán `doctor–hospital` tăng, còn `doctor–physician` gần như không đổi hoặc giảm nhẹ vì vốn đã dựa vào context gần ở window nhỏ, window lớn làm nó "loãng" sang quan hệ topical.

Reason: Window nhỏ nghiêng về quan hệ thay thế được cho nhau, window lớn nghiêng về quan hệ cùng chủ đề.

Confidence: 60% 
---

## 3

Prediction: Không. Vocabulary chỉ ~200 từ, corpus nhỏ và có cấu trúc đơn giản, nên 50 chiều
đã đủ dung lượng; 100 và 300 chiều cho chất lượng tương đương, trong khi thời gian huấn luyện và
kích thước model tăng gần tuyến tính theo số chiều. Có thể 300 chiều còn kém hơn một chút vì quá nhiều tham số
so với dữ liệu.

Reason: Chất lượng phụ thuộc đồng thời vào capacity, lượng dữ liệu và độ phức tạp của quan hệ
cần biểu diễn; tăng một yếu tố khi các yếu tố còn lại không đổi không đảm bảo cải thiện.

Confidence: 80%

---

## 4

Prediction: Không. Với 100 câu, `physician` chỉ xuất hiện khoảng 8–10 lần; vector được cập nhật rất ít lần nên gần như còn ngẫu nhiên. Kết quả sẽ dao động mạnh theo random seed và theo 100 câu nào được lấy mẫu.

Reason: Embedding là ước lượng thống kê; ít dữ liệu → phương sai lớn. Distributional hypothesis chỉ có tác dụng
khi có đủ quan sát về context.

Confidence: 85%

---
| 1a (doctor–physician gần nhất, conf 85%) | đúng | |
| 1b (banana/car xa, conf 65%) | **sai** | banana–car đứng thứ 2 |
| 2 (hướng thay đổi) | đúng hướng, hiệu ứng ≈ nhiễu | conf 60%/40% hợp lý |
| 3 (dim không chắc tăng, conf 80%) | đúng | |
| 4 (không ổn định, conf 85%) | đúng, kèm phát hiện bất ngờ về over-training | |
