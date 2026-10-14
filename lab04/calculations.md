# Calculations

## Bài 1 — Binary classification

Cho:

\[
\mathbf{x}=[1,2,0],\qquad \mathbf{w}=[0.5,-0.2,0.4],\qquad b=0.1
\]

Tính linear score:

\[
z=\mathbf{w}^{T}\mathbf{x}+b
=0.5\cdot1+(-0.2)\cdot2+0.4\cdot0+0.1
=0.2
\]

Sigmoid:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
=\frac{1}{1+e^{-0.2}}
\approx 0.5498
\]

Vậy \(P(y=1\mid x)\approx 0.5498\), tương đương khoảng **54.98%**.

## Bài 2 — Prediction threshold

Quy tắc dự đoán là positive khi xác suất lớn hơn hoặc bằng threshold.

- Với \(P(y=1\mid x)=0.72\) và \(\tau=0.5\), dự đoán **class 1**.
- Với \(P(y=1\mid x)=0.72\) và \(\tau=0.8\), dự đoán **class 0**.

Khi tăng threshold, mô hình khó dự đoán positive hơn. Thông thường precision tăng và recall giảm. Đây là xu hướng thường gặp, không phải bảo đảm toán học cho mọi tập dữ liệu.

## Bài 3 — Confusion matrix

| | Predicted positive | Predicted negative |
|---|---:|---:|
| Actual positive | TP = 80 | FN = 20 |
| Actual negative | FP = 10 | TN = 90 |

\[
\text{Precision}=\frac{TP}{TP+FP}=\frac{80}{80+10}\approx0.8889
\]

\[
\text{Recall}=\frac{TP}{TP+FN}=\frac{80}{80+20}=0.8
\]

\[
F1=2\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
\approx0.8421
\]

Precision ≈ **88.89%**, recall = **80%**, F1 ≈ **84.21%**.

Nếu positive là spam:

- **False positive:** email hợp lệ bị đánh dấu nhầm là spam.
- **False negative:** email spam lọt vào hộp thư đến.

## Bài 4 — Accuracy trap

Có 950 negative và 50 positive. Mô hình luôn dự đoán negative.

\[
\text{Accuracy}=\frac{950}{1000}=0.95=95\%
\]

Accuracy cao không có nghĩa mô hình phát hiện positive tốt. Với class positive:

- TP = 0, FP = 0, FN = 50, TN = 950.
- Recall = 0%.
- F1 = 0 khi quy ước precision không xác định được là 0.

Cần xem confusion matrix, precision, recall, F1 cho từng class, macro F1 và balanced accuracy. Với bài toán lệch lớp, nên xem thêm precision-recall curve và PR-AUC khi cần chọn threshold.
