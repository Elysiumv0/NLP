# Prediction trước experiment (LAB 03)

> Các dự đoán dưới đây được viết **trước khi** chạy Word2Vec (chỉ mới chạy phần co-occurrence
> thuần count ở `cooccurrence.py`). Phần "Đối chiếu kết quả" ở cuối được điền **sau** khi chạy.

Thiết lập dự kiến: corpus tổng hợp ~30.000 câu ngắn (4–8 từ) gồm các chủ đề y tế, động vật,
phương tiện, công nghệ, thể thao, trái cây, hoàng gia, địa lý, và từ `bank` (2 nghĩa).
Word2Vec: `vector_size=100, window=5, min_count=2, epochs=10`.

---

## Prediction 1 — Những từ nào gần nhau nhất? (doctor, physician, hospital, banana, car)

**Prediction:** Cặp gần nhất là `doctor–physician`; tiếp theo là `doctor–hospital` và
`physician–hospital`. `banana` và `car` nằm xa cả nhóm y tế và xa nhau (cosine thấp, gần 0 hoặc âm).
Thứ tự dự đoán: doctor–physician > doctor–hospital ≈ physician–hospital >> banana–car ≈ car–doctor.

**Reason:** Hai từ `doctor` và `physician` đứng trong cùng các khung câu (`the ___ treated the patient`,
`the ___ prescribed ...`) nên theo distributional hypothesis chúng có context gần như giống nhau.
`hospital` cùng chủ đề nhưng đóng vai trò khác (nơi chốn, không phải người), nên chỉ giống ở mức
"cùng chủ đề". `banana` và `car` thuộc hai chủ đề không giao nhau với y tế.

**Confidence:** 85% cho việc doctor–physician là cặp gần nhất; 65% cho thứ tự hospital > banana/car.

---

## Prediction 2 — Window tăng từ 2 → 5, similarity có đổi không?

**Prediction:** Có đổi. Câu trong corpus rất ngắn (4–8 từ) nên window=5 gần như phủ cả câu: mọi từ
cùng câu đều thành context. Tôi dự đoán `doctor–hospital` **tăng** (quan hệ cùng chủ đề/cùng câu được
củng cố), còn `doctor–physician` **gần như không đổi hoặc giảm nhẹ** (vì vốn đã dựa vào context gần ở window nhỏ,
window lớn làm nó "loãng" sang quan hệ topical).

**Reason:** Window nhỏ nghiêng về quan hệ thay thế được cho nhau (syntactic/paradigmatic); window lớn
nghiêng về quan hệ cùng chủ đề (topical/associative).

**Confidence:** 60% (hướng thay đổi của doctor–hospital), 40% (hướng của doctor–physician).

---

## Prediction 3 — Dimension 50 → 100 → 300, chất lượng có chắc chắn tăng không?

**Prediction:** Không chắc chắn. Vocabulary chỉ ~200 từ, corpus nhỏ và có cấu trúc đơn giản, nên 50 chiều
đã đủ dung lượng; 100 và 300 chiều cho chất lượng tương đương (±nhiễu), trong khi thời gian huấn luyện và
kích thước model tăng gần tuyến tính theo số chiều. Có thể 300 chiều còn kém hơn một chút vì quá nhiều tham số
so với dữ liệu.

**Reason:** Chất lượng phụ thuộc đồng thời vào capacity (số chiều), lượng dữ liệu và độ phức tạp của quan hệ
cần biểu diễn; tăng một yếu tố khi các yếu tố còn lại không đổi không đảm bảo cải thiện.

**Confidence:** 80% rằng 300 chiều không tốt hơn rõ rệt so với 100 chiều.

---

## Prediction 4 — doctor, physician có chắc chắn gần nhau nếu corpus chỉ có 100 câu?

**Prediction:** Không chắc chắn. Với 100 câu, `physician` chỉ xuất hiện khoảng 8–10 lần (có thể bị loại nếu
`min_count` cao); vector được cập nhật rất ít lần nên gần như còn ngẫu nhiên. Kết quả sẽ dao động mạnh theo
random seed và theo 100 câu nào được lấy mẫu.

**Reason:** Embedding là ước lượng thống kê; ít dữ liệu → phương sai lớn. Distributional hypothesis chỉ có tác dụng
khi có đủ quan sát về context.

**Confidence:** 85% rằng độ gần doctor–physician sẽ không ổn định (độ lệch chuẩn lớn giữa các seed).

---

## Prediction bổ sung (CBOW vs Skip-gram) — xem `calculations.md`, Bài 5

Đã liệt kê training pairs bằng tay, không chạy code (CBOW: 4 mẫu, Skip-gram: 6 cặp với `window=1`).

---

## Đối chiếu kết quả (điền sau khi chạy `word_embedding.ipynb`; số liệu gốc trong `results.csv`)

Corpus: 30.000 câu, vocabulary 212 từ; Word2Vec skip-gram, `vector_size=100, window=5, min_count=2, epochs=10`.

### Prediction 1 — **đúng một phần, sai ở phần cuối**
Ma trận cosine thực tế (Word2Vec):

| | doctor | physician | hospital | banana | car |
|---|:--:|:--:|:--:|:--:|:--:|
| doctor | 1 | **0,737** | 0,389 | 0,237 | 0,253 |
| physician | 0,737 | 1 | 0,321 | 0,110 | 0,192 |
| hospital | 0,389 | 0,321 | 1 | 0,220 | 0,177 |
| banana | 0,237 | 0,110 | 0,220 | 1 | **0,461** |
| car | 0,253 | 0,192 | 0,177 | 0,461 | 1 |

- ✅ `doctor–physician` là cặp gần nhất trong 5 từ (0,737) — đúng như dự đoán (confidence 85%).
- ❌ Dự đoán `banana–car` xa và thấp nhất **sai**: `banana–car = 0,461` đứng **thứ 2**, cao hơn cả `doctor–hospital = 0,389`.
- Bài học: cosine tuyệt đối **không được hiệu chỉnh quanh 0**. Cosine trung bình giữa *mọi* cặp từ trong vocabulary là **0,287**, nên 0,2–0,3 gần như là "mức nền",
  và 0,46 của `banana–car` chỉ hơn mức nền ~0,17. Chỉ nên so sánh **thứ hạng/tương đối**, không đọc 0,4 là "gần". Nguyên nhân cụ thể của 0,461 là giả thuyết chưa kiểm chứng
  (cả hai là danh từ vật thể có khung câu giống nhau như "the X ...", "a new X"; corpus nhỏ).
- Ghi chú: `nurse` (không nằm trong 5 từ) mới là láng giềng gần nhất của `doctor` (0,989) — xem `error_analysis.md`.

### Prediction 2 — **hướng đúng, nhưng hiệu ứng quá nhỏ để khẳng định** (3 seed, mean ± std)

| cặp | window 2 | window 5 | window 10 |
|---|:--:|:--:|:--:|
| doctor–physician | 0,749 ± 0,007 | 0,739 ± 0,003 | 0,732 ± 0,007 |
| doctor–hospital | 0,375 ± 0,009 | 0,384 ± 0,004 | 0,361 ± 0,004 |
| cat–dog | 0,845 ± 0,010 | 0,797 ± 0,009 | 0,780 ± 0,006 |

- Similarity **có** thay đổi, nhưng 2→5 chỉ ~0,01 (cùng bậc với độ lệch chuẩn giữa các seed). `doctor–hospital` tăng nhẹ (+0,009) và `doctor–physician` giảm nhẹ (−0,010) đúng chiều dự đoán,
  nhưng không đủ lớn để kết luận. Ở window 10, `doctor–hospital` **giảm** (0,361) → trái với giả thuyết "window lớn = càng topical".
- Ở **count thuần** hiệu ứng lại rất lớn: `cos(doctor, hospital)` = 0 (window 1, 2) → 0,495 (window 3) → 0,651 (window 5). Embedding dense "bù" phần lớn khoảng trống này
  nên ít nhạy với window hơn.

### Prediction 3 — **đúng**

| dim | train (s) | model (KB) | cos(doc,phys) | Spearman gold | analogy top-1 | downstream acc |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| 50 | 2,1 | 96 | 0,743 | 0,801 | 0,750 | 1,000 |
| 100 | 2,1 | 179 | 0,737 | 0,796 | 0,833 | 1,000 |
| 300 | 2,6 | 510 | 0,736 | 0,803 | 0,833 | 1,000 |

Chất lượng **không** tăng chắc chắn: Spearman và cos gần như không đổi; chênh analogy top-1 (0,75 → 0,833) chỉ là **1 trên 12** analogy (nhiễu). Kích thước tăng ~5,3 lần (96 → 510 KB).
Lưu ý: downstream task bão hoà ở 1,000 (task quá dễ với corpus tổng hợp) nên không phân biệt được các cấu hình.

### Prediction 4 — **đúng về sự không ổn định, nhưng có một phát hiện bất ngờ**
30 lần lấy mẫu 100 câu:

| cấu hình | cả 2 từ trong vocab | cos(doctor,physician) | cos(doctor, mọi từ khác) | rank chuẩn hoá của physician (0 = gần nhất) |
|---|:--:|:--:|:--:|:--:|
| 100 câu, 10 epoch | 28/30 | **0,249 ± 0,108** (0,057 – 0,501) | 0,207 | 0,392 |
| 100 câu, 100 epoch | 28/30 | 0,981 ± 0,014 | 0,955 | 0,169 |
| 30.000 câu, 10 epoch | — | 0,737 | 0,228 | 0,014 |

- ✅ Với 100 câu, kết quả dao động mạnh, 2/30 lần thậm chí thiếu `doctor` hoặc `physician` trong vocabulary (do `min_count=2`).
  Ở 10 epoch `cos(doctor, physician)` (0,249) gần như **bằng mức nền** (0,207) và thứ hạng 0,39 (gần ngẫu nhiên 0,5) → **không** có bằng chứng hai từ gần nhau.
- ⚠️ Bất ngờ: train 100 epoch cho cosine **0,981** — nhìn tưởng "rất gần", nhưng cosine trung bình tới *mọi* từ khác cũng là 0,955: các vector **sụp về cùng một hướng** (over-train trên ít dữ liệu).
  Thứ hạng của `physician` (0,169) vẫn kém xa corpus lớn (0,014). Bài học: phải so với mức nền/thứ hạng, không đọc cosine tuyệt đối.

### Tổng kết độ hiệu chỉnh (calibration) của mình
| Prediction | Kết quả | Ghi chú |
|---|---|---|
| 1a (doctor–physician gần nhất, conf 85%) | đúng | |
| 1b (banana/car xa, conf 65%) | **sai** | banana–car đứng thứ 2 |
| 2 (hướng thay đổi) | đúng hướng, hiệu ứng ≈ nhiễu | conf 60%/40% hợp lý |
| 3 (dim không chắc tăng, conf 80%) | đúng | |
| 4 (không ổn định, conf 85%) | đúng, kèm phát hiện bất ngờ về over-training | |
