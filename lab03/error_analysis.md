## A. Ba similarity ĐÚNG

### A1. `doctor → surgeon` (rank 2, cosine 0,791)
- Observed: `surgeon` là láng giềng gần thứ 2 của `doctor`.
- Expected: gần (cùng là nhân viên y tế).
- Possible explanation: hai từ đứng trong cùng khung câu.
- Evidence from corpus: chỉ 1 câu chứa cả hai từ (chúng không đứng cạnh nhau), nhưng chia sẻ nhiều context: `patient(299), surgery(111), about(111), asked(109), treated(104)`.
  Ví dụ: the doctor / surgeon treated the patient in the hospital, the patient asked the doctor / surgeon about the cancer.
  `surgeon` thấp hơn `nurse` vì có thêm template riêng và ít dữ liệu hơn.

### A2. `doctor → physician` (rank 3, cosine 0,737)
- Observed: nằm trong top-3.
- Expected: rất gần (đồng nghĩa; điểm cảm tính của tôi 9,5/10).
- Possible explanation: distributional hypothesis — cùng context dù không bao giờ đứng cạnh nhau.
- Evidence from corpus: 0 câu chứa cả hai; context chung: `patient(503), recommended(161), provided(161), examined(161), by(161)`.
  Count thuần ở window 5 cũng cho 0,769. Điểm trừ của cosine đến từ các context riêng: chỉ `doctor` có `said(183), should(177), rest(177), visited(154), long(150), home(150)`;
  chỉ `physician` có `clinical(356), published(198), study(194), reviewed(164), evidence(162)`.

### A3. `banana → orange / mango / apple` (cosine 0,937–0,941)
- Observed: ba loại quả khác chiếm top-3 của `banana`, cosine > 0,93.
- Expected: gần nhau (cùng là trái cây).
- Possible explanation: thay thế được trong cùng khung câu.
- Evidence from corpus: 0–1 câu chứa cả hai, nhưng chung context `sweet(226), market(119–140), sold(130–133), fresh(130–133), farmer(130–133)`, và `breakfast(129)` với `orange`.

---

## B. Ba similarity sai

### B1. `doctor → nurse` (0,989) cao hơn `doctor → physician` (0,737)
- Observed: `nurse` là láng giềng số 1 với cosine 0,989; `physician` chỉ rank 3.
- Expected: `physician` (đồng nghĩa) gần nhất; `nurse` thấp hơn — trong xếp hạng cảm tính của tôi `doctor–physician` hạng 1,5 còn `doctor–nurse` hạng 10.
- Possible explanation: embedding đo độ giống phân bố context trong corpus này, không phải quan hệ đồng nghĩa trong từ điển. Ở đây `doctor` và `nurse` dùng chung các template đời thường, còn `physician` có các template "trang trọng/nghiên cứu".
- Evidence from corpus: context có ở `doctor` mà không có ở `nurse` gần như bằng 0  — hai phân bố gần như trùng khít.
  Trái lại context có ở `doctor` nhưng không có ở `physician`: `said(183), should(177), rest(177), visited(154)...`; và ngược lại `clinical(356), published(198), study(194)...`.
- Cause: domain/corpus bias + thiết kế template. Về nguyên tắc chung: distributional similarity nổi tiếng là khó phân biệt đồng nghĩa với đồng đẳng — nên hiện tượng này cũng có thể gặp trên corpus thật, nhưng độ lớn 0,989 ở đây là do dữ liệu nhân tạo.

### B2. `doctor → rest` (0,572, rank 4) và `doctor → should` (0,555, rank 5) xếp trên `patient`, `cancer`, `infection`
- Observed: hai từ không mang nghĩa y tế lọt top-5; `patient` chỉ rank 11 (0,439), `hospital` rank 22 (0,389).
- Expected: các từ chủ đề y tế (`patient`, `hospital`, `disease`).
- Possible explanation: hiện tượng collocation của một khung câu lặp lại.
- Evidence from corpus: 177 câu chứa cả `doctor` và `rest`; context chung chỉ có `said(183), should(177)` — tức cặp này giống nhau gần như hoàn toàn nhờ một template duy nhất.
- Cause: corpus nhỏ/lặp template, tần suất của một khung câu, context window. Trên corpus lớn, đa dạng, loại láng giềng này sẽ bị pha loãng.

### B3. `banana – car` (0,461) cao hơn `doctor – hospital` (0,389)
- Observed: hai từ hoàn toàn khác chủ đề lại gần nhau hơn một cặp cùng chủ đề y tế.
- Expected: `doctor–hospital` cao hơn hẳn; `banana–car` thấp (thậm chí ≈ 0).
- Possible explanation (có bằng chứng):
  (1) Cosine không được hiệu chỉnh: cosine trung bình giữa mọi cặp từ là 0,287, nên 0,461 chỉ cao hơn mức nền ~0,17 và 0,389 cao hơn ~0,10 — chênh lệch nhỏ.
  (2) Similarity ≠ relatedness: `hospital` là địa điểm, `doctor` là người → đứng ở vai trò cú pháp khác nhau nên context không thay thế được cho nhau
  dù thường xuất hiện cùng câu
---

## C. Các ví dụ được nêu trong đề

| Cặp | Cosine | Rank / 211 | #câu chứa cả hai | Nhận xét |
|---|:--:|:--:|:--:|---|
| doctor → nurse | 0,989 | 1 | 0 | xem B1 |
| doctor → hospital | 0,389 | 22 | 460 | xem B3: cùng chủ đề nhưng khác vai trò → cosine vừa phải |
| doctor → disease | 0,410 | 17 | 120 | cùng chủ đề; cosine thấp hơn `cancer`/`infection` (0,45/0,44) (giả thuyết, chưa kiểm chứng) vì `disease` là từ chung chung và `cancer`/`infection` dùng cùng khung với nhau |
