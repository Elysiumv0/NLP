# Reflection — Từ Word2Vec đến Transformer (LAB 03)

## 1. Bảng so sánh các biểu diễn

| Representation | Context-dependent? | Sparse/Dense | Một từ có nhiều vector? |
|---|---|---|---|
| **TF-IDF** | **Không.** Trọng số tính từ thống kê toàn corpus (TF trong văn bản × IDF), không phụ thuộc cách từ được dùng trong câu cụ thể. Vector là của *văn bản* (hoặc của một từ chỉ như một cột), không mã hoá nghĩa của từ. | **Sparse** (|V| chiều, hầu hết bằng 0) | **Không** (mỗi từ là một cột cố định) |
| **Co-occurrence** | **Không.** Context chỉ được dùng *một lần* để xây vector; vector tạo xong là cố định cho mọi câu. | **Sparse** (|V|×|V|); có thể nén thành dense bằng PPMI+SVD | **Không** |
| **Word2Vec** (static) | **Không.** Một từ → đúng một vector, dùng lại ở mọi câu. | **Dense** (vd. 100–300 chiều) | **Không** — mỗi từ đúng một vector |
| **Contextual embedding** (ELMo, BERT, Transformer) | **Có.** Vector của từ được tính *lại cho từng câu* từ toàn bộ ngữ cảnh. | **Dense** | **Có** — cùng một từ, vector khác nhau theo từng câu |

Lưu ý khi đọc bảng: TF-IDF và co-occurrence **có dùng thống kê ngữ cảnh** (nên "context-based"), nhưng không **context-dependent** theo nghĩa *vector thay đổi theo từng lần xuất hiện của từ*. Hai khái niệm này khác nhau.

## 2. Tại sao `bank` cần contextual representation?

Từ `bank` có (ít nhất) hai nghĩa không liên quan — *ngân hàng* và *bờ sông* — như trong *I deposited money in the bank* và *We sat on the river bank*. Word2Vec tĩnh chỉ có **một** vector cho `bank`, nên vector đó buộc phải là một dạng
**trung bình có trọng số theo tần suất** của mọi ngữ cảnh mà từ xuất hiện. Hệ quả: vector không đại diện đúng cho nghĩa nào trong một câu cụ thể.

**Bằng chứng từ thí nghiệm (notebook mục 14, corpus có hai nghĩa của `bank` với tần suất tương đương):**
- Top láng giềng của `bank` trộn cả hai nghĩa: `money, save, account, deposited, interest, loan` (tiền) lẫn `river` (sông), cộng với các từ khung câu (`we, an, opened`).
- Vector tĩnh `bank` có cosine **0,627** với prototype-tiền và **0,575** với prototype-sông — gần như "lưng chừng" hai nghĩa, và **cùng một con số cho mọi câu**.
- Nếu *thêm ngữ cảnh của từng câu* (trung bình vector các từ xung quanh), việc phân biệt trở nên rõ ràng: *i deposited money in the bank* → tiền 0,88 vs sông 0,55; *we sat on the river bank* → tiền 0,63 vs sông 0,91.
  Đây chỉ là một cách "contextual" thô sơ (có lợi thế là ta tự định nghĩa sẵn hai prototype), nhưng cho thấy thông tin để phân biệt nằm ở **ngữ cảnh của câu**, thứ mà vector tĩnh đã vứt bỏ.

Contextual embeddings (ELMo dùng BiLSTM; BERT/Transformer dùng self-attention) giải quyết điều này một cách có hệ thống: biểu diễn của `bank` là **hàm của cả câu**, nên hai câu trên cho hai vector khác nhau.
Với Transformer, self-attention cho phép mỗi từ "nhìn" trực tiếp vào mọi từ khác (`money`, `deposited` hay `river`, `sat`) để quyết định nghĩa, và việc huấn luyện song song trên dữ liệu lớn khiến điều này mở rộng được — đó là lý do contextual embeddings và Transformer là bước tiếp theo tự nhiên của Word2Vec.

## 3. Những điều rút ra từ lab (kèm bằng chứng)

1. **Distributional hypothesis hoạt động, nhưng đo "giống ngữ cảnh trong corpus này", không phải "đồng nghĩa".** `doctor` gần `nurse` (0,989) hơn `physician` (0,737) vì corpus tổng hợp cho chúng chung template; hai từ đồng nghĩa không bao giờ đứng cạnh nhau mà vẫn gần nhau nhờ context chung (0 câu chứa cả hai).
2. **Dense không mặc nhiên "tốt hơn" sparse.** Trên 18 cặp tự chấm: Spearman với gold là count 0,816 · PPMI 0,838 · Word2Vec 0,796 · SVD-dense 0,686 (chênh lệch này **không** có ý nghĩa thống kê). Điểm mạnh thật của dense: khái quát hoá — `cos(doctor, hospital)` = 0 ở count window ≤ 2 nhưng 0,375 ở Word2Vec window 2.
3. **Cosine tuyệt đối không được hiệu chỉnh.** Mức nền 0,287; `banana–car` 0,461 > `doctor–hospital` 0,389; ở corpus 100 câu train 100 epoch cho cos 0,981 nhưng mọi từ đều giống nhau (0,955). Luôn so sánh thứ hạng/mức nền.
4. **Window lớn hơn không luôn tốt hơn.** Count thuần rất nhạy (0 → 0,651), Word2Vec trên corpus này ít nhạy (±0,01–0,07); window lớn chuyển từ quan hệ thay thế sang quan hệ chủ đề, và với câu ngắn thì window 10 phủ trọn câu.
5. **Dimension không tự động nâng chất lượng.** 50 → 300 chiều: Spearman 0,801 → 0,803 (không đổi), thời gian 2,1 → 2,6 s, kích thước 96 → 510 KB (×5,3).
6. **Analogy là pattern hình học, không phải "hiểu".** `king − man + woman` có queen trong top-5 và các vector offset song song (cos 0,858), nhưng `princess` đứng trước `queen` vì corpus không cho cách tách `king/prince`.
7. **Dữ liệu ít → kết quả không ổn định** (100 câu: 0,249 ± 0,108, sát mức nền 0,207).
8. **Tất cả dùng corpus tổng hợp.** Số liệu chỉ minh hoạ cơ chế; cần kiểm chứng trên corpus thật trước khi khái quát.

---

## 4. Individual learning check (3 phút/câu — câu trả lời gợi ý)

**Distributional hypothesis là gì?**
Những từ xuất hiện trong ngữ cảnh giống nhau có xu hướng có nghĩa giống nhau ("You shall know a word by the company it keeps"). Do đó có thể biểu diễn mỗi từ bằng thống kê ngữ cảnh của nó
(co-occurrence → vector) và đo độ giống nghĩa bằng độ giống ngữ cảnh (cosine). Ví dụ: *The doctor / physician treated the patient.*

**Tại sao `doctor` và `physician` có thể gần nhau?**
Vì chúng đứng trong cùng các khung câu (*the ___ treated the patient*, *the ___ prescribed medicine*) nên có context chung (`patient(503)`, `recommended(161)`...), dù **không bao giờ đứng cạnh nhau** (0 câu chứa cả hai trong thí nghiệm).
Cần đủ dữ liệu: với corpus 100 câu, độ gần này không còn đáng tin.

**CBOW khác Skip-gram ở đâu?**
CBOW: *context → target* (gộp embedding các context rồi dự đoán từ giữa; ví dụ `[the, eats] → cat`). Skip-gram: *target → context* (từ giữa dự đoán từng context; `cat → the`, `cat → eats`).
Skip-gram sinh nhiều cặp huấn luyện hơn trên cùng một cửa sổ (6 so với 4 ở ví dụ `the cat eats fish`), thường tốt hơn với từ hiếm; CBOW nhanh hơn.

**Tại sao tăng context window có thể vừa tốt vừa xấu?**
Tốt: có thêm thông tin, bắt được quan hệ chủ đề/xa (ở count: `cos(doctor, hospital)` từ 0 lên 0,65). Xấu: context xa thường kém liên quan nên làm loãng quan hệ thay thế/cú pháp, tăng nhiễu và chi phí; với câu ngắn,
window lớn biến mọi từ trong câu thành context của nhau.

**Tại sao Word2Vec không phân biệt được hai nghĩa của `bank`?**
Vì nó học **một vector cho mỗi dạng từ** (static), cập nhật từ mọi ngữ cảnh của từ đó; vector kết quả là trung bình của các nghĩa. Cùng một vector được dùng cho *money in the bank* và *the river bank*.

**Tại sao TF-IDF không phải word embedding?**
TF-IDF là vector **sparse theo văn bản** với mỗi chiều là một từ cụ thể (độ dài |V|); nó không học biểu diễn dense, không đặt các từ gần nghĩa gần nhau (`doctor` và `physician` là hai cột độc lập → không có quan hệ),
và không được huấn luyện để dự đoán/ghi nhận ngữ cảnh. Word embedding là vector **dense, ít chiều, mỗi từ một vector** sao cho từ có ngữ cảnh giống nhau có vector gần nhau.
