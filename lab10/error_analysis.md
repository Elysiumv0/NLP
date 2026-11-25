# Error analysis

Tổng số câu: 1500

Wrong start riêng: 185
Wrong end riêng: 123
Wrong span: 333
Tổng số sai start: 518
Tổng số sai end: 456

Các nhóm được xác định theo vị trí ký tự, đối chiếu với đáp án chuẩn có biên phù hợp nhất. EM và F1 lấy giá trị cao nhất trên các đáp án chuẩn.

## wrong start

Số ví dụ tìm được: 185

| Question | Gold | Prediction | Error type | Explanation | F1 |
|---|---|---|---|---|---:|
| What is the definition of agency as it relates to capabilities?  | the ability to pursue valued goals | ability to pursue valued goals | wrong start | Thiếu phần đầu của gold. Partial answer. | 1.0000 |
| What protection caused Luther's Reformation to flourish? | wing of the secular powers | under the wing of the secular powers | wrong start | Lấy thêm văn bản trước gold. | 0.8889 |
| Who was appointed as second in command to Lor Loudoun in 1756? | Major General James Abercrombie | James Abercrombie | wrong start | Thiếu phần đầu của gold. Partial answer. | 0.6667 |
| The mountain ranges tail off into what kind of geographical formation? | interior valleys | Transverse and Peninsular Ranges with their peaks, into the large and small interior valleys | wrong start | Lấy thêm văn bản trước gold. | 0.2667 |
| How do ctenophores control buoyancy? | It is uncertain | uncertain | wrong start | Thiếu phần đầu của gold. Partial answer. | 0.5000 |

## wrong end

Số ví dụ tìm được: 123

| Question | Gold | Prediction | Error type | Explanation | F1 |
|---|---|---|---|---|---:|
| Many organizations have recently called for broader acceptance of which community? | LGBT community | LGBT community within the UMC | wrong end | Lấy thêm văn bản sau gold. | 0.6667 |
| Whose theory did Tesla disagree with? | Einstein's | Einstein's theory of relativity | wrong end | Lấy thêm văn bản sau gold. | 0.4000 |
| What councils assign tasks to the IPCC? | WMO Executive Council and UNEP Governing Council | WMO Executive Council | wrong end | Thiếu phần cuối của gold. Partial answer. | 0.6000 |
| What is the oxygen partial pressure of space suits? | about 30 kPa (1.4 times normal) | about 30 kPa (1.4 times normal | wrong end | Thiếu phần cuối của gold. Partial answer. | 1.0000 |
| What is William Rankine's nationality? | Scottish | Scottish polymath | wrong end | Lấy thêm văn bản sau gold. | 0.6667 |

## wrong span

Số ví dụ tìm được: 333

| Question | Gold | Prediction | Error type | Explanation | F1 |
|---|---|---|---|---|---:|
| Who won the battle of Lake George? | inconclusively, with both sides withdrawing from the field | Fort Edward and Fort William Henry | wrong span | Span dự đoán nằm ngoài vị trí đáp án được chọn. | 0.0000 |
| What were the origins of the Raouliii family? | an Italo-Norman named Raoul | Italo-Norman | wrong span | Cả hai biên sai nhưng span còn giao với đáp án. Partial answer. | 0.5000 |
| Why did the 5th president of the university decide to get rid of the football program? | to emphasize academics over athletics | Robert Maynard Hutchins, took office; the university underwent many changes during his 24-year tenure. Hutchins eliminated varsity football | wrong span | Span dự đoán nằm ngoài vị trí đáp án được chọn. | 0.0000 |
| At what speed did the turbine operate? | 16,000 rpm | 100–5,000 hp | wrong span | Span dự đoán nằm ngoài vị trí đáp án được chọn. | 0.0000 |
| Classifying the stages of what is important to mapping aspects of the Amazon? | tree growth | classification of tree growth stages | wrong span | Cả hai biên sai nhưng span còn giao với đáp án. | 0.5714 |

## EM = 0 và F1 > 0

Số ví dụ tìm được: 333

| Question | Gold | Prediction | Error type | Explanation | F1 |
|---|---|---|---|---|---:|
| What were the origins of the Raouliii family? | an Italo-Norman named Raoul | Italo-Norman | wrong span | Cả hai biên sai nhưng span còn giao với đáp án. Partial answer. | 0.5000 |
| What protection caused Luther's Reformation to flourish? | wing of the secular powers | under the wing of the secular powers | wrong start | Lấy thêm văn bản trước gold. | 0.8889 |
| Who was appointed as second in command to Lor Loudoun in 1756? | Major General James Abercrombie | James Abercrombie | wrong start | Thiếu phần đầu của gold. Partial answer. | 0.6667 |
| The mountain ranges tail off into what kind of geographical formation? | interior valleys | Transverse and Peninsular Ranges with their peaks, into the large and small interior valleys | wrong start | Lấy thêm văn bản trước gold. | 0.2667 |
| Classifying the stages of what is important to mapping aspects of the Amazon? | tree growth | classification of tree growth stages | wrong span | Cả hai biên sai nhưng span còn giao với đáp án. | 0.5714 |

## Nhận xét

Sai start có thể do model chọn thiếu từ bổ nghĩa hoặc lấy thêm từ đứng trước thực thể. Sai end có thể do model chọn thiếu thành phần phía sau hoặc mở rộng sang cụm từ lân cận. Sai cả hai biên có thể liên quan đến distractor, nhiều span hợp lý hoặc sự mơ hồ của câu hỏi. Truncation, context dài, thiếu thông tin và tokenization cũng là các giả thuyết cần kiểm tra trực tiếp bằng context và offset mapping. Không thể xác định nguyên nhân ngữ nghĩa chỉ từ hai vị trí biên.

F1 vẫn dương nếu prediction và gold có token chung sau normalization, dù EM bằng 0.
