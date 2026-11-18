# Bài tập tính toán NER

## Bài 1. Gán nhãn BIO

Câu: Elon Musk founded SpaceX in California.

| Token | Nhãn BIO |
|---|---|
| Elon | B-PER |
| Musk | I-PER |
| founded | O |
| SpaceX | B-ORG |
| in | O |
| California | B-LOC |
| . | O |

Elon Musk là một thực thể PER gồm hai token. SpaceX là một thực thể ORG. California là một thực thể LOC. Các token ngoài thực thể mang nhãn O.

## Bài 2. Ranh giới thực thể

Cụm: University of California Berkeley

| Token | Nhãn BIO |
|---|---|
| University | B-ORG |
| of | I-ORG |
| California | I-ORG |
| Berkeley | I-ORG |

Việc gán tất cả token là ORG chỉ thể hiện loại mà không mô tả điểm bắt đầu và các token nằm trong cùng một thực thể. BIO sử dụng B-ORG để xác định điểm bắt đầu và I-ORG cho phần tiếp theo. Nhờ đó, mô hình phân biệt được một tổ chức kéo dài bốn token với nhiều tổ chức liền kề hoặc các token cùng loại xuất hiện ở vị trí khác nhau.

## Bài 3. Kích thước tensor

Cho B = 8, T = 128, d = 768, C = 9.

| Thành phần | Shape | Giải thích |
|---|---|---|
| Hidden states H | 8 × 128 × 768 | Mỗi câu có 128 token, mỗi token có vector 768 chiều |
| Classification weight W | 768 × 9 | Ánh xạ hidden representation sang 9 nhãn |
| Bias b | 9 | Một bias cho mỗi nhãn |
| Logits Z | 8 × 128 × 9 | Mỗi token có 9 logits |
| Labels Y | 8 × 128 | Mỗi token tương ứng một nhãn hoặc ignore index |

Công thức: Z = HW + b.

## Bài 4. Ignore index

Chiến lược được chọn: gán nhãn cho subword đầu tiên. Các subword tiếp theo và special tokens được gán -100.

Ví dụ minh họa:

| Word | Subword | Label |
|---|---|---|
| playing | play | B-MISC |
| playing | ##ing | -100 |

Với labels thực tế của CoNLL-2003, hệ thống chỉ sử dụng O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC. B-MISC trong bảng là minh họa cho cơ chế alignment, không khẳng định playing là một entity trong dữ liệu CoNLL-2003.

Loss chỉ được tính tại các vị trí có nhãn khác -100. Subword bị ignore vẫn tham gia self-attention và ảnh hưởng đến hidden representation của những token khác.

## Bài 5. Entity-level evaluation

Gold: New York có nhãn B-LOC, I-LOC.

Prediction: New York có nhãn B-LOC, O.

Ở token đầu, New được dự đoán đúng. Tuy nhiên span chuẩn là New York, còn span dự đoán chỉ là New. Theo exact-span matching, mô hình có 0 TP, 1 FP và 1 FN cho tình huống này. Vì vậy entity-level precision, recall và F1 đều bằng 0.

Công thức: Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 = 2 × Precision × Recall / (Precision + Recall).

Khi không có mẫu dự đoán hay mẫu chuẩn, chương trình gán giá trị 0 cho tỷ lệ có mẫu số bằng 0. Kết quả được tính trên toàn tập, không lấy trung bình đơn giản của F1 từng câu.
