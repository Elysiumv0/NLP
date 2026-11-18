# Phân tích lỗi NER

## Phạm vi

Dữ liệu đánh giá là tập test CoNLL-2003. Lỗi được xác định bằng span entity và nhãn BIO ở cấp độ word.

## False positives

| Text | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|
| SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . | JAPAN [LOC]; CHINA [PER] | JAPAN [PER]; CHINA [LOC]; SURPRISE [MISC] | false positive | Sai loại thực thể |
| SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . | JAPAN [LOC]; CHINA [PER] | JAPAN [PER]; CHINA [LOC]; SURPRISE [MISC] | false positive | Sai loại thực thể |
| SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . | JAPAN [LOC]; CHINA [PER] | JAPAN [PER]; CHINA [LOC]; SURPRISE [MISC] | false positive | Thực thể không có trong nhãn chuẩn |
| Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday . | Japan [LOC]; Asian Cup [MISC]; Syria [LOC] | Japan [LOC]; Asian Cup [MISC]; Syria [LOC]; Group C [MISC] | false positive | Thực thể không có trong nhãn chuẩn |
| Oleg Shatskiku made sure of the win in injury time , hitting an unstoppable left foot shot from just outside the area . | Oleg Shatskiku [PER] | Oleg [MISC]; Shatskiku [PER] | false positive | Sai loại thực thể |

Số ví dụ báo cáo: 5.

## False negatives

| Text | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|
| SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . | JAPAN [LOC]; CHINA [PER] | JAPAN [PER]; CHINA [LOC]; SURPRISE [MISC] | false negative | Bỏ sót thực thể hoặc nhầm loại |
| SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . | JAPAN [LOC]; CHINA [PER] | JAPAN [PER]; CHINA [LOC]; SURPRISE [MISC] | false negative | Thực thể hiếm trong train |
| Oleg Shatskiku made sure of the win in injury time , hitting an unstoppable left foot shot from just outside the area . | Oleg Shatskiku [PER] | Oleg [MISC]; Shatskiku [PER] | false negative | Thực thể hiếm trong train |
| Defender Hassan Abbas rose to intercept a long ball into the area in the 84th minute but only managed to divert it into the top corner of Bitar 's goal . | Hassan Abbas [PER]; Bitar [PER] | Hassan Abbas [PER]; Bitar [ORG] | false negative | Thực thể hiếm trong train |
| RUGBY UNION - CUTTITTA BACK FOR ITALY AFTER A YEAR . | RUGBY UNION [ORG]; CUTTITTA [PER]; ITALY [LOC] | RUGBY UNION [ORG]; CUTTITTA [LOC]; ITALY [ORG] | false negative | Thực thể hiếm trong train |

Số ví dụ báo cáo: 5.

## Boundary errors

| Text | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|
| Oleg Shatskiku made sure of the win in injury time , hitting an unstoppable left foot shot from just outside the area . | Oleg Shatskiku [PER] | Oleg [MISC]; Shatskiku [PER] | boundary error | Đúng loại nhưng sai vị trí bắt đầu hoặc kết thúc |
| Cuttitta announced his retirement after the 1995 World Cup , where he took issue with being dropped from the Italy side that faced England in the pool stages . | Cuttitta [PER]; 1995 World Cup [MISC]; Italy [LOC]; England [LOC] | Cuttitta [PER]; World Cup [MISC]; Italy [LOC]; England [LOC] | boundary error | Đúng loại nhưng sai vị trí bắt đầu hoặc kết thúc |
| FREESTYLE SKIING-WORLD CUP MOGUL RESULTS . | SKIING-WORLD CUP [MISC] | CUP [MISC] | boundary error | Đúng loại nhưng sai vị trí bắt đầu hoặc kết thúc |
| SOCCER - ASIAN CUP GROUP C RESULTS . | ASIAN CUP [MISC] | ASIAN CUP GROUP [MISC] | boundary error | Đúng loại nhưng sai vị trí bắt đầu hoặc kết thúc |
| SOCCER - ENGLISH F.A. CUP SECOND ROUND RESULT . | ENGLISH F.A. CUP [MISC] | ENGLISH [MISC]; F.A. CUP [MISC] | boundary error | Đúng loại nhưng sai vị trí bắt đầu hoặc kết thúc |

Số ví dụ báo cáo: 5.

## Nhận xét

False positive xuất hiện khi model thêm một span không có trong gold. False negative là span chuẩn bị bỏ sót. Boundary error yêu cầu dự đoán và gold có phần giao nhau và cùng loại nhưng khác ranh giới.

Độ hiếm được xác định từ số lần toàn bộ tên entity cùng loại xuất hiện trong train. Entity có tối đa hai lần xuất hiện được đánh dấu là hiếm.
