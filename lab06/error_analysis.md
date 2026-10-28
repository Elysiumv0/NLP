# Error analysis

`results.csv` được tạo bởi `self_attention.ipynb`. Classifier dùng embedding đặc trưng cố định, scaled self-attention và Logistic Regression. Bộ huấn luyện được tạo từ template; tập thử thách bên dưới là các câu được viết riêng và có hiện tượng ngôn ngữ đa dạng hơn. Kết quả trên 20 câu đạt accuracy 0.60. Đây là thí nghiệm minh họa nhỏ, không phải benchmark đại diện.

## Ba dự đoán đúng

| Câu | Nhãn thực | Dự đoán | Quan sát |
|---|---|---|---|
| `I like this movie.` | positive | positive | Tín hiệu sentiment trực tiếp, xuất hiện rõ trong embedding. |
| `I do not like this movie.` | negative | negative | Tiền xử lý minh họa đảo polarity cho token sentiment trong phạm vi negation. |
| `The plot was fine and the acting was poor.` | negative | negative | Từ `poor` tạo tín hiệu tiêu cực mà classifier đã học từ template. |

## Ba dự đoán sai

Các nhãn và dự đoán cụ thể được lấy trực tiếp từ `results.csv`.

| Câu | Dự đoán | Hiện tượng | Giả thuyết về lỗi |
|---|---|---|---|
| `Great, another two hours wasted.` | positive, sai | Sarcasm | `Great` là từ tích cực theo lexicon, trong khi ý nghĩa cả câu mang sắc thái mỉa mai. Embedding cố định không mô hình hóa đầy đủ sarcasm. |
| `This is sick!` | negative, sai | Domain shift | `sick` có thể là tiếng lóng mang nghĩa tích cực nhưng không có trong lexicon; mô hình không có tín hiệu đủ rõ để suy ra nghĩa theo ngữ cảnh. |
| `Despite a very slow beginning, the final scene was absolutely fantastic.` | negative, sai | Long-range dependency | Câu có sentiment tiêu cực sớm và tín hiệu tích cực muộn. Pooling context có thể không làm nổi bật đúng mệnh đề cuối. |

## Đọc attention cẩn trọng

Attention heatmap được dùng để kiểm tra phân phối trọng số. Nếu trọng số không tập trung vào token mà người phân tích cho là quan trọng, đó chỉ là một giả thuyết để kiểm tra tiếp. Không thể kết luận riêng từ heatmap rằng model sai vì “nhìn sai từ”, cũng không thể xem weights là bằng chứng trực tiếp về reasoning.

Đối chiếu các hàng thực tế trong `results.csv` trước khi dùng bảng phân tích này để báo cáo kết quả.
