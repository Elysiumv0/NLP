# Error Analysis

## Mẫu lỗi

Ba dòng đầu là false positives, ba dòng tiếp theo là false negatives, hai dòng cuối là các lỗi có confidence cao nhất còn lại.

<!-- table:start -->
| Text | True | Pred | Confidence | Error type | Hypothesis |
|---|---|---|---|---|---|
| While this isn't an all time classic comedy it is a pretty good little movie to watch if you catch it on a rainy Saturday morning with not a lot else going on right then.  Harold Lloyd plays Ezekiel C | negative | positive | 0.9959 | label ambiguity | Cụm khen nhẹ ở đầu review là tín hiệu tích cực bề mặt, còn nhận xét tiêu cực nếu có nằm ngoài phần đầu hoặc nhãn không rõ ràng. |
| This has to be one of, if not THE greatest Mob/Crime films of all time. Every thing about this movie is great, the acting in this film is of true quality; Master P's acting skills make you actually be | negative | positive | 0.9940 | sarcasm | Các cụm greatest và true quality là tín hiệu tích cực bề mặt, nếu review mỉa mai thì model theo từ vựng và bỏ qua giọng điệu. |
| I went to see this 3 nights ago here in Cork, Ireland. It was the world premiere of it, in the tiny cinema in the Triskel Arts Centre as part of the Cork Film Festival.  I found "Strange Fruit" to be  | negative | positive | 0.9935 | long-range dependency | Phần mở đầu chỉ kể bối cảnh xem phim, đánh giá thật nằm sau, model dự đoán từ tín hiệu trung tính của phần đầu. |
| This is the kind of movie that you rent when you are incredibly tired, or impaired in some other way... The acting in this movie is so bad it seems intentional, and to let you know how bad the special | positive | negative | 0.9959 | ambiguous sentiment | Nội dung chê diễn xuất trực tiếp nhưng nhãn là positive, có thể là kiểu so bad it's good hoặc nhãn không khớp nội dung. |
| This is not the kind of movie that really merits critical attention. It's not going to win any Oscars - it's really not a very good movie at all. Heck, it's not even really a movie - just a string of  | positive | negative | 0.9956 | negation, long-range dependency | Nhiều cụm phủ định và chê trực tiếp ở đầu, ý kiến tích cực nếu có nằm ở phần sau nên model theo tín hiệu tiêu cực. |
| This was Laurel and Hardy's last silent film for Roach Studios. However, since the public had a real thirst for "talkies", this same short was re-made by the team just a few years later with only a fe | positive | negative | 0.9953 | context | Đoạn đầu là mô tả lịch sử phim không có từ cảm xúc, bằng chứng nhìn thấy được rất ít nhưng confidence vẫn rất cao. |
| This cheapo remake of the terrific Five Star Final suffers from terrible acting. Humphrey Bogart stars as the manager of a sleazoid radio station that is desperate to boost sagging ratings. The owner  | positive | negative | 0.9946 | context | Cheapo, terrible acting, sleazoid là tín hiệu tiêu cực mạnh, còn terrific nói về phim gốc nên model không tách được đối tượng của từng nhận xét. |
| Right, here we go, you have probably read in previous reviews on this film that it is awful, badly acted, avoid at all costs. Well i suppose in some ways this is true, it is fair to say that you could | positive | negative | 0.9946 | context, long-range dependency | Reviewer nhắc lại ý kiến của người khác rồi phản bác ở phần sau, model đọc awful và avoid at all costs như quan điểm của chính reviewer. |
<!-- table:end -->

Trên 5000 mẫu IMDb test, mô hình pretrained fine-tuned có 277 false positives và 350 false negatives. Cả tám mẫu trên đều có confidence từ 0.9935 trở lên.

## Câu hỏi bắt buộc

### Q1. Model nào hưởng lợi nhiều hơn khi training data ít?

Observation: Khi dữ liệu giảm từ 100% xuống 10%, accuracy của mô hình pretrained giảm 4.56 điểm, còn mô hình from scratch giảm 14.22 điểm.

Evidence: Mô hình pretrained đạt 87.46, 86.24, 85.14, 82.90 ở 100%, 50%, 25%, 10%. Mô hình from scratch đạt 80.32, 78.14, 77.60, 66.10. F1 giảm 3.54 điểm với pretrained và 12.44 điểm với from scratch. Khoảng cách accuracy giữa hai mô hình là 7.14 điểm ở 100% và 16.80 điểm ở 10%. Mô hình pretrained ở 10% dữ liệu đạt 82.90, cao hơn mô hình from scratch ở 100% dữ liệu là 80.32.

Interpretation: Mô hình pretrained hưởng lợi nhiều hơn khi dữ liệu ít. Cách giải thích phù hợp là biểu diễn đã học ở giai đoạn pretraining giảm lượng thông tin phải học từ nhãn, nhưng thí nghiệm này chưa tách được yếu tố đó khỏi chênh lệch kích thước mô hình, và mới dùng một seed.

### Q2. Nếu pretrained model tốt hơn from-scratch, điều đó chứng minh điều gì?

Observation: Mô hình pretrained fine-tuned tốt hơn mô hình from scratch ở mọi tỷ lệ dữ liệu.

Evidence: Ở 100% dữ liệu, accuracy là 87.46 và 80.32, F1 là 0.8731 và 0.7971. Đổi lại, thời gian huấn luyện là 334.4 giây so với 21.0 giây, số tham số là 66.4 triệu so với 9.5 triệu, bộ nhớ đỉnh là 2291 MB so với 312 MB.

Interpretation: Trong cấu hình đã thử, khởi tạo từ pretraining cho kết quả tốt hơn khởi tạo ngẫu nhiên. Điều này chưa chứng minh riêng tác động của pretraining, vì hai mô hình khác nhau về số tầng và số tham số. Muốn tách yếu tố này cần so sánh với một mô hình cùng kiến trúc DistilBERT khởi tạo ngẫu nhiên.

### Q3. Nếu frozen model gần bằng fine-tuned model, điều đó có ý nghĩa gì?

Observation: Frozen model không gần fine-tuned model, kém 8.56 điểm accuracy. Tuy vậy frozen model vẫn cao hơn mức ngẫu nhiên rất nhiều.

Evidence: Accuracy là 78.90 với frozen và 87.46 với fine-tuning, F1 là 0.7879 và 0.8731. Frozen chỉ huấn luyện 1538 tham số, mất 109.8 giây và 412 MB so với 334.4 giây và 2291 MB. Frozen thấp hơn cả mô hình from scratch trên IMDb test là 80.32, với chênh lệch 1.42 điểm, trong khi sai số chuẩn của accuracy trên 5000 mẫu khoảng 0.5 điểm.

Interpretation: Vì khoảng cách lớn, trường hợp frozen gần bằng fine-tuned không xảy ra trong thí nghiệm này. Nếu xảy ra, điều đó có nghĩa là biểu diễn pretrained đã chứa thông tin cảm xúc đủ để một bộ phân loại tuyến tính khai thác, và việc cập nhật encoder đem lại ít lợi ích thêm. Kết quả 78.90 cho thấy thông tin cảm xúc có mặt trong biểu diễn nhưng khai thác tuyến tính từ vị trí [CLS] chưa đủ.

### Q4. Nếu fine-tuning tốt hơn frozen model rất nhiều, điều đó gợi ý điều gì?

Observation: Fine-tuning tốt hơn frozen 8.56 điểm accuracy và 8.52 điểm F1.

Evidence: Accuracy 87.46 so với 78.90, F1 0.8731 so với 0.7879, cùng cấu hình dữ liệu, max length, epoch và seed, chỉ khác encoder có được cập nhật hay không.

Interpretation: Biểu diễn của encoder chưa được sắp xếp để một lớp tuyến tính tách tốt hai lớp cảm xúc, và việc cập nhật encoder giúp thích nghi biểu diễn với tác vụ. DistilBERT không được huấn luyện với mục tiêu next sentence prediction, nên vị trí [CLS] không được tối ưu trực tiếp cho biểu diễn cả câu, và đây là một giả thuyết cần kiểm tra bằng mean pooling. Learning rate và số epoch của chế độ frozen cũng chưa được tinh chỉnh.

### Q5. Nếu performance giảm mạnh trên domain khác, nguyên nhân có thể là gì?

Observation: Mô hình pretrained không giảm trên Amazon polarity, còn mô hình from scratch giảm 5.60 điểm.

Evidence: Pretrained fine-tuned đạt 87.46 trên IMDb và 89.50 trên Amazon. Pretrained frozen đạt 78.90 và 82.14. Mô hình from scratch đạt 80.32 và 74.72. Ở 10% dữ liệu, mô hình from scratch đạt 57.82 trên Amazon với recall 0.914 và precision 0.552, tức dự đoán positive cho phần lớn mẫu.

Interpretation: Kết quả không cho thấy suy giảm của mô hình pretrained, nên không kết luận được nguyên nhân giảm cho mô hình này. Accuracy cao hơn trên Amazon không có nghĩa là mô hình tổng quát hóa tốt hơn, vì hai tập test có độ khó khác nhau và review sản phẩm thường ngắn, từ vựng cảm xúc rõ ràng. Điều so sánh được là mô hình from scratch giảm còn mô hình pretrained thì không. Nguyên nhân có thể của mức giảm ở mô hình from scratch là từ vựng và tín hiệu cảm xúc đặc thù của review phim chưa được học trên Amazon, và vốn từ học từ ít dữ liệu hẹp. Hai domain đều là sentiment tiếng Anh, nên kết quả không cho phép kết luận mô hình hiểu mọi domain.

### Q6. Nếu sequence length tăng nhưng performance gần như không đổi trong khi training time tăng mạnh, có nên tiếp tục tăng sequence length không?

Observation: Trong thí nghiệm này performance vẫn tăng khi tăng sequence length, nên điều kiện của câu hỏi không xảy ra, nhưng mức tăng giảm dần trong khi chi phí tăng.

Evidence: Với T bằng 64, 128, 256, accuracy là 81.48, 87.46, 89.90, F1 là 0.8167, 0.8731, 0.8996. Thời gian huấn luyện là 181.4, 334.4, 701.8 giây, bộ nhớ đỉnh là 1688, 2291, 3544 MB. Từ 64 lên 128 accuracy tăng 5.98 điểm với thời gian gấp 1.84 lần, từ 128 lên 256 accuracy tăng 2.44 điểm với thời gian gấp 2.10 lần.

Interpretation: Với các review IMDb dài, cắt ở 64 token bỏ mất phần lớn nội dung nên tăng độ dài có lợi. Mức tăng thời gian thấp hơn 4 lần của O(T²) vì chi phí của phần feed-forward chỉ tăng tuyến tính theo T. Nếu performance không đổi khi chi phí tăng mạnh thì không nên tăng tiếp. Ở đây có nên dùng 256 hay không phụ thuộc vào ngân sách tính toán, và chưa có bằng chứng về T bằng 512.
