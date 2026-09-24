## 1
Prediction rằng các document có similarity score cao sẽ thường gần với nội dung query là chưa chính xác. Kết quả thực tế cho thấy similarity cao không đảm bảo document liên quan về mặt chủ đề. Ví dụ, query `transformer language model` có D27936 đứng đầu với similarity 0.500487, nhưng document chủ yếu nói về transformer trong hệ thống điện, không phải language model.
## 2
Điều bất ngờ nhất là một document chỉ cần chứa một hoặc hai từ trong query cũng có thể đạt similarity khá cao. Với `natural language processing`, nhiều kết quả đứng đầu liên quan đến từ `language` nhưng không thực sự nói về Natural Language Processing. Điều này cho thấy TF-IDF khá nhạy với lexical overlap.
## 3
Experiment search và error analysis cung cấp evidence rõ nhất về giới hạn của TF-IDF. Các kết quả retrieval cho thấy những từ chung như `language`, `classification`, `image` hoặc `transformer` có thể kéo các document không liên quan lên thứ hạng cao. Experiment preprocessing cũng cho thấy cách xử lý text làm thay đổi đáng kể vocabulary và sparse representation.
## 4
Failure case quan trọng nhất là query `transformer language model`. D27936 đứng ở rank 1 với similarity 0.500487 nhưng không liên quan đến language model. Đây là ví dụ rõ về vấn đề một từ có thể xuất hiện trong các ngữ cảnh khác nhau nhưng TF-IDF không phân biệt được nghĩa của từ đó.
## 5
Em sẽ giữ TF-IDF làm baseline vì nó đơn giản và dễ giải thích, nhưng bổ sung semantic representation hoặc dense retrieval. Cách này có thể giúp hệ thống hiểu quan hệ về nghĩa giữa các từ và document thay vì chỉ dựa vào sự trùng khớp token.
## 6
Em sử dụng AI để hỗ trợ tìm nguyên nhân lỗi Python, sửa một số lỗi trong notebook, giải thích kết quả và hỗ trợ diễn đạt tài liệu. Ví dụ, AI hỗ trợ xác định lỗi `KeyError: False` trong phần error analysis và đề xuất cách truy cập cột Pandas chính xác. Các kết quả thực nghiệm, quan sát từ corpus và nội dung phân tích cuối cùng được kiểm tra dựa trên output của notebook.
