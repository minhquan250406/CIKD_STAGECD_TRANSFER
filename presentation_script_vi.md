# Kịch bản Thuyết trình (Presentation Script) - CIKD++

> **Thời lượng dự kiến:** 6-8 phút.
> **Lưu ý:** Lời thoại dưới đây được thiết kế tự nhiên, dễ nghe, phù hợp để thuyết trình trực tiếp từ file PDF, tập trung nhấn mạnh vào các điểm đã được sửa chữa theo góp ý của cô.

---

## 1. Title / Abstract (Trang bìa & Tóm tắt)
"Dạ em chào cô. Hôm nay nhóm em xin trình bày về bài báo cáo của nhóm với đề tài **'CIKD++: Cải thiện khả năng phát hiện tin giả đa phương thức dựa trên Đồ thị Tri thức với Topic-guided Visual Contradiction'**.
Mục tiêu chính của nhóm là giải quyết bài toán phát hiện tin giả tinh vi (Fine-grained Fake News Detection), nơi mà tin giả không chỉ sai lệch về mặt văn bản hay hình ảnh, mà còn có sự mâu thuẫn giữa nội dung và các kiến thức nền tảng thực tế (Knowledge Graph). Điểm nổi bật của phiên bản này là tụi em đã tiếp thu và hoàn thiện các góp ý của cô để cải thiện chất lượng nghiên cứu."

## 2. Terminology and Definitions (Thuật ngữ & Định nghĩa)
"Đầu tiên, theo góp ý của cô về việc cần làm rõ các khái niệm, nhóm em đã bổ sung phần Thuật ngữ và Định nghĩa ngay phần đầu. Tụi em định nghĩa rõ ràng các từ viết tắt và các lớp phân loại, giúp người đọc dễ dàng theo dõi hơn xuyên suốt bài báo. Ví dụ, CK là Content-Knowledge Inconsistency, tức là mâu thuẫn giữa nội dung bài báo và tri thức nền tảng."

## 3. Dataset and preprocessing (Tập dữ liệu & Tiền xử lý)
"Về phần Dữ liệu và Tiền xử lý, nhóm em đã làm lại quy trình một cách minh bạch và rõ ràng hơn rất nhiều so với trước. 
Tập dữ liệu gốc FineFake ban đầu có **N = 16,909** mẫu.
Tụi em thực hiện tiền xử lý, rút trích đặc trưng đa phương thức (văn bản, hình ảnh) và đặc biệt là các thực thể từ Knowledge Graph (KG)."

## 4. KG filtering and class distribution after KG filtering (Lọc KG & Phân bố dữ liệu sau lọc)
"Đến bước lọc KG, tụi em phát hiện có **4,041** mẫu bị thiếu (missing) hoặc có KG embeddings bằng 0. Do mô hình CIKD++ phụ thuộc mạnh vào tri thức nền, tụi em quyết định loại bỏ hoàn toàn **4,123** mẫu không hoàn thiện này khỏi giao thức thử nghiệm KG-grounded.
Việc này trực tiếp phản hồi lại nhận xét của cô về phân bố dữ liệu sau khi lọc KG. Tập dữ liệu cuối cùng (kg_complete) còn lại **N = 12,786** mẫu. Lúc này, số lượng mẫu lớp Real là **6,298** và lớp CK là **1,211**."

## 5. Class imbalance handling (Xử lý mất cân bằng lớp)
"Một góp ý rất quan trọng từ cô là cần xử lý mất cân bằng dữ liệu ở mức độ tập dữ liệu (dataset-level) trước khi áp dụng các kỹ thuật ở mức mô hình (model-level).
Trong báo cáo hiện tại, tụi em đã làm rõ chiến lược xử lý: sử dụng Class Weights (trọng số lớp) dựa trên nghịch đảo tần suất của từng lớp. Phương pháp này giúp mô hình không bị thiên vị quá mức vào lớp Real (chiếm đa số), mà vẫn chú ý học các lớp thiểu số như CK, mà không làm biến dạng tính toàn vẹn của dữ liệu đa phương thức."

## 6. Model architecture Figure 1–4 (Kiến trúc mô hình)
"Tiếp theo, em xin đi vào phần Kiến trúc mô hình. Trả lời góp ý của cô, nhóm em đã vẽ lại hoàn toàn sơ đồ tổng thể (Figure 1-4) để thể hiện luồng dữ liệu trực quan hơn.
Mô hình CIKD++-RT nhận đầu vào từ Văn bản, Hình ảnh (Global và Patch), cùng với KG Features. Cốt lõi của kiến trúc mới là cơ chế Residual Transformer và đặc biệt là module Topic-guided Visual Contradiction Score (TVCS). TVCS giúp mô hình có khả năng 'nhìn' vào bức ảnh dưới sự dẫn dắt của tri thức KG để tìm ra điểm mâu thuẫn."

## 7. Main results (Kết quả chính)
"Về mặt kết quả đánh giá tổng thể, mô hình của tụi em đạt được các chỉ số sau trên tập test:
- **Accuracy = 58.3140%**
- **Macro-F1 = 46.9755%**
- **Weighted-F1 = 59.5127%**
Dù bài toán phân loại 6 nhãn rất khó, kết quả này cho thấy mô hình học được sự biểu diễn đa phương thức tương đối ổn định."

## 8. Ablation / TVCS evidence (Cắt tỉa & Bằng chứng TVCS)
"Đi sâu hơn vào đánh giá các thành phần (Ablation), tụi em đo lường khả năng bắt nhãn CK (Content-Knowledge Inconsistency). Chỉ số **CK-F1 đạt 37.5546%**.
Đặc biệt, khả năng phân định mâu thuẫn hình ảnh của module TVCS đạt **TVCS AUC = 0.726705**. Các biểu đồ heatmap trong PDF cho thấy rõ TVCS có khả năng khoanh vùng chính xác các vùng ảnh chứa thông tin mâu thuẫn với nội dung văn bản và tri thức."

## 9. Calibration (Hiệu chỉnh xác suất)
"Một vấn đề nhóm phát hiện là mô hình thường bị overconfident (quá tự tin vào dự đoán). Do đó, tụi em áp dụng Temperature Scaling. Kết quả là lỗi hiệu chuẩn (Expected Calibration Error - ECE) đã giảm mạnh từ **14.80% xuống chỉ còn 3.85%**. Điều này giúp xác suất đầu ra của mô hình đáng tin cậy hơn khi ứng dụng thực tế."

## 10. Limitations and conclusion (Hạn chế & Kết luận)
"Cuối cùng, nhóm nhận thức rõ mô hình vẫn còn những hạn chế. Việc loại bỏ các mẫu thiếu KG làm giảm quy mô tập dữ liệu và khiến hệ thống không hoạt động tốt nếu tin tức không chứa thực thể nào rõ ràng.
Tuy nhiên, tụi em kết luận rằng CIKD++ đã cải thiện đáng kể khả năng giải thích và tính định hướng trong việc phát hiện tin giả. Em xin cảm ơn cô đã lắng nghe, và tụi em rất mong nhận được thêm ý kiến đóng góp từ cô ạ."

---

## 🛑 If the teacher asks (Dự phòng trả lời câu hỏi của cô)

**1. Why remove KG-incomplete samples? (Tại sao lại xóa các mẫu thiếu KG?)**
> "Dạ thưa cô, mô hình CIKD++ được thiết kế để đối chiếu nội dung với tri thức nền (Knowledge-grounded). Nếu đầu vào là các vector zeros (0), các trọng số attention sẽ bị nhiễu và vô nghĩa. Việc tạo ra tập `kg_complete` giúp đánh giá chính xác sức mạnh của cơ chế TVCS khi thực sự có tri thức."

**2. Why not use SMOTE/oversampling? (Tại sao không dùng SMOTE hay sinh mẫu?)**
> "Dạ, vì dữ liệu của tụi em là đa phương thức (ảnh, text, đồ thị tri thức). Các phương pháp oversampling truyền thống như SMOTE trên không gian vector nhúng đa phương thức thường phá vỡ sự tương quan ngữ nghĩa giữa văn bản và hình ảnh. Do đó, tụi em chọn Class Weights làm phương pháp an toàn và đáng tin cậy hơn."

**3. What is TVCS? (TVCS là gì?)**
> "TVCS (Topic-guided Visual Contradiction Score) là một module attention, nó dùng embeddings của KG như một 'chủ đề dẫn hướng' (query) để rà quét các vùng (patches) trên hình ảnh, từ đó tính ra một điểm số biểu thị mức độ mâu thuẫn trực quan."

**4. What is no_c_emb? (no_c_emb là gì?)**
> "Dạ, `no_c_emb` là phiên bản ablation study mà tụi em bỏ đi nhúng phân loại (classification embedding) gốc của baseline. Mục đích là ép mô hình phải học các đặc trưng tương tác mới từ module Transformer thay vì phụ thuộc vào các vector học sẵn trước đó."

**5. Why does CK precision improve but recall drop? (Tại sao Precision của lớp CK tăng nhưng Recall lại giảm?)**
> "Dạ do cơ chế TVCS tạo ra bộ lọc rất khắt khe. Nó chỉ phân loại là CK khi tìm thấy mâu thuẫn rõ ràng, giúp dự đoán rất chắc chắn (Precision tăng). Bù lại, nó sẽ bỏ lỡ các mẫu mâu thuẫn tinh vi hoặc thiếu tri thức bổ trợ (Recall giảm)."

**6. Does the system claim SOTA? (Hệ thống có tự nhận là State-of-the-art không?)**
> "Dạ không ạ. Nhóm em không hướng tới điểm Accuracy SOTA, vì bài toán 6 nhãn trên tập dữ liệu này quá khó và nhiễu. Mục tiêu của tụi em là đề xuất một cơ chế (TVCS) để **tăng khả năng diễn giải (interpretability)** và cải thiện độ tin cậy của mô hình."
