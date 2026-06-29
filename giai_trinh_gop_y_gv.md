# BẢN GIẢI TRÌNH CÁC GÓP Ý CỦA GIẢNG VIÊN (RESPONSE TO FEEDBACK)

Dạ em chào cô, nhóm chúng em xin gửi bản giải trình chi tiết về cách nhóm đã tiếp thu và chỉnh sửa báo cáo dựa trên 6 điểm góp ý của cô ạ:

---

### 1. "Thay bằng phân bố data sau khi lọc KG"
👉 **Vị trí trong báo cáo:** Section III (Dataset) - Phần mô tả thống kê dữ liệu.

**Giải trình:** 
Nhóm đã thay thế các bảng/biểu đồ phân bố dữ liệu gốc bằng phân bố dữ liệu thực tế sau khi đã qua bước lọc Knowledge Graph (tập `kg_complete`). 
- Cụ thể, tập FineFake gốc có **16,909** mẫu. Nhóm phát hiện có **4,041** mẫu bị thiếu thực thể hoặc có vector KG bằng 0. 
- Để đảm bảo mô hình thực sự học được từ tri thức, nhóm đã loại bỏ hoàn toàn **4,123** mẫu không hoàn thiện này khỏi giao thức thử nghiệm. 
- Tập dữ liệu cuối cùng đưa vào huấn luyện chỉ còn **12,786** mẫu. Trong đó, số lượng mẫu của lớp *Real* là **6,298** và lớp *CK (Content-Knowledge Inconsistency)* là **1,211**. Sự phân bố mới này phản ánh chính xác dữ liệu mà hệ thống thực sự sử dụng.

### 2. "Tiền xử lý data cho rõ"
👉 **Vị trí trong báo cáo:** Section III (Dataset) - Mục Data Preprocessing.

**Giải trình:** 
Nhóm đã viết lại và làm rõ phần Tiền xử lý dữ liệu trong báo cáo thành các bước logic, minh bạch hơn:
- Làm rõ cách trích xuất đặc trưng văn bản (Text Features) và hình ảnh (Global & Patch Features).
- Trình bày chi tiết cách trích xuất và đối chiếu các thực thể từ Knowledge Graph để tạo ra tập `kg_complete`, giải thích rõ lý do tại sao một số mẫu bị loại bỏ ở bước này.

### 3. "Xử lý mất cân bằng data trên dataset trước rồi mới dùng đến trọng số"
👉 **Vị trí trong báo cáo:** Section V (Experiments) - Mục Experimental Setup / Implementation Details.

**Giải trình:** 
Tiếp thu góp ý của cô, thay vì chỉ can thiệp ở thuật toán, nhóm đã bổ sung cách tiếp cận xử lý mất cân bằng ngay từ mức độ tập dữ liệu (dataset-level). 
- Nhóm đã tính toán **Class Weights** (trọng số lớp) dựa trên nghịch đảo tần suất xuất hiện của từng lớp trong tập `kg_complete` (phân bố dữ liệu sau lọc). 
- Trọng số này được tính toán nghiêm ngặt dựa trên tập huấn luyện (Train set) trước khi đưa vào hàm loss. Điều này giúp cân bằng độ nhạy của mô hình, giúp nó không bị thiên vị lớp đa số (*Real*) mà vẫn học tốt lớp thiểu số (*CK*). Nhóm quyết định không dùng SMOTE (oversampling) vì các kỹ thuật sinh mẫu trên không gian vector đa phương thức có thể làm hỏng tính đồng nhất ngữ nghĩa giữa văn bản và hình ảnh.

### 4. "Vẽ lại kiến trúc model tổng thể"
👉 **Vị trí trong báo cáo:** Section IV (Method) - Các Hình ảnh tổng thể (Figures 1-4).

**Giải trình:** 
Nhóm đã vẽ lại toàn bộ sơ đồ kiến trúc tổng thể của mô hình CIKD++ để minh họa luồng dữ liệu một cách trực quan và mạch lạc hơn. 
Sơ đồ mới thể hiện rõ ràng luồng đi của 3 nhánh đa phương thức (Văn bản, Hình ảnh, KG) tương tác với nhau như thế nào qua cơ chế *Residual Transformer*, và cách chúng được dẫn vào module *Topic-guided Visual Contradiction Score (TVCS)* để đưa ra dự đoán cuối cùng.

### 5. "Thêm giải thích kỹ tại sao fake, từ đâu ra cái score đó"
👉 **Vị trí trong báo cáo:** Section VI (Results) - Mục Qualitative Analysis (Ablation / TVCS Evidence) và cập nhật trực tiếp trên **Ứng dụng Demo**.

**Giải trình:** 
Nhóm đã bổ sung tài liệu giải thích chi tiết về trực giác của mô hình (được hiển thị trực tiếp khi chạy Demo):
- **Tại sao lại là Fake (Mâu thuẫn)?** Hệ thống dùng KG làm "hệ quy chiếu" thực tế. Nếu các chi tiết xuất hiện trong hình ảnh (bối cảnh, nhân vật) không khớp với tri thức nền từ KG, mô hình sẽ nhận diện đó là sự mâu thuẫn (Contradiction - nhãn CK).
- **Điểm số TVCS từ đâu ra?** Điểm TVCS là kết quả từ cơ chế Attention. Mô hình dùng vector KG như một câu truy vấn (query) để rà quét 49 vùng nhỏ (7x7 patches) trên hình ảnh. Vùng ảnh nào sai lệch về mặt ngữ nghĩa so với KG sẽ bị "phạt" và nhận trọng số Attention cao (hiển thị thành vùng sáng trên Heatmap). Điểm TVCS tổng hợp toàn bộ các tín hiệu mâu thuẫn này qua hàm Sigmoid. Điểm càng gần 1.0 chứng tỏ mâu thuẫn càng mạnh.

### 6. "Giải thích các từ viết tắt, definition…"
👉 **Vị trí trong báo cáo:** Section I (Introduction) - Bổ sung mục "Terminology and Definitions" (Thuật ngữ và Định nghĩa).

**Giải trình:** 
Theo định hướng của cô, nhóm đã bổ sung một mục riêng để định nghĩa các thuật ngữ ngay ở phần đầu của báo cáo. 
Mục này liệt kê và giải nghĩa rõ ràng các từ viết tắt (ví dụ: *CK = Content-Knowledge Inconsistency*, *TVCS = Topic-guided Visual Contradiction Score*, v.v.) cũng như các định nghĩa về 6 nhãn phân loại tin giả. Điều này giúp người đọc dễ dàng theo dõi và nắm bắt chính xác các khái niệm xuyên suốt bài báo.
