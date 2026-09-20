# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Trần Kim Phương
**Nhóm:** Transformer
**Ngày:** 20/9/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (High cosine similarity) nghĩa là 2 đoạn văn bản có các vector embedding hướng tương đồng nhau. Nội dung và ý nghĩa của 2 đoạn khá giống nhau, mặc dù câu chữ không nhất thiết phải giống hệt nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: Khách hàng có thể đổi trả sản phẩm trong vòng 7 ngày
- Câu B: Người mua có quyền đổi trả món hàng trong vòng 7 ngày
- Tại sao tương đồng: Mặc dù câu chữ không hoàn toàn như nhau nhưng hai câu cùng diễn đạt quyền đổi trả trong một khoảng thời gian nhất định

**Ví dụ có độ tương tự THẤP:**
- Câu A: Khách hàng phải xuất trình hoá đơn gốc khi đổi trả hàng
- Câu B: Cửa hàng dự kiến đóng cửa trong 3 ngày Tết
- Tại sao khác: Hai câu diễn đạt ý nghĩa khác nhau

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Vì độ tương tự cosine tập trung vào hướng của vector, trong khi khoảng cách Euclid sẽ bị ảnh hưởng bởi độ lớn của 2 vector. Vì vậy 2 đoạn văn bản có ý nghĩa tương tự vẫn có thể bị đánh giá là cách xa nhau nếu độ lớn của 2 vector có sự khác biệt lớn.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Bước nhảy giữa hai chunk liên tiếp là `500 - 50 = 450` ký tự.
> Số lượng chunk là `ceil((10,000 - 50) / (500 - 50)) = ceil(9,950 / 450) = ceil(22.11) = 23`.
> **Đáp án: 23 chunks.**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100, bước nhảy giảm còn `500 - 100 = 400` ký tự, nên số lượng chunk là `ceil((10,000 - 100) / 400) = ceil(24.75) = 25`, tức tăng từ 23 lên 25 chunks. Overlap lớn hơn giúp bảo toàn ngữ cảnh và hạn chế việc thông tin bị cắt ở ranh giới giữa hai chunk, nhưng đồng thời làm tăng dữ liệu trùng lặp, chi phí embedding, lưu trữ và truy xuất.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Dùng regex `(?<=[.!?])\s+` để tách tại khoảng trắng ngay sau dấu kết thúc câu, nhờ đó các dấu `.`, `!`, `?` vẫn được giữ lại trong câu đứng trước. Sau khi loại bỏ các phần rỗng và khoảng trắng thừa, tôi gom tối đa `max_sentences_per_chunk` câu vào mỗi chunk; đầu vào rỗng hoặc chỉ có khoảng trắng trả về danh sách rỗng. Cách này chưa xử lý hoàn hảo chữ viết tắt (ví dụ `TS.`) hoặc số thập phân vì chúng có thể bị nhận nhầm là ranh giới câu.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Tôi lần lượt thử các dấu phân cách theo mức độ ưu tiên `"\n\n"`, `"\n"`, `". "`, `" "`, rồi đệ quy với dấu phân cách nhỏ hơn nếu một mảnh vẫn vượt quá `chunk_size`. Các base case gồm: văn bản rỗng trả về `[]`, văn bản đã đủ ngắn trả về một chunk, còn khi hết separator (hoặc gặp separator rỗng) thì cắt cứng theo số ký tự. Sau khi tách, tôi ghép các mảnh nhỏ liền kề đến gần giới hạn kích thước để tránh tạo nhiều chunk vụn, đồng thời giữ nguyên separator để có thể khôi phục chính xác văn bản gốc.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Với `add_documents`, mỗi `Document` được chuyển thành một record trong bộ nhớ gồm `id`, `content`, bản sao `metadata` và vector embedding; metadata luôn có `doc_id` để truy vết tài liệu gốc. Khi tìm kiếm, tôi embedding câu truy vấn một lần, tính tích vô hướng giữa vector truy vấn và từng vector đã lưu, sau đó sắp xếp điểm giảm dần và lấy tối đa `top_k` kết quả. Vector embedding không được đưa vào kết quả trả về để output gọn hơn.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` lọc tập ứng viên theo tất cả cặp khóa–giá trị trong `metadata_filter` trước, rồi mới tính độ tương tự và xếp hạng; cách này tránh để các tài liệu không phù hợp chiếm mất vị trí trong top-k. Cả tìm kiếm thường và tìm kiếm có bộ lọc đều dùng chung `_search_records` để bảo đảm kết quả nhất quán. `delete_document` duyệt store, loại bỏ toàn bộ chunk có `metadata['doc_id']` trùng với mã cần xóa và trả về `True` chỉ khi số record thực sự giảm.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Tôi triển khai `answer` theo ba bước của RAG: gọi `store.search(question, top_k)` để lấy các chunk liên quan, ghép chúng thành ngữ cảnh được đánh số `[1]`, `[2]`, ... kèm nguồn, rồi truyền prompt hoàn chỉnh cho `llm_fn`. Nguồn được lấy lần lượt từ `source_url`, `source`, `doc_id` hoặc `id`; prompt yêu cầu mô hình chỉ dùng ngữ cảnh được cung cấp, nói rõ khi thiếu thông tin và trích dẫn số chunk khi có thể. Nếu không truy xuất được chunk nào, hàm trả về thông báo không tìm thấy thông tin và không gọi LLM, nhờ đó tránh sinh câu trả lời không có căn cứ.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
# Dán kết quả (output) của: pytest tests/ -v
=============================== test session starts ================================
platform darwin -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0 -- /Users/mac14/HomeSpace/mainapps/K4-DAY07-TranKimPhuong-2A202602565/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /Users/mac14/HomeSpace/mainapps/K4-DAY07-TranKimPhuong-2A202602565
plugins: anyio-4.15.1
collected 42 items                                                                 

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED[  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED       [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED  [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED        [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possiblePASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separatorPASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED       [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED  [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

================================ 42 passed in 0.05s ================================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Khách hàng có thể đổi trả sản phẩm trong vòng 7 ngày kể từ khi nhận hàng. | Người mua được yêu cầu hoàn lại món hàng trong thời hạn một tuần sau khi nhận. | Cao | 0,749 (cao) | Đúng |
| 2 | PyLoHerb hỗ trợ vận chuyển đơn hàng từ công ty đến đối tác bán hàng. | Công ty chịu trách nhiệm giao đơn hàng tới các đại lý của PyLoHerb. | Cao | 0,835 (cao) | Đúng |
| 3 | Khách hàng được đổi sản phẩm trong vòng 7 ngày. | Khách hàng không được đổi sản phẩm trong vòng 7 ngày. | Thấp | 0,805 (cao) | Không |
| 4 | Khách hàng phải cung cấp số tài khoản ngân hàng để nhận tiền hoàn trả. | Nhà bán hàng được hưởng mức chiết khấu 20% khi đạt doanh số quy định. | Thấp | 0,296 (thấp) | Đúng |
| 5 | Phí vận chuyển đổi trả do khách hàng thanh toán. | Doanh nghiệp thanh toán toàn bộ phí vận chuyển đổi trả cho khách hàng. | Thấp | 0,774 (cao) | Không |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là cặp 3 đạt điểm 0,805 dù từ “không” làm hai câu có ý nghĩa trái ngược nhau; cặp 5 cũng có hiện tượng tương tự khi chủ thể chịu phí bị đảo ngược. Điều này cho thấy embeddings nắm bắt rất tốt chủ đề và các từ khóa chung, nhưng cosine similarity không trực tiếp kiểm tra quan hệ phủ định, mâu thuẫn hay tính đúng sai của một phát biểu; vì vậy trong bảng này tôi quy ước điểm từ 0,5 trở lên là cao và dưới 0,5 là thấp.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Đối với người bán hoặc đại lý của PyLoHerb, chi phí vận chuyển và bốc dỡ được hỗ trợ như thế nào? Dùng `metadata_filter={"audience": "seller"}`. | Mục **III. Hỗ trợ đại lý**: PyLoHerb hỗ trợ vận chuyển đơn hàng từ công ty đến đối tác bán hàng; chi phí bốc dỡ mỗi bên chịu một đầu. | 0,737 | Có | PyLoHerb hỗ trợ vận chuyển đến đối tác bán hàng và mỗi bên chịu chi phí bốc dỡ ở một đầu. |
| 2 | Theo chính sách HiClean, khách hàng đổi hàng từ ngày 04 đến ngày 07 phải chịu mức phí bao nhiêu? | Mục **Các trường hợp được trả hàng** của HiClean: thời hạn trả hàng là 3 ngày và phí trả hàng là 10%; chunk không chứa quy định đổi hàng từ ngày 04 đến ngày 07. | 0,659 | Không | Agent trả lời sai rằng từ ngày 04 trở đi không được hỗ trợ, thay vì mức phí đổi hàng 20% cùng các chi phí liên quan. |
| 3 | Với đơn hàng COD tại Smart HomeKit, người mua phải cung cấp thông tin gì để nhận tiền hoàn trả? | Mục **II. Quy định hoàn tiền**: tiền có thể được chuyển vào thẻ tín dụng hoặc tài khoản ngân hàng được chỉ định, nhưng không chứa quy định riêng cho đơn COD về việc cung cấp thông tin tài khoản ngân hàng. | 0,647 | Không | Agent cho biết ngữ cảnh chưa đủ thông tin để xác định yêu cầu riêng đối với đơn COD. |
| 4 | Theo chính sách 102 ngày ngủ thử Tatana của Thế Giới Nệm, thời gian xử lý hoàn tiền là bao lâu? | Mục **3. Chính sách 102 ngày ngủ thử sản phẩm Tatana**: hoàn tiền trong vòng 14 ngày làm việc kể từ khi nhận lại sản phẩm và xác nhận sản phẩm đạt điều kiện. | 0,749 | Có | Hoàn tiền trong vòng 14 ngày làm việc kể từ khi Thế Giới Nệm nhận lại sản phẩm và xác nhận đạt điều kiện. |
| 5 | Khách mua hàng online tại XTREND được yêu cầu đổi trả trong thời gian bao lâu? | Mục **Chính sách kiểm hàng**: khách có thể yêu cầu đổi trả khi thay đổi nhu cầu và phải trả phí giao hàng nếu có; chunk không nêu thời hạn đổi trả. | 0,619 | Không | Agent cho biết không đủ thông tin để xác định mốc 07 ngày. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 2 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Qua phần so sánh của nhóm, tôi nhận ra chunk dễ đọc và giữ được heading chưa chắc có thứ hạng truy xuất tốt nhất. Chiến lược FixedSize của Nguyễn Minh Thái tận dụng overlap để thông tin ở biên có thêm cơ hội xuất hiện trong top-3, còn Recursive của Trần Gia Thành tạo các đoạn mạch lạc mà không lặp heading quá nhiều; vì vậy một chiến lược lai giữa heading, recursive và overlap nhỏ có thể cân bằng ngữ cảnh với chất lượng truy xuất tốt hơn.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) |10 / 10 |
| **Tổng phần cá nhân** | ** 60 / 60** |
