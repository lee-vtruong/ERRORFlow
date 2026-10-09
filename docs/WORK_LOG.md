# ERRORFlow Work Log

## Step 1 — Khởi tạo context và đọc workrule

- Thời gian: 2026-10-09 (Asia/Saigon).
- Mục tiêu: xác định cấu trúc workspace, quy tắc hiện tại và trạng thái Git.
- Đã kiểm tra: `workrule.md`, danh sách file, `git status`, `git remote`.
- Kết quả: workspace chưa phải Git repository; chỉ có Script và workrule. Workrule cũ còn trỏ tới GraphCURE và bị lỗi encoding khi hiển thị.
- Quyết định: chuyển quy tắc sang ERRORFlow, giữ workflow local → GitHub → server; thêm nhật ký bắt buộc và provenance.

## Step 2 — Đọc Script ý tưởng

- Thời gian: 2026-10-09 (Asia/Saigon).
- Mục tiêu: hiểu hướng nghiên cứu trước khi dựng code.
- Đã kiểm tra: toàn bộ paragraph/table text trong `Script - Idea ERRORFlow.docx` bằng `python-docx`.
- Kết quả chính: training-free fact verification với frozen LLM; error memory có diagnosis/root cause/intervention; kết hợp Counterfactual Error Memory với Error-Conditioned Adaptive Multi-Agent Workflow; tối ưu accuracy-cost; cần recovery rate, regression rate, token/call/latency; nghiêm cấm leakage từ validation/test.
- Dataset đề xuất: FEVER cho prototype, AVeriTeC cho paper, SciFact cho domain transfer; cần systematic literature review trước khi chốt novelty.
- Hạn chế QA: render bằng `render_docx.py` không thực hiện được vì máy thiếu `soffice`/LibreOffice (`WinError 2`). Nội dung đã trích xuất được, nhưng chưa xác nhận trực quan layout.

## Step 3 — Kiểm tra môi trường/model hiện có

- Thời gian: 2026-10-09 (Asia/Saigon).
- Mục tiêu: xem có thể dùng lại GraphCURE environment/model hay không.
- Đã kiểm tra: `C:\Users\ASUS\Downloads\GraphCURE`, các thư mục venv gần workspace, dấu vết model trong mã/tài liệu.
- Kết quả: GraphCURE tồn tại và có code/docs liên quan Qwen3/Qwen2.5/Mistral, nhưng không tìm thấy `whale/GraphCURE/.venv` trên local. Chưa có quyền truy cập/đầu ra kiểm tra server nên chưa thể xác nhận model weights/cache trên server.
- Quyết định: ERRORFlow dùng `~/whale/ERRORFlow/.venv` mặc định; chỉ tái sử dụng `.venv` GraphCURE sau khi kiểm tra dependency trên server. Trước khi tải model mới phải kiểm tra Hugging Face cache/Ollama/vLLM.

## Step 4 — Cập nhật workrule

- Thời gian: 2026-10-09 (Asia/Saigon).
- File sửa: `workrule.md`.
- Kết quả: cập nhật repository/workspace/server path, quy trình GitHub, venv, kiểm tra model, nhật ký bắt buộc, provenance và checklist nghiên cứu.
- Việc còn thiếu: khởi tạo Git repository/remote, tạo skeleton code và xác nhận quyền truy cập GitHub/server.

## Step 5 — Khởi tạo skeleton và Git remote

- Thời gian: 2026-10-09 (Asia/Saigon).
- Đã tạo: `README.md`, `pyproject.toml`, package `src/errorflow`, schema `ErrorMemoryEntry`, test mẫu, `.gitignore` và hướng dẫn scripts.
- Đã chạy: `git init -b main`.
- Kết quả: Git repository local đã được khởi tạo. Lệnh Git tiếp theo bị chặn bởi cơ chế `safe.directory` do workspace thuộc user Windows khác với sandbox; chưa tự ý sửa Git global config.
- Kiểm thử lần đầu: pytest không import được package vì skeleton chưa cài editable; đã thêm `tests/conftest.py` để test local không cần cài package.
- Remote: đã cấu hình thành công `origin=https://github.com/lee-vtruong/ERRORFlow.git`. Do `.git/config` khác owner, thao tác này cần quyền phù hợp; chưa commit/push.

## Step 6 — Phân tích output kiểm tra server do người dùng cung cấp

- Thời gian: 2026-10-09 (Asia/Saigon).
- Nguồn: output từ `~/whale/GraphCURE/.venv`, `nvidia-smi`, Hugging Face cache và GraphCURE outputs.
- Môi trường: Python 3.11.15; GraphCURE editable `0.1.0`; đã có PyTorch 2.13.0, Transformers 5.15.0, Accelerate 1.14.0, PEFT 0.20.0, Datasets 5.0.1, FAISS-CPU 1.15.0, sentence-transformers 5.7.0, scikit-learn 1.9.0, pytest 9.1.1.
- Phần cứng: NVIDIA RTX 5090 32 GB, driver 580.159.03, CUDA 13.0; GPU đang rảnh.
- Model/cache đã thấy: Vietnamese Embedding, Vietnamese bi-encoder, roberta-base và một số model hình ảnh. Không thấy Ollama hoặc vLLM CLI. Cache chưa chứng minh có các backbone Qwen3/Qwen2.5/Mistral; GraphCURE có nhiều LoRA adapter và model artifacts trong `outputs/`, `models/`, `checkpoints/`.
- Đánh giá: có thể dùng `GraphCURE/.venv` làm môi trường thử nghiệm ban đầu vì đã có stack ML phù hợp, nhưng không nên cài thêm package hoặc sửa environment trước khi kiểm tra import/version CUDA thực tế. ERRORFlow vẫn nên có venv riêng nếu dependency bắt đầu lệch.
- Việc còn thiếu: xác định chính xác base model tương ứng với các adapter, kích thước file, và kiểm tra `torch.cuda.is_available()`/model loading trước khi thiết kế reuse adapter cho ERRORFlow.

## Step 7 — Xác nhận CUDA và khả năng tái sử dụng môi trường

- Thời gian: 2026-10-09 (Asia/Saigon).
- Kết quả xác nhận: `torch 2.13.0+cu130`, CUDA available `True`, CUDA runtime `13.0`, Transformers `5.15.0`, PEFT `0.20.0`, GPU `NVIDIA GeForce RTX 5090`.
- Dung lượng: `GraphCURE/models` 88 MB, `checkpoints` 28 KB, `outputs` 36 GB.
- Hugging Face cache: chỉ xác nhận `AITeamVN/Vietnamese_Embedding`; chưa thấy cache Qwen/Mistral/Llama.
- Quyết định: dùng chung `~/whale/GraphCURE/.venv` cho prototype đầu tiên để tránh cài lại stack ML; chưa cài thêm package. Khi ERRORFlow cần dependency riêng hoặc thay đổi version, tạo `~/whale/ERRORFlow/.venv`.
- Cảnh báo: 36 GB trong `outputs` chủ yếu là artifact/adapter của GraphCURE; cần truy vết `adapter_config.json` và tên base model trước khi reuse, không suy luận base model chỉ từ tên thư mục.
## Step 8 — Nhận danh sách checkpoint/adapter từ server

- Thời gian: 2026-10-09 (Asia/Saigon).
- Kết quả: GraphCURE có nhiều adapter Qwen3, các nhánh control, teacher Mistral/teacher capacity và checkpoint ModernBERT/NLI.
- Chưa thể xác định base model chỉ từ đường dẫn; cần đọc nội dung `adapter_config.json` và `config.json` của các artifact đại diện.
- Ưu tiên kiểm tra: `mocheg_qwen3_lora_seed42_v16`, `mocheg_packet_qwen3_seed42`, `mocheg_b23_teacher_family/candidate_mistral7b_seed42`, `mocheg_modernbert_text_seed42_v12`.
- Quyết định tạm thời: không tải model mới và không copy 36 GB outputs; chỉ reuse adapter sau khi xác nhận `base_model_name_or_path`, kiến trúc, tokenizer và protocol dữ liệu.

## Step 9 — Implement core baseline

- Mục tiêu: tạo lõi ERRORFlow chạy được không phụ thuộc model/API.
- Đã làm: schema error memory, JSONL persistence, cost-aware heuristic router, transition metrics và test suite.
- Tài liệu: `docs/STEP_01_CORE_BASELINE.md`.
- Kiểm thử dự kiến: schema test cũ + 3 test core mới.
- Bước tiếp theo: commit/push skeleton, sau đó tạo data contract cho baseline predictions và error diagnosis trước khi kết nối Qwen3.

## Step 10 — Implement prediction/trace contracts

- Mục tiêu: chuẩn hóa artifact verifier và execution trace trước khi tích hợp Qwen3.
- Đã làm: `PredictionRecord`, `ExecutionTrace`, demo fixture và 2 test contract.
- Tài liệu: `docs/STEP_02_DATA_CONTRACTS.md`.
- Chưa phải kết quả thực nghiệm: fixture deterministic chỉ dùng kiểm tra format.
