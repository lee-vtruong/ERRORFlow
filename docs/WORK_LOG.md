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

## Step 11 — Implement verifier adapter contract

- Mục tiêu: khóa prompt/label/provenance interface trước khi load Qwen3.
- Đã làm: prompt builder, strict label parser, `VerifierConfig` và record builder.
- Không load model hoặc gọi network ở step này.
- Tài liệu: `docs/STEP_03_VERIFIER_ADAPTER.md`.

## Step 12 — Implement Qwen3 baseline runner

- Mục tiêu: tích hợp frozen Qwen3 inference mà không đưa adapter LoRA vào baseline đầu tiên.
- Đã làm: lazy model/tokenizer loading, greedy decoding, JSONL runner và smoke-test contract.
- Tài liệu: `docs/STEP_04_QWEN3_BASELINE.md`.
- Chưa chạy model trong local test; cần server model path thật sau khi pull.

## Step 13 — Qwen3 smoke test blocked by missing local model path

- Thời gian: 2026-10-09 (Asia/Saigon).
- Lệnh đã chạy: `scripts/run_qwen3_baseline.py --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 ...`.
- Kết quả: thất bại trước khi load model; `transformers` không tìm thấy local directory và chuyển path thành repo id, dẫn tới `HFValidationError`.
- Chẩn đoán: base model chưa được tải/materialized tại path được truyền vào; chưa phải lỗi CUDA, tokenizer hay parser.
- Việc cần làm: tải `Qwen/Qwen3-4B-Instruct-2507` vào đúng local directory, kiểm tra `config.json`, rồi chạy lại smoke test.

## Step 14 — Qwen3 base model downloaded

- Thời gian: 2026-10-09 (Asia/Saigon).
- Kết quả server: `hf download Qwen/Qwen3-4B-Instruct-2507` hoàn tất; local path có `config.json`, tokenizer files, index và 3 shard `model-*.safetensors`.
- Cảnh báo lock: xuất hiện trong lúc tải nhưng download/reconstruction hoàn tất, không xem là lỗi hiện tại.
- HF auth: đang dùng unauthenticated request; nếu bị rate limit ở các lần tải sau thì dùng `hf auth login`/`HF_TOKEN`.
- Bước tiếp theo: chạy smoke inference và ghi prediction output; chưa benchmark.

## Step 15 — Qwen3 smoke inference thành công

- Kết quả server: Qwen3 load weights thành công và dự đoán `demo_001 → SUPPORTED`.
- Output: `outputs/qwen3_baseline/demo_predictions.jsonl`.
- Provenance: base model local Qwen3-4B-Instruct-2507, split `smoke`, 8 generated tokens.
- Quan sát: raw output chứa hai lần `SUPPORTED` (`SUPPORTED` và `Label: SUPPORTED`), parser vẫn xử lý đúng; cần giữ raw output để audit.
- Sửa tiếp theo: thay `torch_dtype` deprecated bằng `dtype` và đo latency inference thực tế.

## Step 16 — Download FEVER raw splits

- Nguồn: official `fever.ai/download/fever/` URLs.
- Đã tải trên server: `train.jsonl` (31.49 MB), `shared_task_dev.jsonl` (4.15 MB), `paper_dev.jsonl` (2.07 MB).
- Chưa chạy Qwen trên FEVER; cần audit schema/label distribution trước.
- Cảnh báo protocol: raw FEVER claim files chứa evidence references, không mặc định chứa passage text. Không được đưa reference IDs vào prompt như evidence giả.

## Step 17 — FEVER audit and label normalization

- Kết quả audit: train 145,449; shared-task dev 19,998; paper dev 9,999.
- Train bị lệch lớp: SUPPORTS 80,035; REFUTES 29,775; NOT ENOUGH INFO 35,639. Hai dev split cân bằng.
- Đã thêm mapping explicit `SUPPORTS → SUPPORTED`, `REFUTES → REFUTED`; unknown labels fail closed.
- Tài liệu: `docs/STEP_07_FEVER_LABEL_NORMALIZATION.md`.
- Insight: sẽ dùng Macro-F1/per-class metrics; chưa chạy main inference vì evidence passage text chưa được resolve.

## Step 18 — Implement FEVER evidence resolver

- Đã làm: trích xuất wiki page/sentence references và resolve sentence text từ FEVER wiki-pages JSONL.
- Chưa tải corpus tự động vì kích thước lớn; cần kiểm tra server trước để tránh tải trùng GraphCURE.
- Tài liệu: `docs/STEP_08_FEVER_EVIDENCE_RESOLUTION.md`.

## Step 19 — Implement FEVER subset preparation

- Đã làm: script chọn claim thật, quét wiki shards, resolve sentence evidence, normalize label và ghi dataset provenance.
- Output dự kiến: `data/processed/fever/*.jsonl`; không commit vào Git.
- Tài liệu: `docs/STEP_09_PREPARE_FEVER_SUBSET.md`.
- Bước tiếp theo: chạy subset paper-dev 100 records, kiểm tra schema/evidence rồi mới inference Qwen3.

## Step 20 — Qwen3 baseline trên FEVER subset

- Kết quả server: prediction JSONL đã sinh cho claim thật từ `paper_dev_first100.jsonl`; output có model path, split, token count và latency.
- Quan sát: raw output có reasoning dù prompt yêu cầu label; một số record không evidence là NEI hợp lệ. Không đánh giá bằng mắt hoặc dùng raw reasoning để sửa nhãn.
- Đã thêm `scripts/evaluate_predictions.py` để join gold/prediction và tính Accuracy, Macro-F1, per-class F1, confusion, latency/token averages.
- Tài liệu: `docs/STEP_10_EVALUATE_BASELINE.md`.

## Step 21 — Baseline metrics insight

- N=99 paired claims; Accuracy `0.63636`; Macro-F1 `0.57322`.
- F1: SUPPORTED `0.76056`, REFUTED `0.68132`, NOT ENOUGH INFO `0.27778`.
- Confusion nổi bật: 14/26 NEI bị dự đoán REFUTED, 7/26 NEI bị dự đoán SUPPORTED; chỉ 5/26 đúng.
- Latency trung bình `772.36 ms/claim`; generated tokens trung bình `31.70`.
- Insight ban đầu: bottleneck rõ nhất là phân biệt NEI với hai lớp còn lại; chưa kết luận nguyên nhân retrieval/reasoning từ 99 mẫu.
- Đã thêm `scripts/build_error_list.py` và `docs/STEP_11_ERROR_LIST.md` để tạo error records observable.
## Step 22 — Qwen3 baseline trên train subset 1,000

- Input dự kiến 1,000 nhưng output còn 995 claims do 5 positive claims không resolve được evidence và bị loại theo policy.
- Accuracy `0.71357`; Macro-F1 `0.62149`.
- F1: SUPPORTED `0.86034` (support 541), REFUTED `0.56209` (189), NOT ENOUGH INFO `0.44205` (265).
- Error list: `285` records.
- Confusion nổi bật: NEI→REFUTED `107`, NEI→SUPPORTED `76`; REFUTED→SUPPORTED `44`.
- Latency trung bình `762.59 ms/claim`; generated tokens `31.74`.
- Diễn giải: baseline tốt hơn trên train subset nhưng phân bố khác paper-dev; đây là dữ liệu để xây memory, không dùng làm final generalization claim.
- Bước tiếp theo: thống kê observable error types và evidence count của 285 lỗi trước khi diagnosis.
## Step 23 — Phát hiện protocol leakage trong FEVER preparation

- Error distribution: `183 nei_no_evidence`; các lỗi còn lại gồm REFUTED→SUPPORTED 44, SUPPORTED→REFUTED 34, REFUTED→NEI 16, SUPPORTED→NEI 8.
- Evidence count của error list: 183 record có 0 evidence, 80 có 1, 19 có 2, 1 có 3, 1 có 6, 1 có 13.
- Phát hiện quan trọng: `prepare_fever_subset.py` đang dùng annotated gold evidence cho SUPPORTS/REFUTES và evidence rỗng cho NEI. Đây là leakage/protocol confound vì evidence availability phụ thuộc gold annotation.
- Trạng thái metrics train 1k: chỉ dùng để debug pipeline, không được báo cáo như kết quả nghiên cứu.
- Quyết định: dừng xây error memory từ artifact hiện tại. Cần retrieval độc lập theo claim, cùng policy cho mọi nhãn, rồi chạy lại baseline trước khi phân tích lỗi.
## Step 24 — Implement independent retrieval baseline

- Đã làm: SQLite FTS5 sentence index builder và BM25 retrieval theo claim.
- Retriever không dùng gold evidence; cùng một policy cho mọi nhãn.
- Tài liệu: `docs/STEP_12_INDEPENDENT_RETRIEVAL.md`.
- Chưa chạy index trên server; đây là job dài và tạo artifact lớn ngoài Git.
## Step 25 — Fix FTS5 query escaping

- Index build thành công: `25,247,890` sentences.
- Retrieval lỗi vì raw claim được truyền trực tiếp vào FTS5 MATCH; token như tên riêng bị hiểu như column/operator.
- Đã sửa query builder: tokenize và quote từng token trước khi MATCH.
- Đã thêm regression test cho claim có tên riêng/dấu câu.
## Step 26 — Add retrieval progress reporting

- Vấn đề: `retrieve_fts.py` không có output trong lúc chạy, gây cảm giác treo.
- Đã sửa: hiển thị tổng số rows, progress, evidence_found, rate, ETA và elapsed time; flush ngay mỗi mốc.
- Index không cần build lại.
## Step 27 — Independent retrieval completed

- Kết quả server: `995/995` claims có evidence; `RETRIEVAL_DONE rows=995 evidence_found=995`.
- Runtime: `9718.4s` (~2h42m), throughput `0.10 claim/s`.
- Protocol: evidence được query độc lập từ SQLite FTS5/BM25, không dùng gold evidence để tạo input verifier.
- Insight kỹ thuật: retrieval hiện đúng về provenance nhưng quá chậm; giữ output làm baseline, tối ưu tốc độ sau khi có verifier metrics.
## Step 28 — Independent-retrieval Qwen3 baseline metrics

- N=995 paired claims; Accuracy `0.62211`; Macro-F1 `0.55315`.
- F1: SUPPORTED `0.77336`, REFUTED `0.47748`, NOT ENOUGH INFO `0.40860`.
- Confusion nổi bật: SUPPORTED→REFUTED 55, SUPPORTED→NEI 68, REFUTED→NEI 37, NEI→REFUTED 94, NEI→SUPPORTED 76.
- Latency `761.69 ms/claim`; generated tokens `31.91`.
- So với gold-evidence input cùng train subset: Accuracy giảm `0.71357 → 0.62211` (-9.15 điểm phần trăm), Macro-F1 giảm `0.62149 → 0.55315` (-6.83 điểm). Đây là bằng chứng thực nghiệm rằng gold-evidence preparation trước đó tạo confound/leakage.
- Quyết định: chỉ dùng run `train_first1000_retrieved` làm baseline chính cho error memory; loại run gold-evidence khỏi kết luận.
## Step 29 — Heuristic diagnosis candidates

- Independent error list có 376 records; mọi record có `evidence_count=5`.
- Đã thêm diagnosis heuristic theo label transition và candidate actions.
- Không gọi LLM diagnosis và không coi diagnosis heuristic là ground truth.
- Tài liệu: `docs/STEP_13_HEURISTIC_DIAGNOSIS.md`.
## Step 30 — Implement intervention runner

- Đã làm: prompt interventions `evidence_critic` và `conflict_check`; runner ghi intervention metadata.
- Error list giờ giữ cả evidence text để có thể reverify đúng context.
- Tài liệu: `docs/STEP_14_INTERVENTION_EVALUATION.md`.
- Chưa chạy interventions trên server.
## Step 31 — Add intervention recovery evaluator

- Đã làm: evaluator đo recovery rate trên baseline error set và số prediction thay đổi.
- Tài liệu: `docs/STEP_15_INTERVENTION_METRICS.md`.
- Chờ output metrics của `evidence_critic` và `conflict_check` từ server.
## Step 32 — Counterfactual intervention results

- `evidence_critic`: recovered 51/376, rate `0.13564`, changed 81.
- `conflict_check`: recovered 20/376, rate `0.05319`, changed 39.
- Đã thêm builder lưu cả intervention outcomes thành counterfactual memory.
- Cảnh báo: chưa đo regression trên baseline-correct claims; chưa chọn policy final.
## Step 33 — Recovery/regression control evaluation

- Đã thêm control subset builder cho baseline-correct claims.
- Đã thêm evaluator tính recovery rate, regression rate và net correct change trên toàn bộ paired set.
- Tài liệu: `docs/STEP_17_RECOVERY_REGRESSION.md`.
- Chưa chạy control interventions trên server.
## Step 34 — Recovery/regression results

- Control set: 619 baseline-correct claims.
- `evidence_critic`: regressed 26/619 (`4.20%`); combined train-subset net change `+51 - 26 = +25`.
- `conflict_check`: regressed 23/619 (`3.72%`); combined train-subset net change `+20 - 23 = -3`.
- Diễn giải: evidence_critic không được áp dụng đại trà; conflict_check bị loại ở phiên bản hiện tại. Cần selective routing theo error type/uncertainty để giảm regression.
## Step 35 — Add selective action analysis

- Đã thêm phân tích recovery của intervention theo `error_type_observable`.
- Tài liệu: `docs/STEP_18_ACTION_BY_ERROR_TYPE.md`.
- Mục tiêu: tìm nhóm lỗi mà evidence_critic có lợi ròng trước khi xây router.
## Step 36 — Evidence critic action analysis

- Hiệu quả cao nhất: NEI→REFUTED recovered 34/94 (`36.17%`).
- Hiệu quả thứ hai: SUPPORTED→REFUTED recovered 7/55 (`12.73%`).
- Các nhóm khác: NEI→SUPPORTED `7.89%`; REFUTED→NEI `2.70%`; REFUTED→SUPPORTED `2.17%`; SUPPORTED→NEI `2.94%`.
- Observable routing hypothesis: chỉ xem xét evidence_critic khi baseline dự đoán REFUTED; không route theo gold-dependent error type.
- Cần đo selective policy trên tất cả claims có baseline prediction REFUTED, gồm cả claim đúng và sai.
## Step 37 — Implement observable selective route subset

- Đã thêm subset builder chọn theo baseline prediction, không dùng gold.
- Hypothesis đầu tiên: trigger evidence_critic khi baseline dự đoán REFUTED.
- Tài liệu: `docs/STEP_19_SELECTIVE_REFUTED_ROUTE.md`.
## Step 38 — Selective REFUTED route subset

- Route size: 255/995 claims (`25.63%`) có baseline prediction `REFUTED`.
- Đã thêm merge/evaluation workflow để chỉ thay prediction ở route subset và giữ baseline ở phần còn lại.
- Tài liệu: `docs/STEP_20_EVALUATE_SELECTIVE_ROUTE.md`.
## Step 39 — Selective REFUTED route result

- Applied route: 255/995 claims; baseline giữ nguyên 740 claims.
- Accuracy: `0.62211 → 0.64623` (+2.41 percentage points).
- Macro-F1: `0.55315 → 0.58459` (+3.14 points).
- F1: SUPPORTED `0.77336 → 0.77063`; REFUTED `0.47748 → 0.49171`; NEI `0.40860 → 0.49143`.
- Kết luận train-side: policy `prediction=REFUTED → evidence_critic` có lợi ròng trên subset này.
- Cảnh báo: chưa generalize; retrieval hiện chỉ `0.10 claim/s`, cần tối ưu trước khi chạy dev/test lớn.
## Step 40 — Add faster FTS retrieval mode

- Đã thêm mode `and_or`: thử AND trước, fallback OR; giới hạn query token mặc định 8.
- Mỗi output ghi mode/token limit/k để provenance.
- Đây là protocol retrieval mới; cần smoke benchmark trên cùng claims trước khi thay baseline chính.
## Step 41 — Fast FTS retrieval benchmark

- `and_or` mode completed `995/995` claims with evidence.
- Runtime `8350.4s` (~2h19m), versus prior `9718.4s` (~2h42m): ~14.1% faster.
- Still too slow for 10k+ dev/test; output is a separate retrieval protocol and must be evaluated separately.
- Next: run Qwen baseline on fast output to measure quality change before further optimization.
## Step 42 — Reject fast retrieval mode on quality regression

- Fast `and_or` retrieval baseline: Accuracy `0.59698`, Macro-F1 `0.53955`.
- Prior OR retrieval baseline: Accuracy `0.62211`, Macro-F1 `0.55315`.
- Quality loss: Accuracy `-2.52` points; Macro-F1 `-1.36` points.
- Decision: do not use `and_or` output for memory/router. Keep original OR output as current valid baseline; optimize speed with a method that preserves retrieval quality.
## Step 43 — Parallelize OR retrieval without changing protocol

- Đã thêm `--workers`; mỗi worker dùng SQLite read-only connection.
- Query mode/chất lượng giữ nguyên nếu chạy `--mode or`.
- Output ghi worker count để provenance.
- Cần benchmark `workers=4` trên cùng 995 claims trước khi dùng cho dev.
## Step 44 — Parallel OR retrieval benchmark

- `workers=4`, `mode=or`, 995/995 claims có evidence.
- Runtime `2448.6s` (~40.8 phút), so với sequential OR `9718.4s` (~161.97 phút): nhanh hơn ~74.8%, throughput ~0.41 claim/s.
- Cần kiểm tra deterministic equality với output sequential trước khi dùng parallel output làm baseline chính.
## Step 45 — Detect nondeterministic parallel retrieval ordering

- Equality check: parallel output matched sequential on `667/995`; `328` differed.
- Diagnosis: BM25 ties had no stable secondary ordering under concurrent connections.
- Fix: add deterministic `sentences_fts.rowid` tie-breaker after BM25 score.
- Must regenerate both reference and parallel outputs before comparison; old outputs are not comparable artifacts.
## Step 46 — Deterministic parallel retrieval validated

- Sequential vs parallel deterministic evidence equality: `995/995`; `different=0`.
- Workers=4 được chấp nhận để tăng tốc mà không đổi evidence/protocol.
- Artifact cũ không deterministic không dùng cho báo cáo; từ đây dùng deterministic output.
## Step 47 — Establish canonical deterministic retrieval baseline

- Comparison: old_or vs new_or same `667/995`; old_or vs fast same `645/995`; new_or vs fast same `963/995`.
- Diagnosis: old_or was generated before stable rowid tie-breaker and is not reproducible; new_or is the canonical OR protocol.
- Canonical baseline metrics currently: Accuracy `0.59698`, Macro-F1 `0.53955`.
- Decision: invalidate old_or-derived error memory/intervention comparisons for final claims. Rebuild error list and interventions from deterministic baseline.
## Step 48 — Deterministic evidence critic intervention

- Deterministic baseline: 401 errors, 594 correct claims.
- Evidence critic recovered 65/401 (`16.21%`).
- Regression on correct control: 36/594 (`6.06%`).
- Net change: `+29` correct predictions if applied to all claims.
- Decision: continue with selective route `baseline prediction=REFUTED`; do not apply globally.
## Step 49 — Deterministic selective route result

- Baseline deterministic: Accuracy `0.59698`, Macro-F1 `0.53955`.
- Selective route: Accuracy `0.62613`, Macro-F1 `0.57279`.
- Gains: Accuracy `+2.91` percentage points; Macro-F1 `+3.32` points.
- F1 change: SUPPORTED `0.74685→0.74788`, REFUTED `0.46764→0.47486`, NEI `0.40417→0.49564`.
- Policy candidate remains `baseline prediction=REFUTED → evidence_critic`.
- Added claim-only FEVER preparation for leakage-safe shared-task dev evaluation.
## Step 50 — Shared-task dev retrieval completed

- Prepared 1,000 claim-only records from `shared_task_dev` without gold evidence.
- Deterministic OR retrieval completed: `1000/1000` records with evidence.
- Runtime `2578.9s` (~43.0 minutes), workers=4, k=5.
- Next: frozen Qwen3 baseline, then apply the train-frozen REFUTED→evidence_critic route without policy changes.
## Step 51 — Shared-task dev baseline

- N=1000, balanced support: SUPPORTED 331, REFUTED 339, NEI 330.
- Accuracy `0.55600`; Macro-F1 `0.54069`.
- F1: SUPPORTED `0.64324`, REFUTED `0.60589`, NEI `0.37294`.
- Major errors: NEI→REFUTED 121, NEI→SUPPORTED 107.
- Policy remains frozen from train: baseline prediction `REFUTED` triggers `evidence_critic`.
## Step 52 — Dev selective-route result and confidence-router design

- Dev baseline: Accuracy `0.55600`, Macro-F1 `0.54069`.
- Dev route-all-REFUTED: Accuracy `0.55000`, Macro-F1 `0.54447`.
- NEI F1 improved `0.37294→0.43546`, but REFUTED F1 fell `0.60589→0.55743`.
- Conclusion: label-only route is too broad outside train.
- Added first-token generation confidence and confidence-threshold routing support.
## Step 53 — Freeze quality-max and cost-aware router variants

- Confidence sweep shows no threshold beats route-all (`1.0`) on train quality metrics.
- Quality-max: threshold `1.0`, routed 290/995, Accuracy `0.62613`, Macro-F1 `0.57279`.
- Cost-aware: threshold `0.90`, routed 66/995, Accuracy `0.61206`, Macro-F1 `0.55582`.
- Thresholds frozen on train before dev evaluation; confidence is not interpreted as calibrated correctness probability.
