# Báo cáo tiến độ ERRORFlow trên quy mô 1K

**Thời điểm tổng hợp:** 2026-10-11  
**Trạng thái:** đã hoàn thành prototype, train-side development trên 995 claim, dev audit trên 1.000 claim và frozen baseline trên toàn bộ 19.998 claim `shared_task_dev`. Full baseline đã qua integrity check; bước tiếp theo là chạy frozen cost-aware router. `paper_dev` vẫn chưa được mở cho đánh giá cuối.

## 1. Tóm tắt kết quả

ERRORFlow hiện đã chứng minh được tính khả thi của hướng **training-free, error-driven và cost-aware** trên FEVER với Qwen3-4B-Instruct-2507 đóng băng hoàn toàn.

Kết quả quan trọng nhất trên 1.000 claim `shared_task_dev` là:

| Hệ thống | Accuracy | Macro-F1 | Route rate | LLM calls/claim | Tokens/claim | Latency/claim |
|---|---:|---:|---:|---:|---:|---:|
| Frozen baseline | 0.55600 | 0.54069 | 0% | 1.00 | 31.885 | 765.19 ms |
| Route tất cả dự đoán REFUTED | 0.55000 | 0.54447 | không phải policy cuối | >1.00 | — | — |
| **Cost-aware ERRORFlow, threshold 0.90** | **0.56700** | **0.55757** | **5%** | **1.05** | **33.485** | **819.85 ms** |

So với frozen baseline, policy cost-aware đạt:

- Accuracy tăng **1.10 điểm phần trăm**.
- Macro-F1 tăng **1.69 điểm**.
- Chỉ route **50/1.000 claim** sang bước kiểm tra thứ hai.
- Số LLM call tăng 5%, generated tokens tăng khoảng 5.0%, latency tăng khoảng 7.1%.
- Trong 50 claim được can thiệp, có **23 trường hợp được sửa đúng** và **12 trường hợp bị sửa sai**, tương ứng lợi ròng **+11 prediction đúng**.

Paired bootstrap 5.000 lần cho thấy mức tăng Macro-F1 ổn định hơn mức tăng Accuracy:

- `Δ Accuracy = +0.01100`, 95% CI `[-0.00100, +0.02300]`, `P(Δ>0)=0.9642`.
- `Δ Macro-F1 = +0.01687`, 95% CI `[+0.00470, +0.02960]`, `P(Δ>0)=0.9968`.

Đây là bằng chứng development-scale tích cực, chưa phải kết quả cuối cùng của nghiên cứu.

## 2. Research question và method hiện tại

Ý tưởng trung tâm của ERRORFlow là:

> Biến các lỗi fact verification trong quá khứ thành tín hiệu có cấu trúc để lựa chọn workflow sửa lỗi cho claim mới, không cập nhật trọng số của LLM và không bắt mọi claim phải chạy workflow nhiều bước.

Prototype hiện tại triển khai hai vòng liên kết với nhau.

### 2.1. Offline error-learning loop

1. Chạy frozen verifier trên dữ liệu development.
2. So prediction với gold label để tạo error list.
3. Phân loại observable error transition và đề xuất intervention.
4. Chạy counterfactual interventions trên các baseline error.
5. Chạy cùng intervention trên baseline-correct control set để đo regression.
6. Lưu recovery, regression, prediction change và cost thành error/counterfactual memory.
7. Từ thống kê recovery–regression, hình thành routing hypothesis có thể quan sát tại inference time.

Hai intervention đã thử:

- `evidence_critic`: buộc model kiểm tra từng evidence sentence là support, contradict hay không đủ thiết lập claim.
- `conflict_check`: buộc model tìm cả evidence ủng hộ và phản bác trước khi chọn nhãn.

`evidence_critic` được giữ lại vì có lợi ròng; `conflict_check` bị loại ở phiên bản hiện tại vì recovery không bù được regression.

### 2.2. Online adaptive verification pipeline

```text
FEVER claim
   ↓
Claim-only deterministic retrieval
SQLite FTS5/BM25, OR query, top-5 evidence
   ↓
Frozen Qwen3 baseline verification
prediction + first-token confidence
   ↓
Cost-aware router
prediction == REFUTED AND confidence <= 0.90 ?
   ├─ No  → giữ baseline prediction, tổng cộng 1 LLM call
   └─ Yes → chạy evidence_critic, dùng verdict mới, tổng cộng 2 LLM calls
   ↓
Prediction artifact + provenance + token/call/latency accounting
```

Policy đang được khóa để xác nhận trên tập lớn là:

```text
IF baseline_prediction == REFUTED
AND first_generated_token_max_probability <= 0.90
THEN run evidence_critic
ELSE keep baseline prediction
```

Confidence này chỉ là **routing signal vận hành**, không được diễn giải là xác suất correctness đã calibration.

## 3. Protocol dữ liệu và chống leakage

### 3.1. Dataset và vai trò từng split

| Dữ liệu | Quy mô hiện dùng | Vai trò |
|---|---:|---|
| FEVER train subset | 995 claim | Phân tích lỗi, thử intervention, chọn và khóa router |
| FEVER `shared_task_dev` subset | 1.000 claim | Development audit độc lập của policy đã khóa |
| FEVER full `shared_task_dev` | 19.998 claim | Retrieval và baseline đã hoàn tất; chờ frozen-router evaluation |
| FEVER `paper_dev` | 9.999 claim | Final holdout, hiện vẫn giữ chưa dùng để chỉnh policy |

Train subset còn 995 thay vì 1.000 vì artifact ban đầu đã loại 5 positive claim không resolve được gold evidence. Đây là giới hạn của subset hiện tại; full-dev được chuẩn bị theo claim-only protocol nên không có bước loại tương tự.

### 3.2. Retrieval hợp lệ hiện tại

- Corpus: FEVER wiki-pages, index gồm **25.247.890 sentence**.
- Backend: SQLite FTS5 với BM25.
- Query: lấy tối đa 8 token đầu của claim, nối bằng `OR`.
- Evidence: top 5 sentence cho mọi claim, không phụ thuộc gold label.
- Determinism: sắp xếp theo `bm25(...), rowid` để phá BM25 tie ổn định.
- Parallelism: 4 worker, mỗi worker dùng SQLite read-only connection.
- Kiểm tra reproducibility: sequential và parallel deterministic giống nhau **995/995**.
- Tốc độ: 995 claim hoàn thành trong `2448.6s` (~40.8 phút), nhanh hơn khoảng 74.8% so với sequential OR ban đầu.
- Full-dev retrieval: `19.998/19.998` claim có evidence, hoàn thành trong `52,885.1s` (~14 giờ 41 phút), throughput trung bình khoảng `0.378 claim/s`; job chạy mới hoàn toàn (`resumed_from=0`).

Gold evidence của FEVER không được đưa vào prompt verifier trong protocol chính.

## 4. Những thành phần đã xây dựng

Đến hiện tại repository đã có:

- Data contracts cho prediction, execution trace và provenance.
- JSONL schema/persistence cho error memory.
- Frozen Qwen3 runner với greedy decoding và strict three-label parser.
- FEVER audit, label normalization, claim-only preparation và evidence utilities.
- FTS5 index builder và deterministic parallel retriever có progress/ETA/resume.
- Evaluator cho Accuracy, Macro-F1, per-class F1, confusion matrix, latency và tokens.
- Error-list builder, heuristic diagnosis và action analysis.
- Intervention runner cho `evidence_critic` và `conflict_check`.
- Recovery evaluator và baseline-correct regression control.
- Counterfactual memory builder.
- Selective route builder, route merge và cost accounting.
- Confidence-threshold sweep.
- Paired bootstrap comparison.
- Unit/regression tests cho core contracts, dataset, retrieval, diagnosis và verifier adapter.

Toàn bộ model, dataset, SQLite index và experiment outputs được giữ ngoài Git; code và tài liệu step-by-step được version control.

Kiểm tra local tại thời điểm lập báo cáo: **18 tests passed**, còn 2 `DeprecationWarning` liên quan `load_module()` trong test loader; không có test failure. Code snapshot trước khi thêm báo cáo là commit `9c963f9` (`Add resumable full dev retrieval`).

### 4.1. Artifact chính trên server

| Vai trò | Đường dẫn |
|---|---|
| Train deterministic retrieval | `data/processed/fever/train_first1000_retrieved_parallel4_deterministic.jsonl` |
| Train baseline có confidence | `outputs/qwen3_baseline/fever_train_first1000_confidence.jsonl` |
| Train intervention route | `outputs/qwen3_interventions/deterministic_refuted_route.jsonl` |
| Train threshold sweep | `outputs/qwen3_interventions/train_confidence_thresholds.json` |
| Dev 1K deterministic retrieval | `data/processed/fever/shared_task_dev_first1000_retrieved.jsonl` |
| Dev 1K baseline có confidence | `outputs/qwen3_baseline/shared_task_dev_first1000_confidence.jsonl` |
| Dev 1K cost-aware merged output | `outputs/qwen3_interventions/shared_task_dev_cost_full.jsonl` |
| Dev 1K paired bootstrap | `outputs/qwen3_interventions/shared_task_dev_cost_bootstrap.json` |
| Full dev deterministic retrieval | `data/processed/fever/shared_task_dev_all_retrieved.jsonl` |

## 5. Kết quả train-side canonical: 995 claim

### 5.1. Frozen deterministic baseline

| Metric | Kết quả |
|---|---:|
| Accuracy | 0.59698 |
| Macro-F1 | 0.53955 |
| F1 SUPPORTED | 0.74685 |
| F1 REFUTED | 0.46764 |
| F1 NOT ENOUGH INFO | 0.40417 |
| Baseline errors | 401 |
| Baseline correct | 594 |

Confusion matrix:

| Gold \ Pred | SUPPORTED | REFUTED | NOT ENOUGH INFO |
|---|---:|---:|---:|
| SUPPORTED | 385 | 72 | 84 |
| REFUTED | 43 | 112 | 34 |
| NOT ENOUGH INFO | 62 | 106 | 97 |

Lỗi nổi bật là xu hướng dự đoán `REFUTED` cho claim có gold `NOT ENOUGH INFO`.

### 5.2. Counterfactual intervention

Trên 401 baseline error, `evidence_critic`:

- Recovered `65/401`, tương đương **16.21%**.
- Thay đổi prediction ở 108 claim.
- Còn sai 336 claim.

Trên 594 baseline-correct control claim:

- Regressed `36/594`, tương đương **6.06%**.
- Nếu áp dụng đại trà cho toàn bộ 995 claim, lợi ròng lý thuyết là `65 - 36 = +29` prediction đúng.

Kết quả này cho thấy intervention có giá trị nhưng không đủ an toàn để áp dụng cho mọi claim.

### 5.3. Router variants được khóa từ train

| Variant | Điều kiện | Routed | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Baseline | Không intervention | 0/995 | 0.59698 | 0.53955 |
| Cost-aware | REFUTED và confidence ≤ 0.90 | 66/995 | 0.61206 | 0.55582 |
| Quality-max | REFUTED và confidence ≤ 1.00 | 290/995 | 0.62613 | 0.57279 |

Quality-max đạt điểm cao nhất trên train nhưng route quá rộng. Cost-aware chỉ dùng khoảng 22.8% số intervention call của quality-max và được chọn làm policy chính để kiểm tra khả năng generalize.

## 6. Kết quả dev audit: 1.000 claim

Tập dev gồm 331 `SUPPORTED`, 339 `REFUTED` và 330 `NOT ENOUGH INFO`, gần cân bằng giữa ba lớp.

### 6.1. Per-class result

| Hệ thống | F1 SUPPORTED | F1 REFUTED | F1 NEI | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.64324 | 0.60589 | 0.37294 | 0.55600 | 0.54069 |
| Route-all REFUTED | 0.64052 | 0.55743 | 0.43546 | 0.55000 | 0.54447 |
| **Cost-aware 0.90** | **0.64619** | **0.61353** | **0.41297** | **0.56700** | **0.55757** |

Route-all sửa được nhiều lỗi NEI nhưng làm hỏng quá nhiều prediction REFUTED đúng. Router confidence 0.90 khắc phục trade-off này: cả ba per-class F1 đều cao hơn baseline, trong khi chỉ can thiệp 5% claim.

### 6.2. Confusion matrix của cost-aware ERRORFlow

| Gold \ Pred | SUPPORTED | REFUTED | NOT ENOUGH INFO |
|---|---:|---:|---:|
| SUPPORTED | 242 | 23 | 66 |
| REFUTED | 66 | 204 | 69 |
| NOT ENOUGH INFO | 110 | 99 | 121 |

So với baseline:

- `NOT ENOUGH INFO → REFUTED` giảm từ 121 xuống 99.
- Số NEI đúng tăng từ 102 lên 121.
- REFUTED đúng giảm nhẹ từ 216 xuống 204, nhưng tổng thể vẫn có lợi ròng.

## 7. Full shared-task dev baseline: 19.998 claim

Full baseline artifact đã qua kiểm tra integrity:

- `19.998` rows và `19.998` unique claim IDs.
- Không có duplicate ID, missing prediction hoặc missing confidence.
- Prediction distribution: 8.095 `SUPPORTED`, 7.287 `REFUTED`, 4.616 `NOT ENOUGH INFO`.
- Confidence range: `0.34657–0.999999`.

| Metric | Full dev | Dev 1K | Chênh lệch full − 1K |
|---|---:|---:|---:|
| Accuracy | 0.56371 | 0.55600 | +0.00771 |
| Macro-F1 | 0.55153 | 0.54069 | +0.01083 |
| F1 SUPPORTED | 0.65158 | 0.64324 | +0.00834 |
| F1 REFUTED | 0.60360 | 0.60589 | -0.00229 |
| F1 NOT ENOUGH INFO | 0.39940 | 0.37294 | +0.02645 |

Full baseline có `11.273/19.998` prediction đúng và `8.725` prediction sai. Confusion matrix:

| Gold \ Pred | SUPPORTED | REFUTED | NOT ENOUGH INFO |
|---|---:|---:|---:|
| SUPPORTED | 4.809 | 743 | 1.114 |
| REFUTED | 1.206 | 4.211 | 1.249 |
| NOT ENOUGH INFO | 2.080 | 2.333 | 2.253 |

NEI tiếp tục là bottleneck: chỉ 2.253/6.666 claim NEI được dự đoán đúng; 2.333 bị dự đoán `REFUTED` và 2.080 bị dự đoán `SUPPORTED`. Kết quả full baseline gần audit 1K, cho thấy subset 1K không tạo ra sai lệch lớn về kết luận baseline.

Chi phí baseline: trung bình `788.06 ms`, `31.947` generated tokens và đúng `1.0` LLM call mỗi claim. Artifact metrics: `outputs/qwen3_baseline/shared_task_dev_all_metrics.json`.

## 8. Những run không được dùng làm kết quả chính

Một số kết quả trước đây chỉ có giá trị debug và đã được loại khỏi kết luận:

1. **Gold-evidence preparation:** SUPPORTS/REFUTES nhận annotated evidence trong khi NEI nhận evidence rỗng. Đây là protocol leakage/confound; kết quả Accuracy `0.71357`, Macro-F1 `0.62149` không được xem là research result.
2. **Old OR retrieval:** chưa có stable tie-break, nên parallel và sequential chỉ giống `667/995`. Artifact này không reproducible và đã bị thay thế.
3. **Fast `and_or` retrieval:** nhanh hơn khoảng 14.1% nhưng làm Accuracy giảm 2.52 điểm và Macro-F1 giảm 1.36 điểm so với old OR run tại thời điểm so sánh; không dùng cho pipeline chính.
4. **`conflict_check`:** recovered `20/376` nhưng regressed `23/619` trong protocol cũ, lợi ròng âm; intervention này bị loại.
5. **Route-all REFUTED trên dev:** Macro-F1 tăng nhẹ nhưng Accuracy và REFUTED F1 giảm; không được chọn làm policy cuối.

Việc lưu cả các kết quả thất bại này là quan trọng vì chúng giải thích tại sao pipeline hiện tại dùng deterministic OR retrieval và selective cost-aware routing.

## 9. ERRORFlow hiện đã đạt gì so với idea ban đầu?

Đã đạt:

- Frozen LLM fact verifier, không fine-tune và không dùng LoRA.
- Leakage-safe independent retrieval cùng một policy cho mọi label.
- Error analysis có recovery và regression control thay vì chỉ đếm lỗi.
- Counterfactual testing để biết intervention nào thực sự sửa được lỗi.
- Adaptive workflow chỉ gọi critic khi router cho rằng chi phí bổ sung đáng dùng.
- Cost accounting và uncertainty audit trên dev độc lập.
- Development result cho thấy adaptive workflow có thể tốt hơn fixed baseline với mức tăng chi phí nhỏ.

Chưa đạt đầy đủ:

- Router hiện là rule từ thống kê lỗi cộng first-token confidence; chưa truy hồi từng memory tương tự để thiết kế workflow riêng cho từng claim.
- Error diagnosis hiện chủ yếu là heuristic/transition-based, chưa có taxonomy nguyên nhân sâu và được validation.
- Full `shared_task_dev` đã có baseline nhưng chưa có frozen-router metrics; chưa có final `paper_dev`, AVeriTeC hoặc SciFact.
- Confidence chưa calibration.
- Retrieval vẫn lexical, lấy 8 token đầu và top-5 BM25; chưa đo retrieval recall so với gold evidence.
- Kết quả mới có một model và một seed/protocol inference deterministic.

Vì vậy, tên gọi chính xác cho phiên bản hiện tại là **training-free error-informed selective verification prototype**, chưa phải hệ thống error-memory optimizer hoàn chỉnh.

## 10. Kết luận hiện tại

Kết quả 1K ủng hộ ba nhận định:

1. Lỗi của frozen verifier có cấu trúc, đặc biệt ở ranh giới `REFUTED` và `NOT ENOUGH INFO`.
2. Một prompt intervention có thể sửa một phần lỗi nhưng sẽ gây regression nếu áp dụng đại trà.
3. Routing theo prediction và uncertainty giúp giữ phần lớn lợi ích trong khi chỉ tăng 5% số LLM call trên dev.

Điểm mạnh nhất hiện tại không phải chỉ là tăng Macro-F1, mà là research story đã khép kín ở mức prototype:

```text
phát hiện lỗi → chẩn đoán → thử sửa counterfactual → đo recovery/regression
→ khóa router theo cost → kiểm tra trên claim mới
```

## 11. Bước tiếp theo đã khóa

1. Tạo full-dev route subset bằng đúng policy đã khóa: baseline prediction `REFUTED` và confidence `≤ 0.90`.
2. Chạy `evidence_critic`, merge với full baseline và đánh giá mà không chỉnh lại threshold.
3. Báo cáo quality, paired uncertainty, route rate, calls, tokens và latency trên full dev.
4. Chỉ sau khi code, policy và reporting protocol được khóa mới mở `paper_dev` một lần cho final evaluation.
5. Nếu full-dev xác nhận xu hướng, bước nghiên cứu tiếp theo là memory-conditioned routing thực sự và cross-domain evaluation trên AVeriTeC/SciFact.

Server root filesystem vẫn ở mức sử dụng cao (`99%`) nhưng tại lần kiểm tra gần nhất còn khoảng `13 GB`. Dung lượng này đủ cho prediction/log JSONL hiện tại, nhưng vẫn cần theo dõi và không tạo thêm model/index lớn hoặc xóa dữ liệu ngoài phạm vi dự án khi chưa kiểm tra.

**Trạng thái route hiện tại:** frozen policy đã chọn `1.181/19.998` claim, tương đương route rate `5,91%`. Critic run đầu bị mất artifact do output/log bị truncate trong một lần chạy lại; không có bằng chứng OOM hay thiếu tài nguyên. Runner đã được bổ sung durable per-record flush và `--resume`, đang chờ commit/pull để chạy lại an toàn.
