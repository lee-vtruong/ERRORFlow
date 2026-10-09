# Step 13 — Conservative error diagnosis

## Kết quả quan sát

Trong 376 lỗi independent-retrieval:

- NEI → REFUTED: 94
- NEI → SUPPORTED: 76
- SUPPORTED → NEI: 68
- SUPPORTED → REFUTED: 55
- REFUTED → SUPPORTED: 46
- REFUTED → NEI: 37
- Mọi lỗi đều có đúng 5 retrieved evidence.

## Đã làm

`diagnose_errors.py` gán `diagnosis_basis` và `candidate_actions` dựa trên label transition. Mỗi record được đánh dấu `diagnosis_status=heuristic_candidate`.

Đây chưa phải nguyên nhân thật và không dùng gold label để chọn policy final. Cần intervention evaluation để biết action nào thực sự sửa được lỗi.
