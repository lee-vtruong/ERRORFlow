# ERRORFlow — Quy tắc làm việc với Codex và server

## Mục tiêu

ERRORFlow là dự án nghiên cứu `Error-Conditioned Memory and Adaptive Workflow Optimization for Training-Free Fact Verification`. Codex hỗ trợ code, kiểm thử, ghi nhật ký và phân tích kết quả; người dùng quyết định thời điểm push/pull, chạy job trên server và cung cấp log/output để phân tích.

## Vòng lặp chuẩn

```text
Người dùng giao việc → Codex sửa/kiểm tra local → Codex ghi step + kết quả vào Markdown
→ người dùng review/commit/push GitHub → người dùng pull trên server
→ chạy training/inference/evaluation → gửi command, log, output cho Codex
→ Codex phân tích insight và ghi tiếp vào Markdown
```

Codex không tự nhìn thấy tiến trình server. Khi cần hỗ trợ, gửi đầy đủ commit hash, lệnh đã chạy, lỗi/log cuối và đường dẫn output. Không giả định server đã có model hoặc môi trường nếu chưa có bằng chứng từ lệnh kiểm tra.

## Cấu hình dự án

### Local

- Repository: `ERRORFlow`
- Workspace: `C:\Users\ASUS\Downloads\ERRORFlow`
- GitHub: `https://github.com/lee-vtruong/ERRORFlow`
- Git remote chuẩn: `origin` trỏ tới URL GitHub trên (kiểm tra trước lần push đầu tiên).
- Mỗi thay đổi phải được ghi vào `docs/WORK_LOG.md` hoặc file Markdown chuyên môn tương ứng.

### Server

- Host/user: `stackops@hvtham-server`
- Repository: `~/whale/ERRORFlow`
- Virtual environment ưu tiên: `~/whale/ERRORFlow/.venv`
- Server hiện có môi trường cũ tại `~/whale/GraphCURE/.venv`. Có thể kiểm tra và tái sử dụng cho ERRORFlow nếu dependency tương thích; không sửa/xóa môi trường đó trước khi có quyết định rõ ràng. Môi trường đích ưu tiên vẫn là `~/whale/ERRORFlow/.venv` nếu cần cô lập dependency.
- GPU: kiểm tra thực tế bằng `nvidia-smi`; không kế thừa thông tin benchmark cũ của GraphCURE.

## Nhật ký bắt buộc

Mỗi step làm việc phải ghi: thời gian/mục tiêu; lệnh hoặc file đã kiểm tra/sửa; kết quả kể cả lỗi; quyết định/giả định/việc còn thiếu; commit hash nếu có.

Kết quả thực nghiệm phải kèm model/checkpoint, seed, dataset split, manifest/retrieval hash, prediction hash, protocol, chi phí token/call/latency và các cờ `official_validation_used`, `test_split_used`. Không dùng gold label validation/test để viết rule, chọn exemplar, chọn intervention hoặc chỉnh policy cuối.

## Local: kiểm tra và commit

```powershell
cd C:\Users\ASUS\Downloads\ERRORFlow
git status --short --branch
git remote -v
git diff --stat
python -m compileall src scripts tests
git add <các-file-thay-đổi-cụ-thể>
git commit -m "Describe the change"
git push origin <branch>
git rev-parse --short HEAD
```

Không dùng `git add .` khi có output, checkpoint, dataset hoặc file tạm. Dùng `.gitignore` cho artifact lớn/nhạy cảm.

## Server: pull và môi trường

```bash
ssh stackops@hvtham-server
cd ~/whale/ERRORFlow
git pull --ff-only origin main
python -m venv .venv  # chỉ chạy nếu chưa có môi trường phù hợp
source .venv/bin/activate
python -m pip install -e .
git rev-parse --short HEAD
nvidia-smi
```

Trước khi tải model mới, kiểm tra cache và model hiện có, ví dụ `find ~/.cache/huggingface -maxdepth 3 -type d`, `ollama list`, `which vllm`; ghi kết quả vào log. Không xóa hoặc tải lại model nếu chưa xác định nhu cầu và dung lượng.

## Chạy job dài

```bash
tmux new -s errorflow-run
cd ~/whale/ERRORFlow
source .venv/bin/activate
mkdir -p outputs/logs
bash scripts/run_job.sh 2>&1 | tee outputs/logs/<job>.log
```

Tách tmux: `Ctrl-b`, rồi `d`. Kết nối lại: `tmux ls`, `tmux attach -t errorflow-run`.

## Theo dõi và gửi kết quả

```bash
tmux ls
nvidia-smi
ps -ef | grep -E "python|errorflow|train|infer" | grep -v grep
cat outputs/<experiment>/summary.json
cat outputs/<experiment>/report.md
tail -n 100 outputs/logs/<job>.log
```

Mẫu gửi: `Commit`, `Lệnh`, `Mục tiêu`, `Output`, `Model/environment`, `Log cuối`.

## Checklist trước khi kết luận

- [ ] Đúng commit, model, checkpoint, seed và môi trường.
- [ ] Tách đúng memory construction / optimizer development / final evaluation.
- [ ] Không dùng test labels để chọn policy.
- [ ] Có baseline fixed workflow và adaptive workflow cùng protocol.
- [ ] Báo cáo Macro-F1/Accuracy, Recovery Rate, Regression Rate, token cost, số call và latency.
- [ ] Có summary, report, log và các hash provenance.
- [ ] Ghi rõ failure, limitation và insight; không biến point estimate thành kết luận nhân quả khi chưa có kiểm định phù hợp.
