# Results

ผลการทดลองจะบันทึกไว้ที่นี่ ไฟล์ทั้งหมดถูก git ignore ยกเว้นไฟล์นี้ ถ้าต้องการเก็บตารางสรุปผลสุดท้ายไว้ใน repo ให้ `git add -f` เฉพาะไฟล์นั้น

## `results.csv` (หนึ่งแถวต่อหนึ่งรัน, เขียนโดย `src/train.py`)

| คอลัมน์ | ความหมาย |
|---|---|
| `dataset`, `fold`, `method`, `role`, `seed` | ระบุการรัน (`fold` = official หรือ 0–4) |
| `lr`, `batch_size`, `epochs_run`, `best_epoch` | การฝึกจริง |
| `params_M`, `trainable_M`, `train_time_min`, `infer_ms_per_sample` | ต้นทุนการคำนวณ (ใช้ประกอบความเป็นธรรม) |
| `val_icbhi_score` | ICBHI score ที่ดีที่สุดบน validation (ใช้เลือก epoch) |
| `icbhi_score`, `macro_f1` | metric หลักบน test |
| `sensitivity`, `specificity`, `balanced_acc`, `accuracy` | metric รอง |
| `f1_normal`, `f1_crackle`, `f1_wheeze`, `f1_both` | F1 รายคลาส (ตอบ RQ2) |
| `gate_mel`, `gate_aux` | ค่าเฉลี่ย SE gate ของแต่ละทาง (เฉพาะโมเดลที่มี fusion) |
| `mel_scale`, `bandpass`, `limit` | ตัวเลือก preprocessing, `limit > 0` = รัน debug (stats จะข้าม) |

- `tune.csv`: ผลของ `--plan tune` (มีเฉพาะ validation ไม่เปิด test)
- `summary_<metric>.csv`: mean, SD, 95% CI ต่อ dataset × method
- `tests_<metric>.csv`: paired t-test / Wilcoxon + Holm เทียบ `proposed` กับทุกวิธี
- `runs/<dataset>/<method>/fold<k>_seed<s>/`: `history.csv` (loss และ metric บน val ทุก epoch), `confusion.csv`, `test_predictions.csv`, `metrics.json`
