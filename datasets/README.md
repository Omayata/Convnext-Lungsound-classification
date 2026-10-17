# Datasets

ไฟล์ข้อมูลทั้งหมดในโฟลเดอร์นี้ถูก git ignore ไว้ เพราะใหญ่เกินไปสำหรับ GitHub และ license ไม่อนุญาตให้แจกต่อ ให้ดาวน์โหลดเองแล้ววางตามโครงสร้างนี้

## ICBHI 2017 → `datasets/ICBHI_2017/`

- ทางการ: https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip
- หรือ Kaggle: "Respiratory Sound Database"
- วางไฟล์ `.wav` และ `.txt` ไว้ระดับเดียวกันตรงๆ ไม่ต้องมีโฟลเดอร์ย่อย:
  ```
  datasets/ICBHI_2017/101_1b1_Al_sc_Meditron.wav
  datasets/ICBHI_2017/101_1b1_Al_sc_Meditron.txt
  ...
  ```
- ชุดที่ถูกต้องมี wav 920 ไฟล์ (ราว 2.1 GB) ชื่อไฟล์ต้องตรงกับ `third_party/2_lung_ICBHI/patch-mix_contrastive_learning/data/icbhi_dataset/official_split.txt` ถ้าไม่ตรง ให้รัน `script.sh` ในโฟลเดอร์ `icbhi_dataset` นั้น
- ไฟล์ split ทางการและข้อมูลวินิจฉัยผู้ป่วยอยู่ใน `third_party/2_lung_ICBHI/patch-mix_contrastive_learning/data/icbhi_dataset/`

## ESC-50 → `datasets/ESC-50/`

```bash
git clone https://github.com/karolpiczak/ESC-50.git datasets/ESC-50
```
ได้ไฟล์เสียง 2,000 ไฟล์ใน `audio/` และ `meta/esc50.csv` (ราว 850 MB) ใช้ fold ทางการ 5 fold (คอลัมน์ `fold`)

## (ถ้ามีเวลา)

- **PhysioNet/CinC 2016** (เสียงหัวใจ): https://physionet.org/content/challenge-2016/1.0.0/ ต้องแบ่งตามผู้ป่วย/database เอง
- **COUGHVID** (เสียงไอ): https://zenodo.org/record/4048312 ใช้ `third_party/4_cough/detect-segment-cough` ตัดเสียงไอก่อนนำไปฝึก
