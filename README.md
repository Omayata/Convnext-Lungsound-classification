# Lung Sound Classification: Mel–MFCC Fusion + AudioSet-Pretrained ConvNeXt

โค้ดทดลองของหัวข้อ **"Enhanced Lung Sound Classification via Mel–MFCC Feature Fusion and AudioSet-Pretrained ConvNeXt under Patient-Independent Evaluation"** ตรงกับเอกสาร `RGQD/RD_v2.docx`

> **สถานะ (2026-10-21):** pipeline ครบทุกส่วนตาม RD_v2 ทดสอบกับข้อมูลจำลองแล้ว ยังไม่ได้รันกับข้อมูลจริง
> **คนที่จะรัน demo ให้อ่าน [`docs/HANDOFF_DEMO.md`](docs/HANDOFF_DEMO.md) ก่อน**
> แนวทางเดิม (benchmark ข้ามประเภทเสียง + ESC-50) ใน `docs/HANDOFF.md` ถูกแทนที่ด้วยแนวทางนี้แล้ว
> รายละเอียดว่า RD แต่ละข้ออยู่ตรงไหนในโค้ด และจุดที่ต่างจากต้นฉบับของคู่เทียบ อยู่ใน [`docs/EXPERIMENT_v2.md`](docs/EXPERIMENT_v2.md)

## สรุปการทดลอง

| หัวข้อ | ค่าที่ใช้ |
|---|---|
| งาน | จำแนก 4 คลาส: normal / crackle / wheeze / both |
| Dataset | ICBHI 2017 (ระดับรอบหายใจ, official split 60/40), KAUH และ HF_Lung_V1 (stratified group 5-fold ตามผู้ป่วย/การบันทึก) |
| Validation | ผู้ป่วย 20% ของชุด train (split คงที่ ไม่ขึ้นกับ seed) ใช้ทำ early stopping |
| Preprocessing | resample 16 kHz, band-pass 120–1800 Hz, ICBHI ตัดตามรอบหายใจแล้วเติมแบบวนซ้ำเป็น 8 วินาที |
| โมเดลที่เสนอ | ConvNeXt-Tiny (AudioSet) รับ log-Mel 50–2000 Hz + ทางเสริม CNN 3 block รับ MFCC + ΔMFCC แล้วรวมแบบ late fusion ผ่าน SE gating |
| Ablation | ตัด/เปลี่ยนทีละเทคนิค 11 แบบ: feature (Mel อย่างเดียว, ไม่มี Δ), pretrain (ImageNet, สุ่ม), SE gating, band-pass, PCEN, SpecAugment, mixup, class weight, drop path |
| คู่เทียบ (ทำซ้ำ) | Wanasinghe, Dongre, Shakya, Puneet, Park, Engin |
| การฝึก | weighted CE, SpecAugment + mixup เฉพาะชุด train, early stopping ตาม ICBHI score บน val, AMP |
| Metric | ICBHI score, macro-F1 (หลัก), Se, Sp, F1 รายคลาส, confusion matrix, จำนวนพารามิเตอร์, เวลา |
| การรันซ้ำ | 5 seeds (0–4), PoC อย่างน้อย 3 |
| สถิติ | paired t-test ทางเดียว + Holm (หลัก), Wilcoxon เมื่อ seed ≥ 6, Friedman + Nemenyi ข้าม dataset |

## วิธีที่มี (`--method`)

| ชื่อ | บทบาท | รายละเอียด |
|---|---|---|
| `proposed` | Proposed | ConvNeXt-Tiny (AudioSet) + Mel + MFCC + ΔMFCC (SE fusion) |
| `abl_mel_only` | Ablation (RQ2) | ConvNeXt-Tiny (AudioSet) + Mel อย่างเดียว |
| `abl_mel_mfcc` | Ablation (RQ2) | ConvNeXt-Tiny (AudioSet) + Mel + MFCC (ไม่มี Δ) |
| `abl_imagenet` | Ablation | ConvNeXt-Tiny (ImageNet) + Mel + MFCC + ΔMFCC |
| `abl_no_pretrain` | Ablation | ConvNeXt เริ่มจากน้ำหนักสุ่ม (ไม่ pretrain) |
| `abl_concat` | Ablation | รวม embedding ด้วยการต่อกันตรงๆ แทน SE gating |
| `abl_no_bandpass` | Ablation | ไม่ทำ band-pass 120–1800 Hz |
| `abl_pcen` | Ablation | PCEN แทน log-Mel |
| `abl_no_specaug` | Ablation | ไม่ใช้ SpecAugment |
| `abl_no_mixup` | Ablation | ไม่ใช้ mixup |
| `abl_no_classweight` | Ablation | ไม่ถ่วงน้ำหนักคลาสใน loss |
| `abl_no_droppath` | Ablation | ไม่ใช้ drop path |
| `wanasinghe_cnn` | คู่เทียบ 1 | CNN เล็ก + Mel/MFCC/Chroma ซ้อน 3 channel |
| `dongre_cnn_bigru` | คู่เทียบ 2 | CNN-BiGRU บน MFCC + SMOTE เฉพาะชุด train |
| `shakya_vgg16` | คู่เทียบ 3 | VGG16 (ImageNet) + Mel 224×224, fine-tune block5 |
| `puneet_mobilenetv2` | คู่เทียบ 4 | MobileNetV2 (ImageNet) + STFT spectrogram |
| `park_inception` | คู่เทียบ 5 | Inception lightweight CNN + Mel/Chroma/MFCC |
| `engin_densenet201` | คู่เทียบ 6 | DenseNet201 (ImageNet) + Gammatonegram |

## ติดตั้ง (Windows + RTX 4070 SUPER)

```bash
conda create -n lung python=3.10 -y
conda activate lung
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"     # ต้องได้ True
```

จากนั้นวางข้อมูลและ checkpoint ตาม [`datasets/README.md`](datasets/README.md)

## Demo สำหรับนำเสนอ (ไม่ต้องรันเต็ม)

ดับเบิลคลิก `scripts\run_demo.bat` (หรือรัน 3 คำสั่งด้านล่าง) ใช้ **ICBHI จริง และ hyperparameter จริงของทุกวิธี** ต่างจากของจริงแค่ฝึก 5 epoch และ seed เดียว
รัน 18 วิธี: โมเดลที่เสนอ + คู่เทียบ 6 งาน + ablation 11 แบบ (ตัดทีละเทคนิค) ใช้เวลาราว 2–4 ชั่วโมงบน RTX 4070 SUPER (ประมาณ ยังไม่ได้วัด)
ถ้าอยากเร็วขึ้นใช้ `--epochs 3` สั่งซ้ำได้ จะข้ามวิธีที่เสร็จแล้ว ได้ไฟล์ `results/demo/demo_report.html` เปิดใน browser ได้เลย

ในรายงาน แต่ละวิธีมีสีของตัวเองและใช้สีเดียวกันทุกกราฟ ส่วน ablation แสดงเป็นกราฟ "คะแนนลบด้วยโมเดลที่เสนอ"
แท่งแดง (ติดลบ) คือตัดเทคนิคนั้นออกแล้วแย่ลง แปลว่าเทคนิคนั้นช่วยจริง

**ต้องมีก่อนรัน:** ข้อมูล ICBHI ใน `datasets/ICBHI_2017/` และ `checkpoints/convnext_tiny_471mAP.pth` (ดู [`datasets/README.md`](datasets/README.md))

```bash
python -m src.prepare_data --dataset icbhi
python -m src.run_experiments --plan demo            # เปลี่ยนจำนวน epoch ได้ เช่น --epochs 3
python scripts/demo_report.py
```

ผล demo เก็บแยกใน `results/demo/` ไม่ปนกับผลจริง และเป็นผลที่ฝึกไม่ครบ จึงใช้แสดงว่า pipeline ทำงานบนข้อมูลจริงเท่านั้น ห้ามใช้สรุปว่าวิธีไหนดีกว่า

## ขั้นตอนรัน

คำสั่งทั้งหมดรันจากโฟลเดอร์นี้ (โฟลเดอร์ที่มี `src/`)

```bash
# 0) (ไม่บังคับ) ทดสอบ pipeline ด้วยข้อมูลจำลองก่อนมีข้อมูลจริง
python scripts/make_dummy_data.py
python -m src.train --dataset icbhi --data_root datasets_dummy/ICBHI_2017 --method proposed --epochs 1 --audioset_ckpt "" --no_pretrained

# 1) เตรียม cache และตรวจข้อมูล (ICBHI ต้องได้ train 4142 / test 2756 รอบ)
python -m src.prepare_data --dataset icbhi

# 2) จับเวลา 1 รันก่อนวางแผน
python -m src.train --dataset icbhi --method proposed --seed 0 --epochs 2

# 3) Proof of Concept: 6 วิธี × 3 seeds บน ICBHI (สั่งซ้ำได้ จะข้ามรันที่เสร็จแล้ว)
python -m src.run_experiments --plan poc

# 4) สรุปผลและทดสอบสถิติ
python -m src.stats
python -m src.stats --metric macro_f1
```

หรือดับเบิลคลิก `scripts\run_poc.bat` เพื่อรันข้อ 1, 3, 4 ต่อกัน

**แผนฉบับเต็ม:**

```bash
python -m src.run_experiments --plan tune             # (ถ้าต้องการ) ปรับ lr ทุกวิธีด้วยงบเท่ากันบน validation
python -m src.run_experiments --plan main --use_tuned # ICBHI ทุกวิธี (18 วิธี รวม ablation) × 5 seeds
python -m src.prepare_data --dataset kauh
python -m src.prepare_data --dataset hf_lung
python -m src.run_experiments --plan cross            # KAUH + HF_Lung × 5 fold × 5 seeds (RQ3)
python -m src.stats                                   # จะมี Friedman + Nemenyi เมื่อมีผลครบหลาย dataset
```

**ตัวเลือกที่ใช้บ่อย:** `--epochs 50 --patience 10`, `--batch_size 16`, `--micro_batch 8` (สะสม gradient สำหรับ GPU เล็ก effective batch เท่าเดิม และเปิดอัตโนมัติเมื่อ VRAM < 8 GB), `--mel_scale pcen` (ablation PCEN), `--no_bandpass`, `--fusion concat`, `--tag _pcen` (ต่อท้ายชื่อในผล), `--save_ckpt`

## ผลลัพธ์

- `results/results.csv`: หนึ่งแถวต่อหนึ่งรัน (ดูคอลัมน์ใน [`results/README.md`](results/README.md))
- `results/runs/<dataset>/<method>/fold<k>_seed<s>/`: `history.csv`, `confusion.csv`, `test_predictions.csv`, `metrics.json`
- `results/summary_<metric>.csv`, `results/tests_<metric>.csv`: จาก `src.stats`

## โครงสร้าง

```
src/
  datasets.py          ICBHI / KAUH / HF_Lung loader, cache, การแบ่งข้อมูลแยกผู้ป่วย
  features.py          log-Mel, PCEN, MFCC, ΔMFCC, Chroma, STFT, Gammatonegram (คำนวณบน GPU + cache)
  convnext_audioset.py ConvNeXt-Tiny AudioSet (โหลด checkpoint ของ Pellegrini et al.)
  models.py            โมเดลที่เสนอ + คู่เทียบ 6 งาน
  methods.py           ทะเบียนวิธี: feature, โมเดล, hyperparameter ของแต่ละวิธี
  metrics.py           ICBHI score, macro-F1, F1 รายคลาส
  train.py             ฝึก + ประเมิน 1 รัน
  run_experiments.py   รันตามแผน poc / main / cross / tune แบบสั่งต่อได้
  stats.py             mean ± SD, 95% CI, paired t-test + Holm, Wilcoxon, Friedman + Nemenyi
  prepare_data.py      สร้าง cache และตรวจจำนวนข้อมูล
splits/                official split ของ ICBHI และ fold ที่กำหนดครั้งเดียว (ใช้ร่วมทุกวิธี)
scripts/               make_dummy_data.py, run_poc.bat
third_party/           โค้ดอ้างอิงเดิม (ไม่ได้ import ใช้ตรงๆ)
```

## ข้อควรระวัง

- **official split ของ ICBHI มีผู้ป่วย 2 คน (156, 218) อยู่ทั้ง train และ test** เพราะแบ่งระดับไฟล์ โค้ดคงไว้ตามต้นฉบับเพื่อให้เทียบกับงานอื่นได้ ควรเขียนเป็นข้อจำกัดในรายงาน
- **HF_Lung_V1 ไม่มีรหัสผู้ป่วย** โค้ดจัดกลุ่มตาม อุปกรณ์ + วันที่ + ชั่วโมง ในชื่อไฟล์ ให้เช็กชื่อไฟล์จริงก่อนใช้ (`src/datasets.py: hf_group`)
- **KAUH** อ่าน label จากชื่อไฟล์ ให้รัน `prepare_data` แล้วดูรายการไฟล์ที่ถูกข้าม ถ้าชื่อไฟล์ต่างจากที่คาด ให้แก้ `parse_kauh_sound`
- `--no_pretrained` และ `--audioset_ckpt ""` ใช้ทดสอบ pipeline เท่านั้น ห้ามใช้รายงานผล
- บน Windows ไม่ต้องตั้ง `num_workers` เพราะ feature ถูกโหลดไว้ในหน่วยความจำแล้ว
