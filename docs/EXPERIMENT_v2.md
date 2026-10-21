# การทดลองตาม RD_v2: ตรงไหนอยู่ในโค้ดตรงไหน

อัปเดต: 2026-10-18 · ใช้คู่กับ `RGQD/RD_v2.docx` และ `HANDOFF_2.md` (ถ้าขัดกับ `docs/HANDOFF.md` ให้ยึดไฟล์นี้)

## 1. RD → โค้ด

| RD_v2 | สิ่งที่กำหนด | โค้ด |
|---|---|---|
| 1.4 IV1 วิธีการจำแนก | โมเดลที่เสนอ vs คู่เทียบ 6 งาน | `src/methods.py: METHODS` |
| 1.4 IV2 ชุด feature | Mel / Mel+MFCC / Mel+MFCC+Δ | `proposed`, `abl_mel_mfcc`, `abl_mel_only` |
| 1.4 IV3 น้ำหนักเริ่มต้น | AudioSet / ImageNet | `proposed` vs `abl_imagenet` |
| 1.4 IV4 ชุดข้อมูล | ICBHI / KAUH / HF_Lung_V1 | `--dataset`, `src/datasets.py` |
| 2.1 Target | 4 คลาสระดับรอบหายใจ (ICBHI), ระดับไฟล์ (KAUH, HF_Lung) | `_lab()`, `build_*()` |
| 2.2 Split | ICBHI official 60/40, อีก 2 ชุด stratified group 5-fold, val 20% ของผู้ป่วยใน train | `get_split()`, `kfold_assignments()` (บันทึก fold ไว้ใน `splits/`) |
| 2.3 Leakage | แยกผู้ป่วย, augment/SMOTE หลัง split เฉพาะ train, KAUH ฟิลเตอร์เดียว, เลือกโมเดลจาก val, normalization จาก train | `get_split()` (assert), `smote()`, `--kauh_filter`, early stopping บน val, `channel_stats()` |
| 3.1 Preprocessing | resample, band-pass 120–1800 Hz, ปรับความยาวรอบหายใจ | `load_dataset()`: 16 kHz, Butterworth อันดับ 5 แบบ zero-phase, เติมแบบวนซ้ำเป็น 8 วินาที |
| 3.1 ทางหลัก | log-Mel 50–2000 Hz เข้า ConvNeXt-Tiny AudioSet ช่องเดียว | `features.py: MEL`, `ProposedNet` (ปรับแกนความถี่เป็น 224 bin ตาม checkpoint) |
| 3.1 ทางเสริม | MFCC + ΔMFCC 2 channel, CNN 3 block | `AuxCNN` (MFCC 40 ค่า, Δ กว้าง 9 เฟรม) |
| 3.1 Fusion | ต่อ embedding + SE gating + ชั้นจำแนก 4 คลาส | `SEGate`, `ProposedNet.head`; ค่า gate บันทึกใน `gate_mel`, `gate_aux` |
| 3.1 การฝึก | AdamW, cosine, AMP, SpecAugment + mixup เฉพาะ train, drop path, early stopping ตาม ICBHI score, weighted CE | `train.py` (AdamW lr 1e-4 wd 0.05, warmup 2 epoch, drop path 0.1, mixup α 0.2, patience 10) |
| 3.2 คู่เทียบ 1–6 | ทำซ้ำตามต้นฉบับบน split เดียวกัน | `models.py`, `methods.py` (ดูส่วนที่ 2) |
| 3.2 PoC | วิธี 1–4 และ 7 + โมเดลที่เสนอบน ICBHI | `--plan poc` |
| 4 Fair comparison | split, preprocessing, STFT, epoch, early stopping, augmentation, loss เดียวกัน; งบปรับจูนเท่ากัน; รายงานพารามิเตอร์และเวลา | ค่ากลางใน `train.py`, `STFT` ใน `features.py`, `--plan tune`, คอลัมน์ `params_M`, `train_time_min`, `infer_ms_per_sample` |
| 5 Metrics | ICBHI score, macro-F1, Se, Sp, F1 รายคลาส, confusion matrix | `metrics.py`, `confusion.csv` |
| 6 Repetition | 5 seeds (0–4), PoC 3; mean, SD, 95% CI | `PLANS`, `stats.py: summary()` |
| 7 Statistics | paired t-test (หลัก), Wilcoxon (seed ≥ 6), Holm, Friedman + Nemenyi | `stats.py` |

## 2. การทำซ้ำคู่เทียบ: ตามต้นฉบับตรงไหน ต่างตรงไหน

ทุกวิธีใช้ preprocessing, split, epoch สูงสุด 50, early stopping (patience 10), SpecAugment + mixup และ weighted CE ชุดเดียวกัน (RD ส่วนที่ 4) ส่วนที่ต่างกันคือ feature, โครงโมเดล และ optimizer ตามต้นฉบับ

| วิธี | ตามต้นฉบับ | ต่างจากต้นฉบับ / ต้นฉบับไม่ระบุ |
|---|---|---|
| Wanasinghe 2024 | Mel/MFCC/Chroma ซ้อน 3 ch ขนาด 128×264, conv 3×3 same + ReLU, maxpool 2×2, BN, dropout, GAP, dense, Adam 1e-4, class weight | จำนวนชั้นและ filter ไม่ได้เขียนในเนื้อความ (อยู่ใน Fig. 4) ใช้ 4 block 32–256 ได้ราว 0.46M parameter (ต้นฉบับ 0.51M) **ควรเทียบกับ Fig. 4 แล้วปรับ** |
| Dongre 2025 | MFCC → conv + maxpool → BiGRU → dense + dropout, Adam, SMOTE | ทำ SMOTE **เฉพาะชุด train หลัง split** (ต้นฉบับทำทั้ง dataset); จำนวน MFCC, ชั้น, hidden size ต้นฉบับไม่ระบุ ใช้ 40 MFCC, conv 3 ชั้น, BiGRU 128 |
| Shakya 2026 | VGG16 ImageNet, Mel 224×224, fine-tune เฉพาะ block สุดท้าย, BN, dropout 0.5, Adam lr 1e-6 | head ใหม่ Dense 256 บน GAP (แบบ include_top=False); ไม่ใช้ augmentation แบบภาพ (หมุน/พลิก) เพราะไม่สมเหตุสมผลกับ spectrogram และใช้ augmentation กลางแทน; **lr 1e-6 อาจต่ำเกินสำหรับ 50 epoch ควรรัน `--plan tune`** |
| Puneet 2025 | MobileNetV2 ImageNet, STFT 1024 จุด, Hann 128, hop 64 | ไม่รวม EVMD และส่วน FPGA; lr ต้นฉบับไม่ระบุ ใช้ Adam 1e-4 |
| Park 2025 | Mel/Chroma/MFCC ซ้อน (128×216), MFCC 40, Inception block แบบ asymmetric (1×3/3×1, 1×5/5×1), 3 stage, conv 3×3 + adaptive max pool + FC, lr 1e-4, batch 32 | sample rate 16 kHz และความยาว 8 วินาทีตาม preprocessing กลาง (ต้นฉบับ 22.05 kHz, 5 วินาที); จำนวน channel ต่อ stage ประมาณเอง |
| Engin 2026 | DenseNet201 ImageNet, Gammatonegram 64 ช่อง 25 ms / 10 ms เริ่ม 50 Hz, ภาพ 256×256 3 ch, Adam 1e-3, batch 16 | ใช้ annotation รอบหายใจของ ICBHI แทนระบบตัดรอบอัตโนมัติ; gammatone คำนวณแบบประมาณบน FFT ไม่ใช่ filter bank เชิงเวลา |

ทุกข้อในคอลัมน์ขวาควรเขียนไว้ในรายงาน (RD ส่วนที่ 4: "บันทึกทุกจุดที่การทำซ้ำต่างจากต้นฉบับ")

## 3. สิ่งที่ตัดสินใจเองระหว่างเขียนโค้ด (เปลี่ยนได้)

- **STFT กลาง:** n_fft 2048, window 400 (25 ms), hop 160 (10 ms) ที่ 16 kHz, Mel 128 bin 50–2000 Hz ใช้ n_fft ใหญ่เพื่อไม่ให้ filter ช่วงความถี่ต่ำว่าง
- **ความยาวคลิป:** ICBHI 8 วินาที (ตาม Patch-Mix), KAUH และ HF_Lung 15 วินาที
- **ConvNeXt AudioSet:** checkpoint ฝึกด้วย Mel 224 bin ช่วง 50–14000 Hz ที่ 32 kHz เราใช้ Mel ช่วงเสียงปอดแล้ว interpolate แกนความถี่เป็น 224 bin เพื่อใช้ชั้นแรกและ bn0 เดิมได้ ช่วงความถี่ที่แต่ละ bin หมายถึงจึงต่างจากตอน pretrain ซึ่งการ fine-tune ต้องปรับให้
- **batch:** ConvNeXt ใช้ 16 (ราว 4–5 GB VRAM ต่อ 16 ตัวอย่างเมื่อเปิด AMP ประมาณจาก CPU ยังไม่ได้วัดบน GPU) ถ้า VRAM เหลือลอง `--batch_size 32`
- **validation:** ใช้ผู้ป่วย 20% ของ train แบบ stratified group และ split คงที่ (seed 0) ทุกการรัน
- **Ablation ครบทุกเทคนิค (เพิ่มจาก RD_v2 ที่มีแค่ 3 แบบ):** แต่ละ `abl_*` ใน `src/methods.py` คือโมเดลที่เสนอที่เปลี่ยนไปอย่างเดียวผ่าน `overrides`
  (ไม่มี Δ, Mel อย่างเดียว, ImageNet, ไม่ pretrain, concat แทน SE, ไม่ band-pass, PCEN, ไม่ SpecAugment, ไม่ mixup, ไม่ class weight, ไม่ drop path)
  ทุกตัวรวมอยู่ในแผน `demo`, `main`, `cross` แล้ว ถ้าต้องการลดเวลาให้เลือกด้วย `--methods` ควรเพิ่มรายการนี้ใน RD ส่วน ablation ด้วย

## 4. ข้อจำกัดที่พบตอนเขียนโค้ด (ควรใส่ในรายงาน)

- `official_split.txt` ของ ICBHI มีผู้ป่วย 156 และ 218 อยู่ทั้ง train และ test (แบ่งระดับไฟล์) เราคงไว้ตามต้นฉบับเพื่อเทียบกับงานอื่นได้
- HF_Lung_V1 ไม่มีรหัสผู้ป่วย จึงจัดกลุ่มตามการบันทึกแบบหยาบ
- KAUH และ HF_Lung ยังไม่ได้เห็นไฟล์จริง loader เขียนตามรูปแบบในเปเปอร์ ต้องรัน `prepare_data` ตรวจก่อน

## 5. ประมาณเวลา (ยังไม่ได้วัด)

ประมาณการเดิมในเธรดคือ ICBHI 6 วิธี × 3 seeds ราว 4–7 ชั่วโมงบน RTX 4070 SUPER ให้รัน `python -m src.train --method proposed --epochs 2` จับเวลาต่อ epoch ก่อนวางแผน feature ถูก cache ใน `cache/` หลังรันครั้งแรก รอบต่อไปจึงเร็วขึ้น
