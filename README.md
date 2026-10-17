# ConvNeXt Lung Sound Classification

โปรเจกต์ทดลองจำแนกเสียงด้วย **ConvNeXt + Mel spectrogram / MFCC** โดยมีเสียงปอด (ICBHI 2017) เป็นแกนหลัก แล้วเทียบข้ามประเภทเสียงอื่น (เช่น ESC-50) ภายใต้วิธีแบ่งข้อมูลและ metric เดียวกัน

> **สถานะ:** ตั้งโครงโปรเจกต์ เตรียมโค้ดอ้างอิง และเตรียม dataset ในเครื่องแล้ว ยังไม่ได้เขียน pipeline หลัก (`src/`) และยังไม่ได้รันการทดลอง
> บริบทเต็มของโปรเจกต์ (ที่มา ข้อสรุปจากงานวิจัย แผนการทดลอง ประมาณการเวลา) อยู่ใน [`docs/HANDOFF.md`](docs/HANDOFF.md)

## คำถามวิจัย

ConvNeXt ตัวเดียวกันทำนายเสียงประเภทไหนได้ดีที่สุด และ feature แบบไหน (Mel / MFCC / ซ้อนกัน) เหมาะกับเสียงแบบไหน

## แผนการทดลอง

| ตัวแปร | ค่าที่ทดลอง |
|---|---|
| Dataset | ICBHI 2017 (ปอด), ESC-50 (สิ่งแวดล้อม); ถ้ามีเวลา: PhysioNet 2016 (หัวใจ), COUGHVID (ไอ) |
| Feature | Mel / MFCC / Mel + MFCC + delta ซ้อนเป็น 3 channel |
| Model | CNN, ResNet50, CNN-BiGRU, ConvNeXt-Tiny (ImageNet), ConvNeXt-Tiny (AudioSet), ConvNeXt + BiGRU |
| Metric | macro-F1, balanced accuracy, AUC (+ ICBHI score สำหรับ ICBHI) |

**กติกาที่ต้องรักษา:**
- แบ่งข้อมูลตามผู้ป่วยหรือ fold ทางการเท่านั้น
- ทำ augmentation / SMOTE **หลัง** split และเฉพาะชุด train
- รันแต่ละชุด 3 seed

**เป้าเทียบที่สมจริง:** ICBHI score ราว 59–62% (4 คลาส, official split) และ ESC-50 ราว 95% (CNN14)

## โครงสร้าง

```
.
├── README.md
├── docs/HANDOFF.md            # บริบทและการตัดสินใจทั้งหมด อ่านก่อนเริ่มงาน
├── src/                       # pipeline ของเรา (ยังว่าง)
├── notebooks/                 # notebook ทดลอง (ยังว่าง)
├── results/                   # ผลการทดลอง (git ignore ยกเว้น README)
├── datasets/                  # ข้อมูลเสียง (git ignore ทั้งหมด ดู datasets/README.md)
│   ├── ESC-50/
│   └── ICBHI_2017/
└── third_party/               # โค้ดอ้างอิงจาก repo อื่น (คัดลอกมาเพื่อดัดแปลง)
    ├── 1_models/              # pytorch-image-models (timm), audioset-convnext-inf
    ├── 2_lung_ICBHI/          # patch-mix_contrastive_learning, scl_icbhi2017, lung-sound-classification
    ├── 3_environmental_ESC50/ # audio-tagging
    └── 4_cough/               # detect-segment-cough
```

## โค้ดอ้างอิงใน `third_party/`

คัดลอกจาก GitHub เมื่อ 2026-10-17 แบบไม่มีประวัติ git ทุกโฟลเดอร์คงไฟล์ LICENSE เดิมไว้ ส่วน `.gitignore` เดิมของแต่ละ repo เปลี่ยนชื่อเป็น `.gitignore.upstream` เพื่อไม่ให้ซ่อนไฟล์สำคัญ (เช่น `official_split.txt`)

| โฟลเดอร์ | ต้นฉบับ | ใช้ทำอะไร |
|---|---|---|
| `1_models/pytorch-image-models` | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | ConvNeXt pretrained ImageNet (ใช้ผ่าน `pip install timm` ก็ได้) |
| `1_models/audioset-convnext-inf` | [topel/audioset-convnext-inf](https://github.com/topel/audioset-convnext-inf) | ConvNeXt-Tiny ที่ฝึกบน AudioSet (ไม่มีโค้ด train; checkpoint ราว 378 MB ที่ [Zenodo](https://zenodo.org/record/8020843)) |
| `2_lung_ICBHI/patch-mix_contrastive_learning` | [raymin0223/patch-mix_contrastive_learning](https://github.com/raymin0223/patch-mix_contrastive_learning) | **โครงหลักที่แนะนำ**: ICBHI loader, official split, ICBHI score (ต้นฉบับใช้ AST) |
| `2_lung_ICBHI/scl_icbhi2017` | [ilyassmoummad/scl_icbhi2017](https://github.com/ilyassmoummad/scl_icbhi2017) | Baseline PANNs CNN6/10/14 |
| `2_lung_ICBHI/lung-sound-classification` | [pedramebd/lung-sound-classification](https://github.com/pedramebd/lung-sound-classification) | Notebook ไล่ทีละขั้น เหมาะเริ่มต้น |
| `3_environmental_ESC50/audio-tagging` | [sithu31296/audio-tagging](https://github.com/sithu31296/audio-tagging) | Fine-tune CNN14 บน ESC-50 (เป้าเทียบ) |
| `4_cough/detect-segment-cough` | [bagustris/detect-segment-cough](https://github.com/bagustris/detect-segment-cough) | ตรวจจับและตัดเสียงไอ (สำหรับ COUGHVID) |

ถ้าต้องการอัปเดตจากต้นฉบับหรือดูประวัติ ให้ clone repo ต้นฉบับแยกต่างหาก

## เริ่มต้นใช้งาน

```bash
git clone https://github.com/Omayata/Convnext-Lungsound-classification.git
cd Convnext-Lungsound-classification

conda create -n sound python=3.10 -y
conda activate sound
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm librosa scikit-learn pandas matplotlib tqdm
python -c "import torch; print(torch.cuda.is_available())"   # ควรขึ้น True
```

จากนั้นวางข้อมูลตาม [`datasets/README.md`](datasets/README.md)

## งานถัดไป

1. เขียน `src/`: `datasets.py`, `features.py`, `models.py`, `train.py`
2. ทดสอบ pipeline กับ ESC-50 fold 1 และจับเวลาต่อ epoch
3. รัน baseline ของ patch-mix บน ICBHI เพื่อยืนยันว่า data/score ถูก
4. รันการทดลองหลัก แล้วบันทึกผลลง `results/`

## ข้อควรระวัง

- **ESC-50:** มี license CC BY-NC และไฟล์บางส่วนจาก Freesound อาจผ่าน preprocessing ที่ผูกกับคลาส ให้เขียนเป็นข้อจำกัดในรายงาน
- **ICBHI:** ข้อมูลไม่สมดุลมาก ห้ามรายงานแค่ accuracy
- **ช่องว่างงานวิจัย:** มีงานชื่อ "Enhanced Lung Sound Classification via Multi-Feature Integration and Lightweight ConvNeXt Model" ต้องอ่านเทียบก่อนอ้างว่าเป็นช่องว่างงานวิจัย
