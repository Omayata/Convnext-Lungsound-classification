# สรุปสำหรับคนที่จะรัน demo (เพื่อน + Claude ของเพื่อน อ่านไฟล์นี้ก่อน)

อัปเดต 2026-10-21 · branch `demo-v2`

## 1. งานนี้คืออะไร

หัวข้อ: **Enhanced Lung Sound Classification via Mel–MFCC Feature Fusion and AudioSet-Pretrained ConvNeXt under Patient-Independent Evaluation**

- งาน: จำแนกเสียงปอด 4 คลาสระดับรอบหายใจ: normal / crackle / wheeze / both
- โมเดลที่เสนอ (`proposed`): ConvNeXt-Tiny ที่ pretrain บน AudioSet รับ log-Mel (50–2000 Hz) + ทางเสริม CNN เล็กรับ MFCC + ΔMFCC แล้วรวมแบบ late fusion ผ่าน SE gating
- ชุดหลัก: ICBHI 2017 official split (train 4,142 / test 2,756 รอบ) ชุดเสริม KAUH และ HF_Lung_V1 (ยังไม่ต้องใช้ใน demo)
- metric หลัก: ICBHI score = (Se + Sp) / 2 และ macro-F1
- รายละเอียดการออกแบบทั้งหมด: [`README.md`](../README.md) และ [`docs/EXPERIMENT_v2.md`](EXPERIMENT_v2.md)

## 2. สิ่งที่ต้องการจากการรันครั้งนี้

**ผล demo จากข้อมูล ICBHI จริง** เพื่อเอาไปให้อาจารย์ที่ปรึกษาดูว่าการออกแบบการทดลองโอเคไหม

- ใช้ข้อมูลจริง และ hyperparameter จริงของทุกวิธี (lr, batch, optimizer ตาม `src/methods.py`)
- ต่างจากการทดลองเต็มแค่ **5 epoch และ seed เดียว** (การทดลองเต็มคือสูงสุด 50 epoch + early stopping, 5 seeds)
- รันทั้งหมด 18 วิธี:
  - `proposed` 1 ตัว
  - คู่เทียบที่ทำซ้ำจากเปเปอร์ 6 ตัว: `wanasinghe_cnn`, `dongre_cnn_bigru`, `shakya_vgg16`, `puneet_mobilenetv2`, `park_inception`, `engin_densenet201`
  - ablation 11 ตัว แต่ละตัวคือ `proposed` ที่ตัดหรือเปลี่ยนเทคนิคเดียว: `abl_mel_only`, `abl_mel_mfcc`, `abl_imagenet`, `abl_no_pretrain`, `abl_concat`, `abl_no_bandpass`, `abl_pcen`, `abl_no_specaug`, `abl_no_mixup`, `abl_no_classweight`, `abl_no_droppath`
- ผลที่ต้องได้: `results/demo/demo_report.html` (ไฟล์เดียว เปิดใน browser ได้) และ `results/demo/results.csv`

## 3. วิธีรัน (Windows + GPU NVIDIA)

```bash
git clone -b demo-v2 https://github.com/Omayata/convnext-lungsound-classification
cd convnext-lungsound-classification

conda create -n lung python=3.10 -y
conda activate lung
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"     # ต้องได้ True
```

**วางไฟล์ (ไม่อยู่ใน git ต้องวางเอง):**

```
datasets/ICBHI_2017/   ไฟล์ .wav + .txt 920 คู่ วางตรงๆ ห้ามมีโฟลเดอร์ย่อย
                       เช่น 101_1b1_Al_sc_Meditron.wav + 101_1b1_Al_sc_Meditron.txt
checkpoints/           convnext_tiny_471mAP.pth (Zenodo 8020843)
                       หรือ model.safetensors (Hugging Face: topel/ConvNeXt-Tiny-AT) อย่างใดอย่างหนึ่ง
```

- ICBHI จากเว็บทางการ ไฟล์อยู่ในโฟลเดอร์ `ICBHI_final_database/` ใน zip ถ้ามาจาก Kaggle "Respiratory Sound Database" อยู่ใน `audio_and_txt_files/`
- โค้ดหา checkpoint ทั้งสองชื่อเจอเอง ถ้าวางไฟล์ `.pth` ที่โหลดไม่ครบไว้ จะ error ให้ลบออก
- น้ำหนัก ImageNet (ConvNeXt, VGG16, MobileNetV2, DenseNet201) โหลดอัตโนมัติผ่าน `timm` ตอนรันครั้งแรก ต้องต่อเน็ต

**รัน:**

```bash
python -m src.prepare_data --dataset icbhi     # ต้องได้ train 4142 / test 2756 รอบ ถ้าไม่ได้ ห้ามรันต่อ
python -m src.run_experiments --plan demo      # 18 วิธี สั่งซ้ำได้ จะข้ามวิธีที่เสร็จแล้ว
python scripts/demo_report.py                  # -> results/demo/demo_report.html
```

หรือดับเบิลคลิก `scripts\run_demo.bat` (ทำทั้ง 3 ขั้นแล้วเปิดรายงาน)

- ถ้าเวลาไม่พอ: ต่อท้าย `--epochs 3` หรือเลือกบางวิธีด้วย `--methods proposed abl_mel_only wanasinghe_cnn ...`
- เวลาโดยประมาณบน RTX 4070 SUPER: 1–3 ชั่วโมงสำหรับ 18 วิธี × 5 epoch (**ยังไม่ได้จับเวลาจริง** ควรจับเวลารันแรกก่อน)
- ถ้า GPU memory ไม่พอ: `--batch_size 8`

## 4. ส่งอะไรกลับมา

1. `results/demo/demo_report.html`
2. `results/demo/results.csv`
3. ผลของ `prepare_data` (จำนวนรอบ train/test) และรายชื่อวิธีที่รันล้มเหลวพร้อม error ถ้ามี
4. GPU ที่ใช้ และเวลารวม

ไม่ต้อง commit ข้อมูล ผลรัน หรือ checkpoint ขึ้น git (`.gitignore` กันไว้แล้ว)

## 5. ข้อตกลงที่ต้องเข้าใจตรงกัน (สำคัญ)

- **ห้ามแต่งหรือปรับตัวเลขผล** รายงานตามที่รันได้จริงเท่านั้น ถ้าผลไม่สวยก็ส่งตามจริง
- **ห้ามใช้ `--no_pretrained`, `--audioset_ckpt ""`, `--limit` หรือ `--data_root datasets_dummy/...` ใน demo จริง** ตัวเลือกพวกนี้มีไว้ทดสอบ pipeline เท่านั้น
- **ห้ามแก้ hyperparameter ใน `src/methods.py`** ถ้าจำเป็นต้องแก้ (เช่น batch เพราะ memory ไม่พอ) ให้จดไว้และแจ้งกลับมา
- ถ้ารันแล้ว crash ให้แก้เฉพาะที่จำเป็นต่อการรัน แล้วแจ้งว่าแก้อะไร
- **รายงานที่มีอยู่ตอนนี้ทั้งหมดมาจากข้อมูลสังเคราะห์** (`demo_report_synthetic.html`, `demo_report_dummy.html`) ตัวเลขไม่มีความหมาย ห้ามเอาไปปนกับผลจริง ไฟล์ `demo_report_dummy.html` ฉบับเก่าไม่มีป้ายบอกว่าเป็นข้อมูลจำลอง ระวังสับสน
- `scripts/demo_report.py --synthetic` ใช้เฉพาะตอนผลมาจากข้อมูลสังเคราะห์ ผลจริงให้รันแบบไม่มี flag

## 6. อ่านผล demo อย่างไร

รายละเอียดว่าแต่ละส่วนของ report บอกอะไร และควรนำเสนออย่างไร: [`docs/DEMO_REPORT_SUMMARY.md`](DEMO_REPORT_SUMMARY.md)

- demo ฝึก 5 epoch และ seed เดียว **ใช้ดูแนวโน้มเท่านั้น สรุปไม่ได้ว่าวิธีไหนดีกว่าจริง** ผลบน ICBHI แกว่งได้หลาย % ระหว่าง seed และยังไม่ได้ทดสอบสถิติ
- กราฟ ablation แสดงคะแนนของแต่ละ ablation ลบด้วย `proposed`: แท่งแดง (ติดลบ) = ตัดเทคนิคนั้นออกแล้วแย่ลง แปลว่าเทคนิคนั้นช่วย
- การสรุปว่า "ดีกว่าจริง" ต้องใช้การทดลองเต็ม: `python -m src.run_experiments --plan main` (5 seeds) แล้ว `python -m src.stats` (paired t-test ทางเดียว + Holm, Friedman + Nemenyi เมื่อมีหลาย dataset)
- ตัวเลขอ้างอิงจากเปเปอร์บน official split เดียวกัน (ไม่ได้รันเอง): BTS (Interspeech 2024) ICBHI score 63.54%

## 7. ข้อจำกัดที่รู้อยู่แล้ว

- official split ของ ICBHI มีผู้ป่วย 156 และ 218 อยู่ทั้ง train และ test (ต้นฉบับแบ่งระดับไฟล์) คงไว้เพื่อเทียบกับงานอื่นได้
- คู่เทียบเป็นการทำซ้ำจากเปเปอร์ จุดที่ต้นฉบับไม่ระบุ ใช้ค่าทั่วไป ดูตาราง deviation ใน `docs/EXPERIMENT_v2.md`
- Mel ช่วงเสียงปอดถูก interpolate เป็น 224 bin ให้เข้ากับ checkpoint AudioSet ความหมายของแต่ละ bin จึงต่างจากตอน pretrain
