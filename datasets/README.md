# Datasets และ Checkpoint

ไฟล์ข้อมูลทั้งหมดในโฟลเดอร์นี้ถูก git ignore ไว้ เพราะใหญ่เกินไปสำหรับ GitHub และ license ไม่อนุญาตให้แจกต่อ ให้ดาวน์โหลดเองแล้ววางตามโครงสร้างนี้ (ถ้าวางที่อื่น ใช้ `--data_root <path>`)

```
datasets/
├── ICBHI_2017/   101_1b1_Al_sc_Meditron.wav + .txt ... (920 คู่)
├── KAUH/         Audio Files/BP1_Asthma,I E W,P L L,70,M.wav ...
└── HF_Lung_V1/   train/*.wav + *_label.txt, test/*.wav + *_label.txt
checkpoints/
└── convnext_tiny_471mAP.pth   (หรือ model.safetensors)
```

## 1. ICBHI 2017 → `datasets/ICBHI_2017/` (ชุดหลัก)

- อ้างอิง: Rocha et al., *Physiological Measurement*, 2019
- ดาวน์โหลด: https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip หรือ Kaggle "Respiratory Sound Database"
- วางไฟล์ `.wav` และ `.txt` ไว้ระดับเดียวกัน ไม่มีโฟลเดอร์ย่อย ต้องมี wav 920 ไฟล์ (ราว 2.1 GB)
- ชื่อไฟล์ต้องตรงกับ `splits/icbhi_official_split.txt` (คัดลอกมาจาก Patch-Mix) ถ้าไม่ตรง ให้รัน `script.sh` ใน `third_party/2_lung_ICBHI/patch-mix_contrastive_learning/data/icbhi_dataset/`
- ตรวจด้วย `python -m src.prepare_data --dataset icbhi` ต้องได้ train 4,142 / test 2,756 รอบ

## 2. KAUH → `datasets/KAUH/`

- อ้างอิง: Fraiwan et al., "A dataset of lung sounds recorded from the chest wall using an electronic stethoscope", *Data in Brief*, 2021
- ดาวน์โหลด: Mendeley Data https://data.mendeley.com/datasets/jwyy9np4gv แตกไฟล์ลงโฟลเดอร์นี้ (โค้ดค้น `.wav` ทุกโฟลเดอร์ย่อย)
- ทุกไฟล์มี 3 สำเนาตามโหมดฟิลเตอร์ (ตัวอักษรแรกของชื่อไฟล์ B = Bell, D = Diaphragm, E = Extended) โค้ดใช้โหมดเดียว ค่าเริ่มต้น `--kauh_filter D`
- label มาจากช่องที่ 2 ของชื่อไฟล์: N = normal, C/Crep = crackle, มี W = wheeze, มีทั้งสอง = both ชนิดอื่น (เช่น Bronchial) ถูกข้ามและแสดงรายชื่อตอนรัน `prepare_data`
- **ต้องเช็กกับไฟล์จริง:** รูปแบบชื่อไฟล์เขียนตามที่ระบุในเปเปอร์ ถ้าต่างไป ให้แก้ `parse_kauh_sound` และ `build_kauh` ใน `src/datasets.py`

## 3. HF_Lung_V1 → `datasets/HF_Lung_V1/`

- อ้างอิง: Hsu et al., "Benchmarking of eight recurrent neural network variants for breath phase and adventitious sound detection on a self-developed open-access lung sound database—HF_Lung_V1", *PLOS ONE*, 2021
- ดาวน์โหลด: https://gitlab.com/techsupportHF/HF_Lung_V1 (wav 15 วินาที 4 kHz + ไฟล์ `_label.txt`)
- label เหตุการณ์ถูกแปลงเป็นระดับคลิป: มี `D` (DAS) = crackle, มี `Wheeze`/`Stridor`/`Rhonchi` (CAS) = wheeze, มีทั้งสอง = both, ไม่มีเลย = normal
- ไม่มีรหัสผู้ป่วย จึงจัดกลุ่มตาม อุปกรณ์ + วันที่ + ชั่วโมง ในชื่อไฟล์ (`hf_group`) ก่อนทำ group 5-fold และเขียนเป็นข้อจำกัด

## 4. Checkpoint ConvNeXt-Tiny AudioSet → `checkpoints/`

- อ้างอิง: Pellegrini et al., "Adapting a ConvNeXt model to audio classification on AudioSet", Interspeech 2023
- ดาวน์โหลด `convnext_tiny_471mAP.pth` (ราว 378 MB) จาก https://zenodo.org/record/8020843
  หรือ `model.safetensors` จาก https://huggingface.co/topel/ConvNeXt-Tiny-AT
- ชี้ไฟล์อื่นได้ด้วย `--audioset_ckpt <path>`
- น้ำหนัก ImageNet ของ ConvNeXt, VGG16, MobileNetV2, DenseNet201 โหลดอัตโนมัติผ่าน `timm` ครั้งแรกที่รัน (ต้องต่ออินเทอร์เน็ต)

## (ไม่ใช้ในแนวทาง v2)

`ESC-50/` เป็นของแนวทางเดิมใน `docs/HANDOFF.md` ไม่ได้ใช้แล้ว
