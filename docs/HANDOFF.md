# HANDOFF — โปรเจกต์จำแนกเสียงด้วย ConvNeXt + Mel/MFCC

อัปเดต: 2026-10-17 · เจ้าของ: อาคิรา (ตะวัน) นักศึกษา CP-AI

ไฟล์นี้สรุปทุกอย่างที่คุยกันไว้ เพื่อให้คนอื่นหรือ AI ตัวอื่นทำต่อได้ทันที อ่านส่วนที่ 1 และ 7 ก่อน

---

## 1. สถานะตอนนี้ (สั้นที่สุด)

- **โจทย์:** สร้างโมเดล ConvNeXt ใช้ร่วมกับ Mel spectrogram / MFCC แล้วหาคำตอบว่าทำนายเสียงประเภทไหนได้ดีที่สุด
- **ที่มา:** ข้อเสนอเดิมทำแค่เสียงปอด ตอนนี้จะขยายเป็น benchmark ข้ามหลายประเภทเสียง โดยยังให้เสียงปอดเป็นแกนหลัก
- **ทำเสร็จแล้ว:**
  - หางานวิจัยและ dataset สาธารณะ
  - ออกแบบการทดลอง
  - โหลดโค้ดจาก GitHub 8 repo มาไว้ที่ `C:\Users\tonwa\Downloads\sound-classification-repos\`
- **ยังไม่ได้ทำ:**
  - ยังไม่ได้เขียน pipeline กลาง
  - ยังไม่ได้รันโมเดลสักตัว
  - ยังไม่ได้โหลด dataset ICBHI / PhysioNet / COUGHVID
- **งานถัดไปที่ควรทำ:** ดูส่วนที่ 7

---

## 2. งานวิจัยที่ผู้ใช้ส่งมาตอนแรก (เสียงปอด 6 ฉบับ)

1. Classification of Lung Diseases from Respiratory Acoustic Sounds
2. Deep Learning-Based Classification of Common Lung Sounds via Auto-Detected Respiratory Cycles (Engin et al., Bioengineering 2026)
3. Edge-Enabled Portable Classifier for Lung Sounds Using Convolutional Neural Networks
4. Hybrid CNN-VGG16 Model for Spectrogram-Based COPD Classification and Respiratory Sound Analysis
5. Lung Sound Classification Model for On-Device AI
6. Lung Sound Classification With Multi-Feature Integration Utilizing Lightweight CNN Model

**ข้อสังเกตสำคัญจากการรีวิว:**
- งานที่รายงาน accuracy 99%+ มักแบ่งข้อมูลแบบหลวม:
  - ทำ SMOTE หรือ augmentation **ก่อน** split
  - หรือแบ่งระดับ respiratory cycle แทนระดับผู้ป่วย
  - ผลคือตัวเลขสูงเกินจริง ใช้เป็นเป้าเทียบไม่ได้
- ICBHI ใช้ "ICBHI score" (ค่าเฉลี่ยของ sensitivity กับ specificity) ซึ่งเอาไปเทียบตรงๆ กับ accuracy ไม่ได้
- ข้อมูลใน ICBHI ไม่สมดุลมาก (ไฟล์ส่วนใหญ่เป็น COPD) ต้องรายงาน macro-F1 หรือ balanced accuracy ด้วย

---

## 3. ข้อสรุปเรื่อง ConvNeXt (ตอบคำถามว่า "จะ work ไหม")

- **ConvNeXt คือ CNN:** เป็น CNN ปี 2022 ที่ปรับโครงสร้างให้ทันสมัย (depthwise conv, kernel ใหญ่, LayerNorm, GELU) ดังนั้นคำถามที่ถูกคือเทียบ CNN รุ่นใหม่กับ CNN รุ่นเก่า (VGG/ResNet) ไม่ใช่ CNN กับ ConvNeXt
- **มีคนใช้กับเสียงแล้ว:**
  - Pellegrini et al. (Interspeech 2023) ได้ mAP 0.471 บน AudioSet เท่าหรือดีกว่า audio transformer ขนาดใหญ่ โดยใช้พารามิเตอร์น้อยกว่าราว 3 เท่า
  - LungNeXt (2024) ทำเสียงปอดบน SPRSound
  - ConvNeXt ที่ปรับปรุงสำหรับเสียงกรนได้ 90.24% บน Mel spectrogram
- **ทำไมเปเปอร์ส่วนใหญ่ยังใช้ VGG/ResNet + LSTM:** เป็นเรื่องเวลา (งานจำนวนมากทำก่อน ConvNeXt ออก) ความง่ายในการตีพิมพ์ และความคุ้นเคย ไม่ใช่เพราะ ConvNeXt แย่
- **จุดอ่อนจริง:** overfit เร็ว ขนาด Pellegrini ฝึกบน AudioSet ราว 2 ล้านไฟล์ยังต้องใช้เทคนิคช่วย ส่วน ICBHI มีแค่ 920 ไฟล์ ต้องรับมือด้วย:
  - ใช้ pretrained weights
  - ใช้รุ่นเล็ก (Tiny/Nano)
  - ทำ SpecAugment และ mixup
  - ตั้ง drop path และ early stopping
- **สิ่งที่หาไม่เจอ:** งานที่เทียบ ConvNeXt + Mel/MFCC ข้ามหลายประเภทเสียงภายใต้วิธีแบ่งข้อมูลเดียวกัน นี่คือช่องว่างที่งานนี้จะเติม แต่ควรค้นใน IEEE Xplore / Google Scholar ซ้ำก่อนเขียน proposal

---

## 4. แผนการทดลอง (ที่ตกลงแนวทางไว้)

**คำถามวิจัย:** ConvNeXt + Mel/MFCC ตัวเดียวกัน ทำงานได้ดีแค่ไหนกับเสียงแต่ละประเภท และ feature แบบไหนเหมาะกับเสียงแบบไหน

### Dataset (ทั้งหมดเป็นสาธารณะ)

| ประเภทเสียง | Dataset | วิธีแบ่งข้อมูล | เป้าเทียบที่สมจริง |
|---|---|---|---|
| ปอด (แกนหลัก) | ICBHI 2017 | official split ตามผู้ป่วย | score ราว 59–62% (4 คลาส) |
| หัวใจ | PhysioNet/CinC 2016 | แบ่งตามผู้ป่วย/database (ไม่มี official split) | accuracy ราว 89% |
| ไอ | COUGHVID | แบ่งตามผู้ใช้ | ยาก ขึ้นกับ label |
| สิ่งแวดล้อม | ESC-50 | official 5-fold | CNN14 ได้ราว 95.75% |

ถ้าเวลาน้อย ใช้แค่ ICBHI + ESC-50 ก็ได้ข้อสรุปแล้ว เพราะครอบคลุมทั้งปลายยากและปลายง่าย

### ตัวแปรที่ทดลอง

- **Feature:** Mel อย่างเดียว / MFCC อย่างเดียว / Mel + MFCC + delta-MFCC ซ้อนเป็น 3 channel
  - MFCC มีแค่ 13–40 แถว ต้อง resize ให้สูงเท่ากับ Mel ก่อน
- **โมเดล:**

  | กลุ่ม | โมเดล |
  |---|---|
  | Baseline | CNN ธรรมดา, ResNet50, CNN-BiGRU |
  | ตัวหลัก | ConvNeXt-Tiny (ImageNet) และ ConvNeXt-Tiny (AudioSet) |
  | ส่วนขยาย | ConvNeXt + BiGRU (เพื่อดูว่าการทำ hybrid จำเป็นจริงไหม) |

### คุมให้เหมือนกันทุกการทดลอง

- ขนาดภาพ, augmentation, optimizer, epochs, seed
- **ทำ augment / SMOTE เฉพาะชุด train หลังจาก split แล้ว**
- metric เดียวกันทุก dataset: macro-F1, balanced accuracy, AUC (ICBHI รายงาน ICBHI score เพิ่ม)
- ใช้ PyTorch ทั้งหมด ไม่ผสม Keras

**ไม่ว่าผลจะออกมาแบบไหน งานนี้มีข้อสรุปเสมอ:**
- ConvNeXt ชนะ → CNN รุ่นใหม่ดีกว่ารุ่นเก่า
- เสมอกัน → การทำ hybrid ซับซ้อนไม่จำเป็น
- ConvNeXt แพ้ → อธิบายได้ว่าเพราะข้อมูลน้อย

---

## 5. Repo ที่โหลดมาแล้ว (`Downloads\sound-classification-repos\`)

| โฟลเดอร์ | บทบาท | หมายเหตุ |
|---|---|---|
| `pytorch-image-models` (timm) | ConvNeXt pretrained ImageNet | `timm.create_model('convnext_tiny', pretrained=True, num_classes=N)` |
| `audioset-convnext-inf` | ConvNeXt-Tiny ที่ฝึกบน AudioSet | **ไม่มีโค้ด train** และยังไม่ได้โหลด checkpoint (ราว 378 MB, Zenodo record 8020843) |
| `patch-mix_contrastive_learning` | **โครงหลักที่แนะนำ** มี ICBHI loader, official split, ICBHI score | ต้นฉบับใช้ AST ให้สลับ backbone เป็น ConvNeXt |
| `scl_icbhi2017` | Baseline PANNs CNN6/10/14 บน ICBHI/SPRSound | ใช้ weights ของ PANNs (Zenodo 3987831) |
| `lung-sound-classification` | Notebook ไล่ทีละขั้น เหมาะเริ่มต้นทำความเข้าใจ | อธิบายปัญหาการแบ่งระดับ cycle |
| `audio-tagging` | Fine-tune CNN14 บน ESC-50 (95.75%) | เป้าเทียบฝั่งเสียงสิ่งแวดล้อม |
| `ESC-50` | Dataset ESC-50 ครบ 2,000 ไฟล์เสียง + meta | ไฟล์บางส่วนจาก Freesound อาจรั่ว (preprocessing ผูกกับคลาส) ให้เขียนเป็นข้อจำกัด |
| `detect-segment-cough` | ตรวจจับและตัดเสียงไอ (fork จาก COUGHVID) | ใช้ก่อนเอา COUGHVID ไปฝึก |

**โครงสร้างโฟลเดอร์ (จัดใหม่ 2026-10-17, ลบไฟล์ชั่วคราวและสำเนาซ้ำแล้ว รวม 881 MB):**
```
sound-classification-repos/
├── HANDOFF.md
├── 1_models/              pytorch-image-models, audioset-convnext-inf
├── 2_lung_ICBHI/          patch-mix_contrastive_learning, scl_icbhi2017, lung-sound-classification
├── 3_environmental_ESC50/ audio-tagging
├── 4_cough/               detect-segment-cough
└── datasets/
    ├── ESC-50/            (846 MB, ไฟล์เสียงครบ 2,000 ไฟล์)
    └── ICBHI_2017/        (2.1 GB, wav 920 + annotation 920 ไฟล์ ชื่อไฟล์ตรงกับ official_split.txt แล้ว ไม่ต้องรัน script.sh)
```

**ข้อควรรู้เรื่องไฟล์:**
- **ไม่มีโฟลเดอร์ `.git`** เพราะโหลดฝั่ง cloud แล้วส่งข้ามมา ถ้าอยากได้ประวัติ git หรือจะ `git pull` ให้ clone ใหม่บนเครื่อง:
  ```
  git clone https://github.com/raymin0223/patch-mix_contrastive_learning.git
  ```
- **Repo ที่เคยพิจารณาแต่ยังไม่ได้โหลด:**
  - `qiuqiangkong/audioset_tagging_cnn` (PANNs ต้นฉบับ, Python 3.7)
  - `SiyuLou/AutomaticHeartSoundClassification` (เสียงหัวใจ ต้องแก้ให้แบ่งตามผู้ป่วย)
  - `skanderhamdi/attention_cnn_lstm_covid_mel_spectrogram` (เสียงไอ เป็น Keras)
  - `SJTU-YONGFU-RESEARCH-GRP/Lung-Sound-Classification-System-LungSys-I` (ตัวอย่าง deploy เป็น tflite บน Android)

---

## 6. ไอเดียหัวข้อเสียงอื่นที่เคยคุยกัน (ถ้าอยากเปลี่ยนหรือต่อยอด)

- **ไอเดียจากอาจารย์ (เสียงไอหมา):** มีงานไทยงานเดียวจาก มจธ. (DWT + ANN ได้ราว 90%) แต่ไม่มี dataset สาธารณะ ต้องเก็บข้อมูลเอง
- **เสียงไอหมู:** โยงกับวิชาเกษตรได้ มีงาน ม.เชียงใหม่ แต่ dataset ส่วนใหญ่เป็นของเอกชน
- **เสียงไอวัณโรค:** CODA TB DREAM มีเสียงไอ 733,756 เสียงจาก 2,143 คน โจทย์ยาก AUROC อยู่ที่ 0.69–0.74 เหมาะถ้าอยากทำงานระดับวิจัยจริง
- **ระบบตรวจจับเหตุฉุกเฉินในชุมชน:**
  - ใช้เสียงร้องขอความช่วยเหลือ เสียงกรีดร้อง และเสียงคนล้ม
  - dataset ที่มี: RISC shout corpus, CIRDO, คลาส Scream ใน AudioSet, A3FALL
- **อื่นๆ:** เสียงเลื่อยยนต์/ปืนในป่า (FSC22, ESC-50), ไซเรน, ยุง (HumBugDB), ผึ้ง, ช้าง

---

## 7. งานถัดไป (เรียงลำดับ)

1. ~~ลบไฟล์ชั่วคราว~~ (เสร็จแล้ว)
2. ~~โหลด ICBHI 2017~~ (เสร็จแล้ว อยู่ที่ `datasets/ICBHI_2017/` คัดลอกมาจาก `D:\ยำdowload\archive\ICBHI_final_database`) ตอนใช้กับ patch-mix ให้ชี้ path มาที่นี่ หรือคัดลอกไปไว้ใน `data/icbhi_dataset/`
   - ข้อควรเช็ค: ใน `D:\ยำdowload` มีโฟลเดอร์ภาพชื่อ "Enhanced Lung Sound Classification via Multi-Feature Integration and Lightweight ConvNeXt Model" ชื่อนี้บอกว่ามีคนใช้ ConvNeXt + multi-feature กับเสียงปอดแล้ว ต้องอ่านเทียบก่อนอ้างว่าเป็นช่องว่างงานวิจัย (ในโฟลเดอร์ยังไม่มีไฟล์เสียง มีแค่ `official_split.txt`, metadata และ `script.sh` ใน `patch-mix_contrastive_learning/data/icbhi_dataset/`)
   - ลิงก์ทางการ: https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip หรือ Kaggle "Respiratory Sound Database"
   - แตกไฟล์ลงโฟลเดอร์ข้างต้น แล้วรัน `script.sh` เพื่อแก้ชื่อไฟล์ให้ตรงกับ split
   - ระหว่างรอ ใช้ ESC-50 (มีในเครื่องแล้ว) ทดสอบ pipeline ไปก่อน
3. **รัน baseline เดิมของ `patch-mix_contrastive_learning`** ให้ได้ตัวเลขใกล้ที่เปเปอร์รายงาน (เช็คว่า environment ถูกต้อง)
4. **สลับ backbone เป็น ConvNeXt-Tiny จาก timm** แล้วรันบน ICBHI ด้วย feature 3 แบบ
5. **โหลด checkpoint AudioSet** ของ `audioset-convnext-inf` แล้วเขียน training loop สำหรับ fine-tune
6. **เขียน dataset class ให้ ESC-50** (มีแล้วในเครื่อง) โดยใช้ pipeline เดียวกัน
7. (ถ้ามีเวลา) เพิ่ม PhysioNet 2016 และ COUGHVID
8. **ทำตารางผล:** dataset × feature × model รายงาน macro-F1, balanced accuracy, AUC

**สภาพแวดล้อมที่แนะนำ:** เริ่มบนเครื่อง Windows ของผู้ใช้ (มี anaconda และน่าจะมีการ์ด NVIDIA แต่ยังไม่รู้รุ่น ให้เช็คด้วย `nvidia-smi`) ถ้า VRAM ต่ำกว่า 4 GB หรือไม่มี GPU ให้ใช้ Colab T4

**ประมาณการเวลา (ยังไม่ได้วัดจริง):** ConvNeXt-Tiny ภาพ 224 เปิด AMP และคำนวณ spectrogram ไว้ล่วงหน้า
- GPU ระดับ RTX 3060/4060: ราว 15–40 วินาทีต่อ epoch บน ICBHI
- Colab T4: ราว 30–60 วินาทีต่อ epoch
- CPU: ราว 5–10 นาทีต่อ epoch
- โมเดลอิ่มตัวใน 20–50 epoch จึงควรคิดเป็นจำนวน run: GPU ระดับกลางรันได้ราว 10–30 run ต่อคืน (run ละ 50 epoch + early stopping)
- ก่อนวางแผนให้รัน 1 epoch จับเวลาจริงก่อน

**ติดตั้ง environment:**
```
conda create -n sound python=3.10 -y
conda activate sound
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm librosa scikit-learn pandas matplotlib tqdm
python -c "import torch; print(torch.cuda.is_available())"
```

**โค้ดที่ต้องเขียนเอง (ยังไม่มี):** `datasets.py` (ICBHI/ESC-50 + split), `features.py` (Mel/MFCC/stack), `models.py` (รวมโมเดลทุกตัว), `train.py` (train + eval + บันทึก CSV สั่งผ่าน argument เช่น `--dataset esc50 --feature mel --model convnext_tiny`)

---

## 8. แหล่งอ้างอิงหลัก

- Pellegrini et al. — ConvNeXt on AudioSet: https://arxiv.org/pdf/2306.00830
- audioset-convnext-inf: https://github.com/topel/audioset-convnext-inf · checkpoints: https://zenodo.org/record/8020843
- Patch-Mix CL (ICBHI SOTA 62.37%): https://arxiv.org/pdf/2305.14032 · https://github.com/raymin0223/patch-mix_contrastive_learning
- PANNs: https://github.com/qiuqiangkong/audioset_tagging_cnn
- scl_icbhi2017: https://github.com/ilyassmoummad/scl_icbhi2017
- lung-sound-classification: https://github.com/pedramebd/lung-sound-classification
- audio-tagging: https://github.com/sithu31296/audio-tagging
- ESC-50: https://github.com/karolpiczak/esc-50
- detect-segment-cough: https://github.com/bagustris/detect-segment-cough
- Spectral & rhythm features on ESC-50 (Mel/MFCC ดีกว่า chroma/tempogram): https://arxiv.org/abs/2410.06927
- CODA TB DREAM dataset: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10996751/
- RISC shout corpus: https://arxiv.org/abs/2306.04143
- LungNeXt: https://research.mpu.edu.mo/en/publications/lungnext-a-novel-lightweight-network-utilizing-enhanced-mel-spect/
- Claude Doc รายการ repo (ของบัญชีเดิม ต้องแชร์ก่อนถึงจะเปิดได้): https://claude.ai/code/artifact/73297136-2c17-4560-aecc-88018d786ebe
