# دليل التدريب والتحديث التلقائي لموديل (Stage A Compact Detector)

## 📌 نبذة عامة والهدف الفني
موديل **Stage A Compact Detector** هو نسخة خفيفة الوزن مصممة خصيصاً لتلبية قيود الحجم والذاكرة الصارمة لمتصفحات الموبايل وتطبيقات الـ WebAR ($\le 5\text{ MB}$ للـ FP32، و $\le 2.5\text{ MB}$ للـ FP16، و $\le 1.5\text{ MB}$ للـ INT8).

تم بناء معمارية الموديل بنظام **YOLOv8-Pico (Width Scale = 0.16)** مع 16 نقطة أساسية. ونظراً لأن تدريب هذا الموديل الصغير من الصفر (From Scratch) يسبب نقصاً في الدقة (Underfitting)، فإن هذا الـ Pipeline يعتمد نظام **نقل المعرفة المسبقة (Knowledge Transfer / Teacher-Student Warm-start)** من الموديل الأصلي (Teacher) لنقل الأوزان قبل بدء التدريب، مما يرفع دقة التعرف بأكثر من **4 أضعاف**.

---

## 🛠️ هيكلية الملفات المضافة للمشروع

تم تنظيم كافة خطوات التدريب والتصدير والضغط في ملفات بايثون معيارية وقابلة لإعادة الاستخدام في كل تحديث للداتا:

```text
Shoes_VTO/
│
├── configs/
│   └── yolov8n-compact-16kp.yaml            # توصيف المعمارية المخففة (Width 0.16, 16 KPs, 2 Classes)
│
├── src/
│   ├── models/detector/
│   │   ├── transfer_compact_weights.py     # أداة نقل الأوزان من الموديل المعلم (Teacher) للـ Student
│   │   └── train_compact_detector.py       # اسكريبت تدريب الموديل المخفف بـ Ultralytics YOLO
│   │
│   └── export/
│       └── export_compact_stage_a.py       # التصدير، قص الـ 18 قناة (Path A)، وتوليد FP16 و INT8 وتحديث العقود
│
├── tools/
│   ├── train_and_package_compact_stage_a.py # الأداة الشاملة لتشغيل الـ Pipeline بالكامل بأمر واحد (Master Script)
│   └── evaluation/
│       └── eval_video_models.py            # اسكريبت المقارنة البصرية وإنتاج فيديوهات الاختبار لجميع النسخ
│
└── scripts/
    └── retrain_compact_stage_a.bat         # ملف تشغيل سريع بضغطة زر لنظام ويندوز
```

---

## 🚀 كيفية تشغيل التدريب والتحديث عند إضافة داتا جديدة

عند إضافة صور أو تسميات جديدة إلى مجلد الداتا `data/stage_a/`، يمكنك ببساطة تشغيل الـ Pipeline الكامل بأمر واحد فقط:

### 1. التشغيل التلقائي بضغطة زر واحدة (Master Script):
```powershell
& "C:\Users\miniconda3\envs\yolo\python.exe" tools/train_and_package_compact_stage_a.py
```
أو عبر الضغط المباشر على ملف:
`scripts\retrain_compact_stage_a.bat`

### خيارات التخصيص المتاحة (CLI Arguments):
```powershell
& "C:\Users\miniconda3\envs\yolo\python.exe" tools/train_and_package_compact_stage_a.py `
    --teacher "outputs/stage_a/run_v2_1/weights/best.pt" `
    --data "data/stage_a/data.yaml" `
    --epochs 200 `
    --batch 16 `
    --imgsz 320 `
    --patience 40 `
    --eval_video
```
*(إضافة خيار `--eval_video` ستقوم تلقائياً بعد التصدير بإنتاج الفيديوهات البصرية للمقارنة على `data/test_video.mp4`)*.

---

## 🔄 ماذا يحدث تلقائياً داخل الـ Pipeline في كل تحديث؟

يقوم الاسكريبت الموحد بتنفيذ 5 مراحل رئيسية بشكل متسلسل:

1. **نقل المعرفة (Knowledge Transfer):**
   - استدعاء `transfer_compact_weights.py`.
   - قراءة الـ 397 طبقة من أوزان الموديل الأصلي (`teacher best.pt`) وقصها وضبط أبعادها بما يناسب شبكة الـ Pico وحفظ الأوزان المبدئية في `outputs/stage_a/{run_name}_init.pt`.

2. **التدريب (Model Training):**
   - استدعاء `train_compact_detector.py`.
   - تدريب الشبكة باستخدام مُحسن `AdamW` ومعدل تعلم `0.001` مع تفعيل الـ Early Stopping الذكي (Patience=40) لحفظ أفضل Checkpoint باسم `best.pt`.

3. **التصدير وتطبيق عقد Path A (ONNX Export & Channel Slicing):**
   - استدعاء `export_compact_stage_a.py`.
   - تصدير الموديل إلى ONNX Opset 12 بدقة أبعاد ثابتة `[1, 3, 320, 320]`.
   - حقن عقدة `Gather` متخصصة لاقتطاع الـ 18 قناة المطلوبة فقط وفق مواصفات الـ AR SDK لتصبح أبعاد المخرج مطابقة تماماً لـ: `[1, 18, 2100]`.

4. **التحويل الكمي وتوليد النسخ الثلاث (Quantization):**
   - إنتاج نسخة **FP32** القياسية (حجم 4.93 MB).
   - إنتاج نسخة **FP16** لـ WebGPU بأوزان Float16 حقيقية وتحديث للـ Graph Shapes (حجم 2.49 MB).
   - إنتاج نسخة **INT8 Dynamic** لـ Safari WASM (حجم 1.46 MB).
   - التأكد التلقائي من عدم تجاوز الميزانية المحددة ($\le 5.0\text{ MB}$، $\le 2.5\text{ MB}$، $\le 1.5\text{ MB}$).

5. **تحديث الحزم والعقود الرسمية (Contract Synchronization):**
   - نسخ الموديلات مباشرة إلى مجلد التسليمات:
     - `shoes-vto-ai-v2/models/`
     - `deliverables/models/`
   - حساب بصمات SHA-256 وأحجام الملفات بدقة وتحديث ملفات `model-contract.json` تلقائياً بدون أي تعديل يدوي.

---

## 🧪 اختبار ومقارنة الفيديوهات (Video Evaluation)

في أي وقت ترغب فيه باختبار أداء النسخ على فيديو حقيقي، يمكنك تشغيل:
```powershell
& "C:\Users\miniconda3\envs\yolo\python.exe" tools/evaluation/eval_video_models.py
```
حيث سيقوم بتوليد 6 فيديوهات مقارنة بدقة تفصيلية في المجلد:
`data/test_video_results/`

