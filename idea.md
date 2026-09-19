أيوه، فهمت الفكرة الآن بشكل أدق، وبعد ما راجعت الـmarkdown والفيديو نفسه، أنا **مش هتعامل مع الموضوع كـ generic human pose estimation**.

الفيديو واضح إنه **Mobile AR shoe try-on**: الكاميرا من أعلى نسبيًا، فيه رجلين، دوران وحركة للقدم، والهدف النهائي إننا نطلع geometry ثابتة تسمح بوضع الـ3D shoe فوق القدم بشكل مستقر. والـAI layer عندكم أصلًا متصمم Detection → Keypoints → Segmentation → Tracking → Pose/Geometry. 

## أول قرار مهم: أنا أغيّر ترشيحي السابق

لـ **Keypoint model تحديدًا**، أفضل اختيار عندي للـproduction سيكون:

### **RTMPose-Tiny كـ backbone/model للـfoot keypoints**

وليس YOLO26n-pose كخيار أول.

السبب إن الـkeypoint model عندنا لن يشوف الـ1080p/full frame أصلًا؛ الـdetector هيجيب الـfoot ROI، وبعدها RTMPose يشتغل على crop صغير. وRTMPose-t عند 256×192 حوالي **3.51M parameters و0.37 GFLOPs**، بينما RTMPose-s حوالي 5.7M و0.70 GFLOPs، مع دعم واضح للتصدير إلى ONNX / ncnn / CoreML وغيرها. ([GitHub][1])

وده مناسب جدًا لأن الـarchitecture عندكم أصلًا مكتوب على شكل:

`Full Frame → Foot Detector → Left/Right ROI → Keypoint Model`

وهي نفس الفكرة اللي أنصح نثبت عليها في الـproduction. 

---

# الشكل الذي أنصح ببنائه

```text
Camera
   │
   ▼
Foot Detector
   │
   ├──────────────┐
   ▼              ▼
Left Foot ROI   Right Foot ROI
   │              │
   └──────┬───────┘
          ▼
   RTMPose-Tiny
          │
          ▼
     Foot Keypoints
          │
          ▼
    Geometry / PnP
          │
          ▼
   Temporal Smoothing
```

والأهم:

### لا تعمل Keypoint model على الصورة كاملة.

هذا وحده ممكن يفرق جدًا في latency.

---

# طيب كام Keypoint؟

هنا عندنا نقطة مهمة جدًا.

الـpublic datasets المتاحة فعليًا للقدم محدودة. أشهر dataset متخصصة من CMU فيها **6 keypoints فقط إجمالًا**:

```text
Left:
  big toe
  small toe
  heel

Right:
  big toe
  small toe
  heel
```

وفيها تقريبًا **13.5K train annotations + 0.5K validation annotations**. ([CMU Perceptual Computing Lab][2])

وكذلك H3WB عنده **6 foot keypoints** ضمن 133 whole-body keypoints، على حوالي **100K images**، مع **2D و3D annotations**. ([GitHub][3])

والـCOCO العادي أصلًا مش كفاية للfoot؛ الـfoot annotations فيه عمليًا محدودة جدًا. ([Ultralytics Docs][4])

---

# هل نستخدم 3 keypoints لكل قدم فقط؟

### ممكن كبداية، لكن مش ده الـfinal model بتاعي.

الـ3 نقاط:

```text
       Big Toe
          ●
         / \
        /   \
       /     \
 heel ●-------● Small Toe
```

ممتازة لمعرفة:

* foot center
* axis
* scale التقريبي
* orientation 2D

لكن لو هدفنا **stable 3D pose / 6DoF** فالـ3 points هتكون محدودة.

وده بالضبط السبب إن الورقة الحديثة الخاصة بالـmobile shoe try-on استخدمت **9 keypoints لكل قدم**، أي 18 إجمالًا، واعتمدت على 2D keypoints ثم PnP لاستخراج وضعية القدم. ([سبرينجر][5])

والـ9 نقاط عندهم:

```text
1. big toe
2. little toe
3. near little toe side
4. near big toe side
5. far big toe side
6. far little toe side
7. dorsum
8. heel
9. upper heel
```

وده قريب جدًا من احتياج مشروعكم. ([سبرينجر][5])

---

# إذًا الـfinal schema الذي أوصي به

أنا أبدأ بـ:

### **9 keypoints / foot**

```text
0  big_toe
1  little_toe

2  near_little_toe
3  near_big_toe

4  far_big_toe
5  far_little_toe

6  dorsum
7  heel
8  upper_heel
```

يعني:

```text
18 keypoints total
9 left
9 right
```

وده أفضل من الـ10 اللي اقترحتها قبل كده لأن عندنا الآن evidence مباشر من نظام shoe-try-on mobile مشابه جدًا لحالتكم.

---

# لكن هنا المشكلة الكبيرة

أنت قلت نقطة مهمة جدًا:

> "مفيش وقت يضيع في الداتا"

وده يغير طريقة بناء الـdataset بالكامل.

**أنا لا أنصح إطلاقًا إننا نبدأ ونصور آلاف الصور ونlabel الـ18 keypoints يدويًا.**

بدل كده نعمل:

## Stage 1 — Public data pretraining

نجمع ونوحّد:

### 1. CMU Human Foot Keypoint Dataset

عنده:

* ~13.5K training
* ~0.5K validation
* 6 foot points
* annotations جاهزة
* CC BY 4.0 للـannotations. ([CMU Perceptual Computing Lab][2])

ودي بالنسبة لي **أهم dataset مباشرة**.

---

### 2. H3WB

ده مهم جدًا لأنه أكبر بكثير:

* 100K images
* 133 whole-body keypoints
* 6 keypoints للقدم
* 2D + 3D annotations
* RGB→3D و2D→3D tasks. ([GitHub][6])

وده ممتاز للـpretraining خصوصًا لو عايزين بعدين نشتغل على geometry/3D.

---

### 3. MOOF

وده dataset حديث جدًا ومفيد جدًا لنا لأنه **foot-specific ومتحرك**.

عنده:

* 41 videos
* 15 subjects
* 30 FPS
* **14,589 frames**
* big toe / small toe / heel لكل قدم
* وفيه ankle circles وheel-toe walking وحركات قدم معقدة، بالإضافة إلى in-the-wild dance/ballet. ([TNT][7])

ده مهم لأن مشكلتكم **مش static image فقط**؛ الفيديو والاستقرار جزء أساسي.

لكن الترخيص هنا **non-commercial research only**، فلازم ننتبه جدًا للـproduction/commercial usage. ([TNT][7])

---

# وفي dataset مهم جدًا لقيته

في ورقة منشورة في 2025 عن **real-time mobile shoe try-on**، نفس المشكلة تقريبًا.

هم اضطروا يعملوا dataset خاص لأن datasets العامة غير كافية:

* 98+ فيديو
* 22 participants
* iOS + Android
* 6,655 images أصلية
* 9 keypoints لكل قدم
* augmentation إلى أكثر من 33K image. ([سبرينجر][5])

والأهم أنهم استخدموا architecture خفيفة جدًا اسمها **Single MobilePose** مبنية على MobileNetV2، وكانت:

```text
5.53M params → 1.426G FLOPs
4.39M params → 1.143G FLOPs
2.80M params → 0.796G FLOPs
```

وكانت النسخة الأخف قادرة على العمل بكفاءة، ووصلت الأجهزة الصغيرة مثل iPhone 11/11 Pro إلى حوالي **30 FPS** باستخدام TensorFlow Lite في تجاربهم. ([سبرينجر][8])

لكن **أنا لن أبني نفس MobilePose من الصفر**.

أنا أفضل:

> **RTMPose-tiny + custom 18-keypoint head**

لأننا نأخذ framework أحدث وأسهل في الـdeployment بدل إعادة بناء architecture قديمة نسبيًا.

---

# وأهم trick لتوفير الداتا

هنا أعتقد إننا نقدر نقلل الـmanual labeling جدًا.

## نعملها على مرحلتين

### Stage A — 6 keypoints

نستخدم مباشرة:

```text
CMU
+
H3WB
+
MOOF
```

كلهم على:

```text
big toe
small toe
heel
```

ثم نعمل normalization للـannotation format.

وندرب:

```text
RTMPose-tiny
        ↓
6 foot keypoints
```

---

### Stage B — نضيف الـ9-point geometry

نستخدم **synthetic data** للحصول على باقي النقاط.

وده منطقي جدًا للقدم لأن geometry بتاعها معروف نسبيًا.

فيه أصلًا datasets وresources ثلاثية الأبعاد للقدم، مثل Foot3D، الذي يوفر:

* 118 high-resolution 3D scanned feet
* 474 calibrated multi-view images
* 14 high-resolution scans للـmultiview evaluation. ([GitHub][9])

فنقدر نعمل:

```text
3D Foot
   ↓
Random Camera
   ↓
Random Rotation
   ↓
Random Scale
   ↓
Random Lighting
   ↓
Random Background
   ↓
Synthetic Image
   +
18 exact keypoints
```

وبكده بدل ما الإنسان يقعد يlabel:

```text
18 × آلاف الصور
```

الـ3D renderer يطلع الـground truth بنفسه.

---

# وبعد كده فقط نحتاج Real Fine-tuning

وده الجزء اللي فعلاً يستحق الـmanual annotation.

أنا أبدأ بـ:

### حوالي 1,000–3,000 real frames

وليس 20K أو 50K.

لكن لازم تكون مختارة بذكاء جدًا من videos حقيقية.

مثلاً:

```text
25% top view
25% rotated feet
20% one foot raised
15% feet crossing/partial occlusion
15% difficult lighting/background
```

وكمان:

```text
bare foot
sock
shoe
different shoes
different foot sizes
```

لأن الفيديو اللي عندك مثلًا واضح فيه **shoes فعلية**، ودي نقطة مهمة جدًا.

---

# نقطة شديدة الأهمية في مشروعك

من الفيديو اللي راجعته، الـinput الحقيقي **مش barefoot foot فقط**.

أنت عندك:

```text
shoe on foot
```

وده معناه أن الـkeypoint semantics لازم تتحسم من البداية:

### هل الـkeypoints بتمثل anatomical foot؟

أم:

### visible shoe landmarks؟

وده فرق ضخم جدًا.

أنا أميل بقوة إلى:

> **الـkeypoints تمثل anatomical foot geometry، لكن يجب أن تكون قابلة للاستدلال عندما تكون القدم مغطاة بالحذاء.**

وده بالذات السبب اللي يخلي training على bare/sock فقط غير كافٍ.

والورقة نفسها كانت datasetها **barefoot + socks فقط، وليس shoes**، ولذلك لن أعتمد عليها وحدها لمشروعكم. ([سبرينجر][5])

---

# الموديل النهائي الذي أرشحه

## Production Candidate

### **RTMPose-Tiny**

```text
Input:
256 × 192

Params:
~3.51M

FLOPs:
~0.37G

Output:
9 keypoints × 2 feet
```

مع:

```text
Foot Detector
      ↓
ROI crop
      ↓
RTMPose-Tiny
      ↓
18 keypoints
      ↓
confidence
      ↓
PnP / geometry
      ↓
Kalman / temporal smoothing
```

RTMPose يدعم deployment إلى ONNX وncnn وCoreML وغيرها، وده مهم جدًا عند الوصول إلى mobile production. ([GitHub][10])

---

# ولو الـTiny طلع مش كفاية؟

نعمل:

```text
RTMPose-tiny
      ↓
benchmark
      ↓
RTMPose-s
```

RTMPose-s تقريبًا ضعف الحساب، لكنه ما زال صغير نسبيًا: **5.7M params / 0.70 GFLOPs** عند 256×192. ([GitHub][1])

أنا **لا أنصح** بالقفز إلى `m/l`.

الـROI الصغيرة عندكم لا تستدعي ذلك غالبًا.

---

# ماذا عن YOLO26n-pose؟

مش وحش إطلاقًا.

الـYOLO26n-pose حاليًا حوالي **3.75M parameters و10.7 GFLOPs** في config الـ640، ويدعم custom pose/export. ([GitHub][11])

لكن المشكلة:

**لماذا أشغّل ~10.7G model على full 640 pose pipeline بينما أقدر أشغل ~0.37G RTMPose على Foot ROI 256×192؟**

عشان كده للـkeypoint stage تحديدًا:

> **RTMPose-Tiny هو اختياري الأول.**

---

# ترتيب الـdatasets عندي

| Dataset            |      Foot KPs |                  الحجم | القيمة للمشروع |
| ------------------ | ------------: | ---------------------: | -------------- |
| **CMU Foot**       |        3/foot |       ~14K annotations | ⭐⭐⭐⭐⭐          |
| **H3WB**           |        3/foot |            100K images | ⭐⭐⭐⭐⭐          |
| **MOOF**           |        3/foot |          14,589 frames | ⭐⭐⭐⭐⭐          |
| **Foot3D**         | 3D foot scans | 118 scans / 474 images | ⭐⭐⭐⭐           |
| **Your real data** |        9/foot |                   صغير | ⭐⭐⭐⭐⭐          |

والأهم: **مش هنستخدم كل dataset بنفس الشكل**.

---

# الخطة التي أرى أنها الأسرع فعليًا

```text
CMU + H3WB + MOOF
        │
        ▼
6-KP pretraining
        │
        ▼
3D/Synthetic foot generation
        │
        ▼
18-KP training
        │
        ▼
1K–3K real annotated frames
        │
        ▼
fine-tune
        │
        ▼
RTMPose-Tiny
        │
        ▼
INT8 / FP16
        │
        ▼
CoreML / NCNN / ONNX
        │
        ▼
Mobile production
```

وده في رأيي أفضل بكتير من الخطة القديمة اللي كانت هتخلينا نجمع dataset ضخم من الصفر.

---

## والأهم: أنا لا أريدك تبدأ الـannotation الآن

قبل أي annotation، أنا أقترح نعمل **Dataset audit حقيقي** للـ3 datasets:

```text
CMU
H3WB
MOOF
```

ونحوّلهم إلى schema واحد:

```json
{
  "left_foot": [
    "big_toe",
    "little_toe",
    "heel",
    ...
  ],
  "right_foot": [...]
}
```

وبعدها نحدد بالضبط **أي الـ18 keypoints ممكن نطلعها من public data، وأي 12 نقطة فقط محتاجة synthetic/real annotation**.

ده أهم خطوة حاليًا لأنه ممكن يوفر عليكم جزء كبير جدًا من الـannotation بدل ما نبدأ غلط.

[1]: https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/body8/rtmpose_body8-halpe26.md?utm_source=chatgpt.com "mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose_body8-halpe26.md at main · open-mmlab/mmpose · GitHub"
[2]: https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/ "Human Foot Keypoint Dataset"
[3]: https://github.com/wholebody3d/wholebody3d "GitHub - wholebody3d/wholebody3d: Official repository of Human3.6M 3D WholeBody (H3WB) dataset · GitHub"
[4]: https://docs.ultralytics.com/datasets/pose/coco?utm_source=chatgpt.com "COCO-Pose Estimation Dataset | Ultralytics"
[5]: https://link.springer.com/article/10.1007/s40747-025-02188-x?utm_source=chatgpt.com "A real-time mobile solution for shoe try-on using foot pose estimation and 3D processing techniques | Complex & Intelligent Systems | Springer Nature Link"
[6]: https://github.com/wholebody3d/wholebody3d?utm_source=chatgpt.com "GitHub - wholebody3d/wholebody3d: Official repository of Human3.6M 3D WholeBody (H3WB) dataset · GitHub"
[7]: https://www.tnt.uni-hannover.de/de/project/HumanMotionDatabases/?utm_source=chatgpt.com "Institut für Informationsverarbeitung (TNT) - Research: Human Motion Databases"
[8]: https://link.springer.com/article/10.1007/s40747-025-02188-x "A real-time mobile solution for shoe try-on using foot pose estimation and 3D processing techniques | Complex & Intelligent Systems | Springer Nature Link"
[9]: https://github.com/OllieBoyne/Foot3D?utm_source=chatgpt.com "GitHub - OllieBoyne/Foot3D: Dataset of scanned 3D feet · GitHub"
[10]: https://github.com/open-mmlab/mmpose/blob/main/docs/en/user_guides/how_to_deploy.md?utm_source=chatgpt.com "mmpose/docs/en/user_guides/how_to_deploy.md at main · open-mmlab/mmpose · GitHub"
[11]: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-pose.yaml?utm_source=chatgpt.com "ultralytics/ultralytics/cfg/models/26/yolo26-pose.yaml at main · ultralytics/ultralytics · GitHub"
