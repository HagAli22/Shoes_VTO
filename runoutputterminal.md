/content/Shoes_VTO
===========================================================================
  STAGE A FOOT DETECTOR — PRODUCTION TRAINING PIPELINE                      
  Base weights: /content/Shoes_VTO/deliverables/stage_a_13kp/models/stage-a-320-13kp-v6-freeze.pt
  Dataset:      /content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/data.yaml
  Device:       0 (Tesla T4)
  Epochs:       60 (patience=15)
  Resolution:   320x320 | Batch: 32 | LR0: 0.0005
  Freeze:       4 layers
  Output:       /content/Shoes_VTO/outputs/stage_a/run_v7_merged_v4_1_v4_2_13kp
===========================================================================
Ultralytics 8.4.177 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=32, bgr=0.0, box=7.5, cache=False, cfg=None, channels_last=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, cls_remap=True, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=True, cutmix=0.0, data=/content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/data.yaml, degrees=10.0, deterministic=True, device=0, dfl=1.5, dgrad=0.5, dis=6.0, distill_model=None, dlam=1.0, dlog=1.0, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=60, erasing=0.3, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=4, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=320, iou=0.7, kobj=1.0, line_width=None, lr0=0.0005, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=/content/Shoes_VTO/deliverables/stage_a_13kp/models/stage-a-320-13kp-v6-freeze.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=run_v7_merged_v4_1_v4_2_13kp, nbs=64, nms=None, opset=None, optimize=False, optimizer=AdamW, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/content/Shoes_VTO/outputs/stage_a, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/content/Shoes_VTO/outputs/stage_a/run_v7_merged_v4_1_v4_2_13kp, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=pose, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf': 100% ━━━━━━━━━━━━ 755.1KB 17.0MB/s 0.0s

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 22        [15, 18, 21]  1    955165  ultralytics.nn.modules.head.Pose             [2, [13, 3], 16, None, [64, 128, 256]]
YOLOv8n-pose summary: 144 layers, 3,214,701 parameters, 3,214,685 gradients, 9.0 GFLOPs

Transferred 397/397 items from pretrained weights
Freezing layer 'model.0.conv.weight'
Freezing layer 'model.0.bn.weight'
Freezing layer 'model.0.bn.bias'
Freezing layer 'model.1.conv.weight'
Freezing layer 'model.1.bn.weight'
Freezing layer 'model.1.bn.bias'
Freezing layer 'model.2.cv1.conv.weight'
Freezing layer 'model.2.cv1.bn.weight'
Freezing layer 'model.2.cv1.bn.bias'
Freezing layer 'model.2.cv2.conv.weight'
Freezing layer 'model.2.cv2.bn.weight'
Freezing layer 'model.2.cv2.bn.bias'
Freezing layer 'model.2.m.0.cv1.conv.weight'
Freezing layer 'model.2.m.0.cv1.bn.weight'
Freezing layer 'model.2.m.0.cv1.bn.bias'
Freezing layer 'model.2.m.0.cv2.conv.weight'
Freezing layer 'model.2.m.0.cv2.bn.weight'
Freezing layer 'model.2.m.0.cv2.bn.bias'
Freezing layer 'model.3.conv.weight'
Freezing layer 'model.3.bn.weight'
Freezing layer 'model.3.bn.bias'
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: downloading yolo26n.pt for AMP checks (one-time, not used for training)...
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'weights/yolo26n.pt': 100% ━━━━━━━━━━━━ 5.3MB 66.6MB/s 0.1s
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1468.1±568.5 MB/s, size: 72.7 KB)
train: Scanning /content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/train/labels... 4128 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 4128/4128 969.1it/s 4.3s
train: New cache created: /content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/train/labels.cache
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1572.4±711.0 MB/s, size: 48.8 KB)
val: Scanning /content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/valid/labels... 516 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 516/516 980.0it/s 0.5s
val: New cache created: /content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/valid/labels.cache
optimizer: AdamW(lr=0.0005, momentum=0.937) with parameter groups 63 weight(decay=0.0), 73 weight(decay=0.0005), 72 bias(decay=0.0)
Plotting labels to /content/Shoes_VTO/outputs/stage_a/run_v7_merged_v4_1_v4_2_13kp/labels.jpg... 
Using 4128 train, 516 val images for fraction=1.0 at imgsz=320
Using 8 dataloader workers
Logging results to /content/Shoes_VTO/outputs/stage_a/run_v7_merged_v4_1_v4_2_13kp
Starting training for 60 epochs...

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       1/60     0.793G      1.413      4.412     0.4662      1.358      1.332         86        320: 100% ━━━━━━━━━━━━ 129/129 8.8it/s 14.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 2.3it/s 4.0s
                   all        516        874      0.539      0.847      0.682      0.392      0.475      0.637      0.509       0.21

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       2/60      1.15G      1.366      4.006      0.457      1.282      1.309         90        320: 100% ━━━━━━━━━━━━ 129/129 12.5it/s 10.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874       0.54      0.851      0.687      0.414      0.492      0.718      0.561      0.243

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       3/60      1.15G      1.337      3.771     0.4504      1.242      1.287         91        320: 100% ━━━━━━━━━━━━ 129/129 12.6it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.531      0.907      0.719      0.439      0.503      0.733      0.612      0.276

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       4/60      1.15G      1.308      3.686     0.4444      1.219      1.273         80        320: 100% ━━━━━━━━━━━━ 129/129 12.5it/s 10.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.558      0.867      0.734      0.444      0.527      0.705      0.612      0.282

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       5/60      1.15G      1.295      3.528     0.4458      1.201      1.263         88        320: 100% ━━━━━━━━━━━━ 129/129 12.9it/s 10.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.7it/s 1.9s
                   all        516        874      0.559      0.894      0.732      0.449      0.519      0.766      0.624        0.3

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       6/60      1.15G      1.288      3.476     0.4406      1.185      1.262         87        320: 100% ━━━━━━━━━━━━ 129/129 12.5it/s 10.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.562      0.887      0.733      0.447      0.527      0.791      0.632      0.298

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       7/60      1.15G      1.272       3.44     0.4375      1.176      1.254         83        320: 100% ━━━━━━━━━━━━ 129/129 12.6it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 1.9s
                   all        516        874      0.565       0.91      0.735      0.459      0.521      0.802      0.636      0.286

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       8/60      1.15G      1.269       3.41      0.435       1.16      1.253         93        320: 100% ━━━━━━━━━━━━ 129/129 12.7it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.561      0.894      0.732      0.445      0.522      0.787      0.637      0.306

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       9/60      1.15G      1.263      3.345     0.4336       1.16      1.244         82        320: 100% ━━━━━━━━━━━━ 129/129 12.7it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 1.9s
                   all        516        874      0.593      0.853      0.782      0.492      0.563      0.769      0.687      0.332

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      10/60      1.15G      1.261      3.321     0.4292      1.151      1.244         89        320: 100% ━━━━━━━━━━━━ 129/129 12.9it/s 10.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.569      0.907      0.754      0.457      0.536       0.81      0.667       0.33

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      11/60      1.15G      1.254      3.319     0.4334      1.144      1.244        108        320: 100% ━━━━━━━━━━━━ 129/129 12.7it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.591      0.862      0.754      0.469      0.564      0.782      0.674       0.32

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      12/60      1.15G       1.25      3.245     0.4256      1.134      1.234         94        320: 100% ━━━━━━━━━━━━ 129/129 12.8it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.569      0.906      0.746      0.462      0.526      0.823      0.655      0.332

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      13/60      1.15G      1.251      3.263     0.4252      1.135      1.237         97        320: 100% ━━━━━━━━━━━━ 129/129 12.9it/s 10.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.583      0.889      0.768      0.476      0.576      0.807      0.705      0.349

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      14/60      1.15G       1.24      3.234     0.4242      1.129      1.228        105        320: 100% ━━━━━━━━━━━━ 129/129 13.0it/s 10.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.563      0.909       0.73       0.45      0.537      0.804      0.651      0.323

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      15/60      1.15G      1.235      3.186     0.4248      1.125       1.23         99        320: 100% ━━━━━━━━━━━━ 129/129 13.0it/s 9.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.4it/s 2.0s
                   all        516        874      0.573      0.893      0.757      0.478      0.557      0.803      0.686      0.342

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      16/60      1.15G      1.231      3.105     0.4193      1.109      1.223         87        320: 100% ━━━━━━━━━━━━ 129/129 12.7it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.7it/s 1.9s
                   all        516        874      0.575      0.913      0.753       0.48      0.543      0.835      0.677       0.34

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      17/60      1.15G      1.229      3.117      0.419      1.114      1.228         83        320: 100% ━━━━━━━━━━━━ 129/129 12.6it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.574      0.896      0.759      0.475      0.547      0.795      0.677      0.346

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      18/60      1.15G      1.217      3.095     0.4177      1.106      1.222        104        320: 100% ━━━━━━━━━━━━ 129/129 12.8it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.567      0.896      0.748      0.477      0.537      0.827      0.672      0.347

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      19/60      1.15G      1.219       3.08       0.42      1.099      1.218         82        320: 100% ━━━━━━━━━━━━ 129/129 12.8it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.583      0.903      0.764      0.487      0.557      0.815      0.688      0.354

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      20/60      1.15G      1.215       3.01     0.4134      1.081      1.211         88        320: 100% ━━━━━━━━━━━━ 129/129 12.3it/s 10.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 1.9s
                   all        516        874      0.574      0.893      0.758      0.476      0.559      0.828      0.695       0.35

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      21/60      1.15G      1.204      2.973     0.4149      1.082      1.206         80        320: 100% ━━━━━━━━━━━━ 129/129 12.8it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.4it/s 2.0s
                   all        516        874      0.595      0.904       0.78      0.487      0.578      0.819      0.714      0.359

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      22/60      1.15G      1.204       2.99     0.4179      1.091      1.209         80        320: 100% ━━━━━━━━━━━━ 129/129 12.9it/s 10.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.609       0.89      0.796      0.506      0.583      0.829      0.734      0.377

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      23/60      1.15G      1.212      2.999     0.4168      1.082      1.208         70        320: 100% ━━━━━━━━━━━━ 129/129 12.5it/s 10.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874       0.58      0.916      0.749      0.451      0.543      0.848      0.679      0.349

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      24/60      1.15G      1.201      2.923     0.4132      1.069      1.208        105        320: 100% ━━━━━━━━━━━━ 129/129 12.6it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.601      0.877      0.793        0.5      0.595      0.806      0.724       0.38

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      25/60      1.15G      1.193      2.923     0.4094      1.065      1.206         98        320: 100% ━━━━━━━━━━━━ 129/129 12.3it/s 10.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.7it/s 1.9s
                   all        516        874      0.584      0.902      0.783      0.508      0.577      0.811      0.704       0.36

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      26/60      1.15G      1.195      2.898     0.4103      1.057        1.2         84        320: 100% ━━━━━━━━━━━━ 129/129 12.9it/s 10.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.4it/s 2.0s
                   all        516        874      0.604      0.878      0.788        0.5      0.584      0.822      0.722      0.378

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      27/60      1.15G      1.192      2.925     0.4092      1.063      1.202         87        320: 100% ━━━━━━━━━━━━ 129/129 12.9it/s 10.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.7it/s 1.9s
                   all        516        874      0.594      0.886       0.79      0.504      0.582      0.829       0.73      0.383

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      28/60      1.15G      1.194      2.862     0.4124      1.056        1.2         95        320: 100% ━━━━━━━━━━━━ 129/129 12.7it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.603      0.896      0.805      0.506      0.582      0.849      0.743      0.393

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      29/60      1.15G      1.183      2.898     0.4103      1.065      1.196        103        320: 100% ━━━━━━━━━━━━ 129/129 12.7it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874        0.6      0.891      0.792      0.507      0.582      0.847      0.734      0.391

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      30/60      1.15G      1.177      2.846     0.4094      1.048      1.192         77        320: 100% ━━━━━━━━━━━━ 129/129 12.6it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.608      0.894      0.814      0.514      0.584      0.842      0.748        0.4

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      31/60      1.15G      1.185      2.823     0.4051      1.054      1.197         84        320: 100% ━━━━━━━━━━━━ 129/129 12.6it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.595      0.916      0.794      0.512       0.57      0.873      0.737      0.398

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      32/60      1.15G      1.173      2.803     0.4082      1.044      1.196         93        320: 100% ━━━━━━━━━━━━ 129/129 12.7it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.591      0.926      0.796      0.514      0.578      0.856      0.747      0.401

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      33/60      1.15G      1.177      2.763     0.4057      1.038      1.197         81        320: 100% ━━━━━━━━━━━━ 129/129 13.0it/s 9.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.7it/s 1.9s
                   all        516        874      0.622      0.872      0.817      0.524      0.615      0.828      0.767      0.405

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      34/60      1.15G      1.172      2.756     0.4118      1.031      1.188         77        320: 100% ━━━━━━━━━━━━ 129/129 12.6it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874        0.6      0.916      0.795      0.514      0.578      0.857      0.735      0.404

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      35/60      1.15G      1.162      2.735     0.4045      1.026      1.183         82        320: 100% ━━━━━━━━━━━━ 129/129 12.8it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874       0.62      0.892      0.809      0.519      0.609      0.841      0.753      0.411

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      36/60      1.15G      1.163      2.749     0.4062      1.033      1.181         78        320: 100% ━━━━━━━━━━━━ 129/129 12.4it/s 10.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.592      0.918      0.796      0.515      0.571      0.859      0.739      0.402

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      37/60      1.15G      1.159      2.724     0.4057      1.026      1.182        124        320: 100% ━━━━━━━━━━━━ 129/129 12.7it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 1.9s
                   all        516        874      0.605      0.901      0.782      0.503      0.585       0.86      0.729      0.393

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      38/60      1.15G      1.165       2.67     0.4055      1.022       1.18         92        320: 100% ━━━━━━━━━━━━ 129/129 12.6it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.627      0.889      0.821      0.529      0.607      0.854      0.769      0.419

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      39/60      1.15G      1.155      2.673     0.4013      1.015      1.184         89        320: 100% ━━━━━━━━━━━━ 129/129 12.6it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874       0.61      0.928      0.808      0.523      0.591      0.888      0.764      0.415

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      40/60      1.15G      1.155      2.702     0.4066      1.019       1.18         90        320: 100% ━━━━━━━━━━━━ 129/129 12.6it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 1.9s
                   all        516        874      0.603      0.913       0.79      0.515      0.588      0.876      0.749      0.411

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      41/60      1.15G      1.144      2.642      0.403      1.011      1.172         95        320: 100% ━━━━━━━━━━━━ 129/129 12.7it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 1.9s
                   all        516        874      0.604      0.926      0.805      0.518       0.59      0.886      0.764      0.421

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      42/60      1.15G      1.152      2.621     0.3998      1.015      1.177        100        320: 100% ━━━━━━━━━━━━ 129/129 12.5it/s 10.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.7it/s 1.9s
                   all        516        874      0.609      0.911      0.798      0.513      0.597      0.874      0.762      0.422

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      43/60      1.15G      1.133      2.611     0.4021     0.9995      1.174        102        320: 100% ━━━━━━━━━━━━ 129/129 12.4it/s 10.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.602       0.92      0.789       0.51       0.59      0.873      0.749      0.417

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      44/60      1.15G      1.138      2.652     0.4023          1      1.173         85        320: 100% ━━━━━━━━━━━━ 129/129 13.0it/s 9.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.613      0.913      0.803      0.521      0.598      0.867      0.758      0.426

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      45/60      1.15G      1.142      2.635     0.4022      1.004      1.174        100        320: 100% ━━━━━━━━━━━━ 129/129 12.7it/s 10.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.605      0.928      0.794      0.515      0.585      0.887      0.749      0.423

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      46/60      1.15G      1.143       2.54     0.3959     0.9974      1.173         91        320: 100% ━━━━━━━━━━━━ 129/129 12.6it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.613      0.907      0.808      0.527      0.591       0.87      0.759      0.432

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      47/60      1.15G      1.142      2.614     0.3995     0.9981      1.177         87        320: 100% ━━━━━━━━━━━━ 129/129 12.7it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.613      0.921      0.812      0.529      0.596      0.877      0.766      0.431

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      48/60      1.15G      1.138      2.612     0.4016     0.9951      1.176         96        320: 100% ━━━━━━━━━━━━ 129/129 13.0it/s 9.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.614      0.897      0.806      0.526      0.598      0.864      0.762       0.43

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      49/60      1.15G       1.13      2.604     0.4021     0.9946       1.17         76        320: 100% ━━━━━━━━━━━━ 129/129 12.4it/s 10.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.613      0.912       0.81      0.528      0.595      0.881      0.767       0.43

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      50/60      1.15G      1.117      2.527     0.3977     0.9863      1.164         78        320: 100% ━━━━━━━━━━━━ 129/129 12.9it/s 10.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.617      0.906      0.808      0.527      0.595      0.878      0.766      0.427
Closing dataloader mosaic
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      51/60      1.15G       1.05      1.756     0.3852     0.8393      1.116         54        320: 100% ━━━━━━━━━━━━ 129/129 10.7it/s 12.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.623      0.899      0.811      0.525      0.608      0.862      0.766      0.429

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      52/60      1.15G      1.038      1.749     0.3818     0.8324      1.111         56        320: 100% ━━━━━━━━━━━━ 129/129 13.0it/s 10.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.7it/s 1.9s
                   all        516        874      0.619      0.893      0.801      0.523        0.6      0.865      0.756      0.425

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      53/60      1.15G      1.039      1.728     0.3834     0.8316      1.115         53        320: 100% ━━━━━━━━━━━━ 129/129 12.6it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874       0.62       0.89       0.81      0.527      0.602      0.861      0.765      0.432

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      54/60      1.15G      1.042       1.72     0.3842     0.8263      1.119         58        320: 100% ━━━━━━━━━━━━ 129/129 12.7it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.618      0.897      0.805      0.524      0.596      0.865      0.755      0.427

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      55/60      1.15G      1.041      1.709     0.3809     0.8278      1.114         57        320: 100% ━━━━━━━━━━━━ 129/129 12.4it/s 10.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.628       0.88      0.811      0.529      0.608      0.849      0.763      0.433

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      56/60      1.15G      1.045      1.715      0.381     0.8316      1.116         54        320: 100% ━━━━━━━━━━━━ 129/129 12.9it/s 10.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.4it/s 2.0s
                   all        516        874      0.622      0.893      0.809      0.528        0.6      0.872      0.763      0.434

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      57/60      1.15G      1.029      1.708     0.3826     0.8273       1.11         53        320: 100% ━━━━━━━━━━━━ 129/129 13.3it/s 9.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.625       0.89       0.81      0.529      0.606      0.859      0.766      0.435

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      58/60      1.15G       1.04      1.724       0.38     0.8252      1.117         54        320: 100% ━━━━━━━━━━━━ 129/129 13.1it/s 9.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.626      0.889      0.809      0.527      0.608      0.858      0.761      0.434

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      59/60      1.15G      1.037      1.695     0.3806     0.8227      1.113         53        320: 100% ━━━━━━━━━━━━ 129/129 13.0it/s 9.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.627      0.893      0.809      0.528      0.607      0.865      0.766      0.434

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      60/60      1.15G      1.034      1.699     0.3807     0.8217      1.116         52        320: 100% ━━━━━━━━━━━━ 129/129 12.7it/s 10.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 1.9s
                   all        516        874      0.624      0.892      0.807      0.528      0.605      0.866      0.764      0.434

60 epochs completed in 0.210 hours.
Optimizer stripped from /content/Shoes_VTO/outputs/stage_a/run_v7_merged_v4_1_v4_2_13kp/weights/last.pt, 6.6MB
Optimizer stripped from /content/Shoes_VTO/outputs/stage_a/run_v7_merged_v4_1_v4_2_13kp/weights/best.pt, 6.6MB

Validating /content/Shoes_VTO/outputs/stage_a/run_v7_merged_v4_1_v4_2_13kp/weights/best.pt...
Ultralytics 8.4.177 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
YOLOv8n-pose summary (fused): 81 layers, 3,209,267 parameters, 0 gradients, 8.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 2.1it/s 4.4s
                   all        516        874      0.626       0.89       0.81      0.529      0.605      0.862      0.767      0.435
             left_foot        424        425      0.609       0.88      0.795      0.512      0.589      0.852       0.75      0.424
            right_foot        447        449      0.643      0.899      0.825      0.545      0.622      0.873      0.784      0.445
Speed: 0.2ms preprocess, 2.2ms inference, 0.0ms loss, 1.8ms postprocess per image
Results saved to /content/Shoes_VTO/outputs/stage_a/run_v7_merged_v4_1_v4_2_13kp

===========================================================================
  Training Completed Successfully!
  Time Elapsed:    13.0 minutes
  Best Checkpoint: /content/Shoes_VTO/outputs/stage_a/run_v7_merged_v4_1_v4_2_13kp/weights/best.pt
===========================================================================

[*] Running final validation on best checkpoint...
Ultralytics 8.4.177 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
YOLOv8n-pose summary (fused): 81 layers, 3,209,267 parameters, 0 gradients, 8.8 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1359.7±724.6 MB/s, size: 29.1 KB)
val: Scanning /content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/valid/labels.cache... 516 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 516/516 166.5Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 33/33 6.9it/s 4.8s
                   all        516        874      0.615       0.91      0.809      0.526      0.594      0.875      0.764      0.436
             left_foot        424        425      0.601      0.901      0.797      0.511      0.579      0.864      0.752      0.428
            right_foot        447        449       0.63       0.92      0.822      0.541       0.61      0.887      0.776      0.445
Speed: 0.2ms preprocess, 2.6ms inference, 0.0ms loss, 1.3ms postprocess per image
Results saved to /content/Shoes_VTO/runs/pose/val
  [+] Validation Results:
      Box mAP@50:    80.93%
      Box mAP@50-95: 52.63%
      Pose mAP@50:   76.37%
      Pose mAP@50-95:43.61%

[OK] Pipeline complete.
