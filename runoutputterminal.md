/content/Shoes_VTO
===========================================================================
  STAGE A FOOT DETECTOR — PRODUCTION TRAINING PIPELINE                      
  Base weights: /content/Shoes_VTO/outputs/stage_a/run_v5_merged_13kp/weights/best.pt
  Dataset:      /content/Shoes_VTO/data/shuffled_v4_1_13kp/data.yaml
  Device:       0 (Tesla T4)
  Epochs:       60 (patience=40)
  Resolution:   320x320 | Batch: 16 | LR0: 0.0003
  Freeze:       10 layers
  Output:       /content/Shoes_VTO/outputs/stage_a/run_v6_v4_1_freeze_lr3e4
===========================================================================
Ultralytics 8.4.176 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, channels_last=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, cls_remap=True, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=True, cutmix=0.0, data=/content/Shoes_VTO/data/shuffled_v4_1_13kp/data.yaml, degrees=10.0, deterministic=True, device=0, dfl=1.5, dgrad=0.5, dis=6.0, distill_model=None, dlam=1.0, dlog=1.0, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=60, erasing=0.3, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=10, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=320, iou=0.7, kobj=1.0, line_width=None, lr0=0.0003, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=/content/Shoes_VTO/outputs/stage_a/run_v5_merged_13kp/weights/best.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=run_v6_v4_1_freeze_lr3e4, nbs=64, nms=None, opset=None, optimize=False, optimizer=AdamW, overlap_mask=True, patience=40, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/content/Shoes_VTO/outputs/stage_a, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/content/Shoes_VTO/outputs/stage_a/run_v6_v4_1_freeze_lr3e4, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=pose, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None

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
Freezing layer 'model.4.cv1.conv.weight'
Freezing layer 'model.4.cv1.bn.weight'
Freezing layer 'model.4.cv1.bn.bias'
Freezing layer 'model.4.cv2.conv.weight'
Freezing layer 'model.4.cv2.bn.weight'
Freezing layer 'model.4.cv2.bn.bias'
Freezing layer 'model.4.m.0.cv1.conv.weight'
Freezing layer 'model.4.m.0.cv1.bn.weight'
Freezing layer 'model.4.m.0.cv1.bn.bias'
Freezing layer 'model.4.m.0.cv2.conv.weight'
Freezing layer 'model.4.m.0.cv2.bn.weight'
Freezing layer 'model.4.m.0.cv2.bn.bias'
Freezing layer 'model.4.m.1.cv1.conv.weight'
Freezing layer 'model.4.m.1.cv1.bn.weight'
Freezing layer 'model.4.m.1.cv1.bn.bias'
Freezing layer 'model.4.m.1.cv2.conv.weight'
Freezing layer 'model.4.m.1.cv2.bn.weight'
Freezing layer 'model.4.m.1.cv2.bn.bias'
Freezing layer 'model.5.conv.weight'
Freezing layer 'model.5.bn.weight'
Freezing layer 'model.5.bn.bias'
Freezing layer 'model.6.cv1.conv.weight'
Freezing layer 'model.6.cv1.bn.weight'
Freezing layer 'model.6.cv1.bn.bias'
Freezing layer 'model.6.cv2.conv.weight'
Freezing layer 'model.6.cv2.bn.weight'
Freezing layer 'model.6.cv2.bn.bias'
Freezing layer 'model.6.m.0.cv1.conv.weight'
Freezing layer 'model.6.m.0.cv1.bn.weight'
Freezing layer 'model.6.m.0.cv1.bn.bias'
Freezing layer 'model.6.m.0.cv2.conv.weight'
Freezing layer 'model.6.m.0.cv2.bn.weight'
Freezing layer 'model.6.m.0.cv2.bn.bias'
Freezing layer 'model.6.m.1.cv1.conv.weight'
Freezing layer 'model.6.m.1.cv1.bn.weight'
Freezing layer 'model.6.m.1.cv1.bn.bias'
Freezing layer 'model.6.m.1.cv2.conv.weight'
Freezing layer 'model.6.m.1.cv2.bn.weight'
Freezing layer 'model.6.m.1.cv2.bn.bias'
Freezing layer 'model.7.conv.weight'
Freezing layer 'model.7.bn.weight'
Freezing layer 'model.7.bn.bias'
Freezing layer 'model.8.cv1.conv.weight'
Freezing layer 'model.8.cv1.bn.weight'
Freezing layer 'model.8.cv1.bn.bias'
Freezing layer 'model.8.cv2.conv.weight'
Freezing layer 'model.8.cv2.bn.weight'
Freezing layer 'model.8.cv2.bn.bias'
Freezing layer 'model.8.m.0.cv1.conv.weight'
Freezing layer 'model.8.m.0.cv1.bn.weight'
Freezing layer 'model.8.m.0.cv1.bn.bias'
Freezing layer 'model.8.m.0.cv2.conv.weight'
Freezing layer 'model.8.m.0.cv2.bn.weight'
Freezing layer 'model.8.m.0.cv2.bn.bias'
Freezing layer 'model.9.cv1.conv.weight'
Freezing layer 'model.9.cv1.bn.weight'
Freezing layer 'model.9.cv1.bn.bias'
Freezing layer 'model.9.cv2.conv.weight'
Freezing layer 'model.9.cv2.bn.weight'
Freezing layer 'model.9.cv2.bn.bias'
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1443.9±378.6 MB/s, size: 58.0 KB)
train: Scanning /content/Shoes_VTO/data/shuffled_v4_1_13kp/train/labels.cache... 1285 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 1285/1285 299.4Mit/s 0.0s
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1907.6±1248.5 MB/s, size: 277.1 KB)
val: Scanning /content/Shoes_VTO/data/shuffled_v4_1_13kp/valid/labels.cache... 161 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 161/161 33.8Mit/s 0.0s
optimizer: AdamW(lr=0.0003, momentum=0.937) with parameter groups 63 weight(decay=0.0), 73 weight(decay=0.0005), 72 bias(decay=0.0)
Plotting labels to /content/Shoes_VTO/outputs/stage_a/run_v6_v4_1_freeze_lr3e4/labels.jpg... 
Using 1285 train, 161 val images for fraction=1.0 at imgsz=320
Using 8 dataloader workers
Logging results to /content/Shoes_VTO/outputs/stage_a/run_v6_v4_1_freeze_lr3e4
Starting training for 60 epochs...

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       1/60     0.355G      1.318      3.332     0.4651     0.9972      1.265         16        320: 100% ━━━━━━━━━━━━ 81/81 10.1it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 3.0it/s 2.0s
                   all        161        286      0.833      0.845      0.929      0.587      0.805      0.792      0.819      0.385

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       2/60     0.516G      1.301      3.038     0.4698     0.9773      1.243         12        320: 100% ━━━━━━━━━━━━ 81/81 15.9it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.0it/s 0.7s
                   all        161        286      0.881      0.857      0.942      0.593      0.811      0.783      0.814      0.441

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       3/60     0.516G      1.267      2.864     0.4613     0.9484      1.235         14        320: 100% ━━━━━━━━━━━━ 81/81 17.3it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 7.9it/s 0.8s
                   all        161        286      0.861      0.873      0.949      0.608      0.827      0.837      0.868      0.464

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       4/60     0.516G      1.285      2.834     0.4594     0.9615      1.237         16        320: 100% ━━━━━━━━━━━━ 81/81 15.8it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.4it/s 0.7s
                   all        161        286      0.856      0.849       0.94        0.6      0.799      0.791      0.838      0.459

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       5/60     0.516G      1.269       2.78     0.4557       0.94      1.234         15        320: 100% ━━━━━━━━━━━━ 81/81 17.0it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.5it/s 0.7s
                   all        161        286      0.818      0.898      0.944      0.608      0.754      0.842      0.828      0.453

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       6/60     0.516G      1.256      2.717     0.4517     0.9326      1.234         22        320: 100% ━━━━━━━━━━━━ 81/81 15.4it/s 5.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 7.9it/s 0.8s
                   all        161        286      0.779      0.927      0.939      0.601      0.827      0.787      0.842      0.469

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       7/60     0.516G      1.279      2.851     0.4587     0.9419      1.237          9        320: 100% ━━━━━━━━━━━━ 81/81 16.6it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 7.8it/s 0.8s
                   all        161        286      0.864      0.871      0.945      0.597      0.831      0.815      0.849      0.459

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       8/60     0.516G      1.257      2.766     0.4607     0.9352      1.223         14        320: 100% ━━━━━━━━━━━━ 81/81 16.7it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 7.9it/s 0.8s
                   all        161        286      0.819      0.893      0.941        0.6      0.784      0.818      0.841       0.47

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       9/60     0.516G      1.265       2.72     0.4554     0.9325      1.231         13        320: 100% ━━━━━━━━━━━━ 81/81 16.3it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.2it/s 0.7s
                   all        161        286      0.835      0.858      0.932      0.599      0.797      0.808       0.84      0.467

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      10/60     0.516G      1.271      2.776     0.4557     0.9438      1.238         17        320: 100% ━━━━━━━━━━━━ 81/81 16.6it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 7.9it/s 0.8s
                   all        161        286      0.838      0.855      0.937      0.598      0.818      0.797       0.85      0.468

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      11/60     0.516G       1.24      2.704     0.4495     0.9257      1.213         15        320: 100% ━━━━━━━━━━━━ 81/81 16.1it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.0it/s 0.7s
                   all        161        286      0.869      0.851      0.942      0.605      0.844      0.801      0.858      0.477

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      12/60     0.516G      1.262      2.749     0.4524     0.9192      1.219         20        320: 100% ━━━━━━━━━━━━ 81/81 16.7it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.0it/s 0.7s
                   all        161        286       0.86      0.866      0.937      0.596      0.815      0.818      0.838      0.461

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      13/60     0.516G       1.24      2.774     0.4584     0.9142      1.222         14        320: 100% ━━━━━━━━━━━━ 81/81 15.8it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.3it/s 0.7s
                   all        161        286      0.855      0.857      0.943      0.606      0.821      0.794      0.841      0.467

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      14/60     0.516G       1.25      2.767     0.4526     0.9267      1.222         12        320: 100% ━━━━━━━━━━━━ 81/81 16.4it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.5it/s 0.7s
                   all        161        286      0.841      0.885      0.941      0.599      0.835      0.815       0.86      0.467

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      15/60     0.516G      1.261      2.714     0.4478     0.9266      1.229         13        320: 100% ━━━━━━━━━━━━ 81/81 16.3it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.0it/s 0.8s
                   all        161        286      0.825      0.888      0.935      0.593      0.808      0.808      0.852      0.463

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      16/60     0.516G      1.245      2.644     0.4517     0.9136      1.213         17        320: 100% ━━━━━━━━━━━━ 81/81 16.6it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.0it/s 0.8s
                   all        161        286      0.845      0.877      0.936      0.591      0.828      0.815      0.852      0.473

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      17/60     0.516G      1.243      2.655     0.4496      0.922      1.219         20        320: 100% ━━━━━━━━━━━━ 81/81 16.8it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.2it/s 0.7s
                   all        161        286      0.877      0.846      0.941      0.599      0.856      0.792       0.86      0.473

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      18/60     0.516G      1.235      2.652     0.4486     0.9052      1.215         21        320: 100% ━━━━━━━━━━━━ 81/81 16.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.1it/s 0.7s
                   all        161        286      0.869      0.856       0.94      0.602      0.848      0.797      0.861      0.465

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      19/60     0.516G      1.233       2.67     0.4487      0.904       1.21         15        320: 100% ━━━━━━━━━━━━ 81/81 17.3it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.6it/s 0.7s
                   all        161        286      0.862      0.879      0.942      0.609       0.83      0.825      0.852      0.475

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      20/60     0.516G      1.209      2.594       0.45     0.8814        1.2          8        320: 100% ━━━━━━━━━━━━ 81/81 15.8it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.5it/s 0.7s
                   all        161        286       0.87      0.874       0.94      0.597      0.845      0.839      0.883      0.481

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      21/60     0.516G      1.242      2.634     0.4466     0.9042      1.215         16        320: 100% ━━━━━━━━━━━━ 81/81 15.8it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.1it/s 0.7s
                   all        161        286      0.872      0.854      0.939      0.596      0.846      0.804       0.86      0.478

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      22/60     0.516G      1.234      2.604     0.4472     0.8901      1.206         10        320: 100% ━━━━━━━━━━━━ 81/81 15.5it/s 5.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.2it/s 0.7s
                   all        161        286      0.867      0.864      0.942      0.609      0.838      0.817      0.857      0.483

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      23/60     0.516G      1.225      2.599       0.45     0.9006      1.207         18        320: 100% ━━━━━━━━━━━━ 81/81 16.5it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.5it/s 0.7s
                   all        161        286       0.81      0.913      0.936      0.596      0.809      0.817      0.855      0.482

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      24/60     0.516G      1.232      2.713     0.4517     0.8993      1.216         14        320: 100% ━━━━━━━━━━━━ 81/81 16.1it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 7.5it/s 0.8s
                   all        161        286      0.849      0.863      0.935      0.604      0.788      0.798      0.833      0.462

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      25/60     0.516G      1.241       2.66     0.4475     0.8948      1.225         13        320: 100% ━━━━━━━━━━━━ 81/81 17.0it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.4it/s 0.7s
                   all        161        286      0.871      0.853       0.93      0.592      0.849      0.815      0.866       0.48

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      26/60     0.516G      1.237      2.647     0.4488      0.898      1.214         19        320: 100% ━━━━━━━━━━━━ 81/81 16.6it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 7.8it/s 0.8s
                   all        161        286      0.828       0.88      0.933      0.595      0.816      0.796      0.843      0.474

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      27/60     0.516G      1.224       2.62     0.4475     0.8987      1.206         10        320: 100% ━━━━━━━━━━━━ 81/81 16.3it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.0it/s 0.7s
                   all        161        286      0.832      0.844       0.92      0.594      0.826      0.788       0.85      0.467

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      28/60     0.516G      1.212      2.695     0.4534     0.8818      1.202         13        320: 100% ━━━━━━━━━━━━ 81/81 16.8it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 7.8it/s 0.8s
                   all        161        286      0.796      0.906      0.933      0.599      0.837      0.788      0.857      0.485

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      29/60     0.516G      1.225       2.64     0.4472     0.8829      1.216         14        320: 100% ━━━━━━━━━━━━ 81/81 16.0it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.1it/s 0.7s
                   all        161        286       0.84      0.857      0.932      0.595      0.822      0.779      0.836      0.477

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      30/60     0.516G      1.193      2.521     0.4497     0.8653       1.19         12        320: 100% ━━━━━━━━━━━━ 81/81 16.7it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.2it/s 0.7s
                   all        161        286      0.828      0.875      0.931      0.589      0.796      0.795      0.841      0.475

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      31/60     0.516G      1.225      2.514     0.4529      0.873      1.214         10        320: 100% ━━━━━━━━━━━━ 81/81 16.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.1it/s 0.7s
                   all        161        286      0.786       0.92      0.932      0.598      0.817      0.805      0.865      0.482

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      32/60     0.516G      1.213      2.584     0.4501     0.8747      1.203          8        320: 100% ━━━━━━━━━━━━ 81/81 16.7it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.1it/s 0.7s
                   all        161        286      0.789      0.937      0.938      0.593       0.83      0.813       0.86      0.486

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      33/60     0.516G      1.209      2.538     0.4472      0.863      1.201         13        320: 100% ━━━━━━━━━━━━ 81/81 16.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 7.9it/s 0.8s
                   all        161        286      0.828      0.881      0.939      0.607      0.844      0.825       0.86      0.486

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      34/60     0.516G      1.229      2.636     0.4491     0.8876      1.213         17        320: 100% ━━━━━━━━━━━━ 81/81 17.3it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.0it/s 0.7s
                   all        161        286      0.796      0.926      0.936      0.594      0.816      0.804      0.834      0.478

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      35/60     0.516G      1.213      2.545     0.4454     0.8814      1.206         16        320: 100% ━━━━━━━━━━━━ 81/81 16.7it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.3it/s 0.7s
                   all        161        286      0.817      0.881      0.924      0.586      0.808      0.811      0.841      0.488

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      36/60     0.516G      1.215      2.573     0.4488     0.8709      1.204         13        320: 100% ━━━━━━━━━━━━ 81/81 16.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.3it/s 0.7s
                   all        161        286       0.78      0.932       0.93      0.588      0.813      0.818      0.848      0.487

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      37/60     0.516G      1.203      2.569      0.449     0.8745      1.195         11        320: 100% ━━━━━━━━━━━━ 81/81 16.8it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.4it/s 0.7s
                   all        161        286      0.815      0.875      0.934      0.587      0.828      0.815      0.858      0.494

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      38/60     0.516G      1.218      2.545      0.446     0.8713      1.199         12        320: 100% ━━━━━━━━━━━━ 81/81 15.5it/s 5.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.1it/s 0.7s
                   all        161        286      0.821      0.892      0.934      0.593      0.794      0.835      0.835      0.488

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      39/60     0.516G      1.202      2.496     0.4466     0.8804      1.201         13        320: 100% ━━━━━━━━━━━━ 81/81 16.4it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.3it/s 0.7s
                   all        161        286       0.82      0.899      0.936      0.595      0.813      0.836      0.848      0.489

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      40/60     0.516G      1.209      2.568     0.4463     0.8722      1.199         12        320: 100% ━━━━━━━━━━━━ 81/81 15.5it/s 5.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.0it/s 0.8s
                   all        161        286       0.85      0.871      0.935      0.596      0.821      0.825      0.845      0.489

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      41/60     0.516G      1.214      2.508     0.4413     0.8631      1.202          9        320: 100% ━━━━━━━━━━━━ 81/81 16.8it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.3it/s 0.7s
                   all        161        286      0.849      0.867      0.934      0.597      0.824      0.829      0.865      0.498

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      42/60     0.516G      1.203      2.462     0.4508     0.8658        1.2          9        320: 100% ━━━━━━━━━━━━ 81/81 16.0it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 7.6it/s 0.8s
                   all        161        286      0.838      0.887      0.938      0.591      0.804      0.827      0.857      0.496

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      43/60     0.516G      1.207       2.53     0.4435     0.8742      1.204         15        320: 100% ━━━━━━━━━━━━ 81/81 16.8it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.1it/s 0.7s
                   all        161        286      0.842      0.888      0.939      0.602      0.825      0.829      0.866      0.498

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      44/60     0.516G      1.195      2.539     0.4461     0.8642      1.188         15        320: 100% ━━━━━━━━━━━━ 81/81 16.7it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 7.9it/s 0.8s
                   all        161        286      0.822      0.892      0.936      0.598      0.811      0.818       0.85      0.489

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      45/60     0.516G      1.196      2.498     0.4477     0.8646      1.194         20        320: 100% ━━━━━━━━━━━━ 81/81 16.0it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.3it/s 0.7s
                   all        161        286      0.842      0.877      0.937      0.598      0.852      0.807      0.856      0.496

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      46/60     0.516G      1.204      2.576     0.4462     0.8682      1.199         17        320: 100% ━━━━━━━━━━━━ 81/81 16.8it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.2it/s 0.7s
                   all        161        286      0.849      0.891      0.938      0.601      0.841      0.825      0.864      0.498

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      47/60     0.516G        1.2      2.556     0.4467     0.8744      1.198         11        320: 100% ━━━━━━━━━━━━ 81/81 15.7it/s 5.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.4it/s 0.7s
                   all        161        286      0.857      0.878      0.935        0.6      0.835      0.825      0.864      0.495

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      48/60     0.516G      1.201      2.544     0.4496     0.8647      1.201         14        320: 100% ━━━━━━━━━━━━ 81/81 17.0it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 7.9it/s 0.8s
                   all        161        286      0.846      0.879      0.933      0.596      0.834      0.813       0.85      0.492

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      49/60     0.516G      1.203      2.531     0.4464     0.8537      1.201         10        320: 100% ━━━━━━━━━━━━ 81/81 15.8it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 7.9it/s 0.8s
                   all        161        286      0.848      0.872      0.934      0.596      0.829      0.818      0.855      0.495

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      50/60     0.516G      1.209      2.484     0.4494      0.852      1.195         16        320: 100% ━━━━━━━━━━━━ 81/81 17.1it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.2it/s 0.7s
                   all        161        286      0.846      0.878      0.934      0.592      0.833      0.822      0.857      0.496
Closing dataloader mosaic
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      51/60     0.516G      1.107      1.796     0.4432     0.7001      1.148         10        320: 100% ━━━━━━━━━━━━ 81/81 12.1it/s 6.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 7.4it/s 0.8s
                   all        161        286      0.835      0.889      0.931      0.591      0.809      0.818      0.845      0.487

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      52/60     0.516G      1.104      1.812     0.4425     0.6906      1.153         10        320: 100% ━━━━━━━━━━━━ 81/81 16.5it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.3it/s 0.7s
                   all        161        286       0.84      0.881      0.932      0.594      0.801      0.815      0.846      0.488

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      53/60     0.516G      1.113      1.768     0.4416     0.6927      1.151          7        320: 100% ━━━━━━━━━━━━ 81/81 16.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.2it/s 0.7s
                   all        161        286      0.821      0.902      0.934      0.596       0.81      0.815      0.851      0.491

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      54/60     0.516G      1.113      1.837     0.4427     0.7013      1.154          9        320: 100% ━━━━━━━━━━━━ 81/81 16.8it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.3it/s 0.7s
                   all        161        286      0.827        0.9      0.934        0.6      0.824      0.822      0.856       0.49

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      55/60     0.516G      1.106      1.767     0.4396     0.6935      1.147          7        320: 100% ━━━━━━━━━━━━ 81/81 16.7it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.2it/s 0.7s
                   all        161        286       0.83      0.894      0.936      0.604      0.814      0.815      0.852      0.488

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      56/60     0.516G      1.101      1.795     0.4395     0.6875      1.143         10        320: 100% ━━━━━━━━━━━━ 81/81 16.3it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.2it/s 0.7s
                   all        161        286      0.812      0.923      0.937      0.602      0.811      0.815      0.851      0.489

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      57/60     0.516G      1.109       1.79     0.4392      0.681      1.138          9        320: 100% ━━━━━━━━━━━━ 81/81 16.8it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.4it/s 0.7s
                   all        161        286      0.832      0.896      0.936      0.601       0.81      0.819      0.849       0.49

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      58/60     0.516G      1.099      1.768     0.4388       0.67      1.147          9        320: 100% ━━━━━━━━━━━━ 81/81 15.8it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.2it/s 0.7s
                   all        161        286      0.831      0.897      0.935      0.602       0.82       0.82      0.852      0.488

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      59/60     0.516G      1.114      1.826     0.4395     0.6957      1.148         10        320: 100% ━━━━━━━━━━━━ 81/81 16.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.2it/s 0.7s
                   all        161        286      0.834      0.898      0.935      0.601      0.817      0.822      0.851      0.489

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      60/60     0.516G      1.102      1.771     0.4418     0.6796      1.144         10        320: 100% ━━━━━━━━━━━━ 81/81 16.5it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 7.3it/s 0.8s
                   all        161        286      0.838      0.895      0.935        0.6      0.794      0.821      0.849      0.488

60 epochs completed in 0.101 hours.
Optimizer stripped from /content/Shoes_VTO/outputs/stage_a/run_v6_v4_1_freeze_lr3e4/weights/last.pt, 6.6MB
Optimizer stripped from /content/Shoes_VTO/outputs/stage_a/run_v6_v4_1_freeze_lr3e4/weights/best.pt, 6.6MB

Validating /content/Shoes_VTO/outputs/stage_a/run_v6_v4_1_freeze_lr3e4/weights/best.pt...
Ultralytics 8.4.176 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
YOLOv8n-pose summary (fused): 81 layers, 3,209,267 parameters, 0 gradients, 8.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.4it/s 2.5s
                   all        161        286      0.842      0.888      0.939      0.602      0.825      0.829      0.866      0.497
             left_foot        140        140      0.806      0.893      0.933      0.607      0.779      0.829      0.837       0.48
            right_foot        146        146      0.877      0.883      0.945      0.597       0.87      0.829      0.895      0.514
Speed: 0.1ms preprocess, 6.0ms inference, 0.0ms loss, 1.9ms postprocess per image
Results saved to /content/Shoes_VTO/outputs/stage_a/run_v6_v4_1_freeze_lr3e4

===========================================================================
  Training Completed Successfully!
  Time Elapsed:    6.3 minutes
  Best Checkpoint: /content/Shoes_VTO/outputs/stage_a/run_v6_v4_1_freeze_lr3e4/weights/best.pt
===========================================================================

[*] Running final validation on best checkpoint...
Ultralytics 8.4.176 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
YOLOv8n-pose summary (fused): 81 layers, 3,209,267 parameters, 0 gradients, 8.8 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3255.0±1115.7 MB/s, size: 135.2 KB)
val: Scanning /content/Shoes_VTO/data/shuffled_v4_1_13kp/valid/labels.cache... 161 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 161/161 48.2Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 11/11 3.8it/s 2.9s
                   all        161        286      0.842      0.887      0.938      0.601      0.804      0.836      0.857      0.497
             left_foot        140        140      0.812      0.893      0.932      0.602      0.756      0.829      0.822       0.48
            right_foot        146        146      0.871      0.882      0.943      0.599      0.852      0.842      0.893      0.513
Speed: 0.4ms preprocess, 5.5ms inference, 0.0ms loss, 2.1ms postprocess per image
Results saved to /content/Shoes_VTO/runs/pose/val-4
  [+] Validation Results:
      Box mAP@50:    93.78%
      Box mAP@50-95: 60.05%
      Pose mAP@50:   85.72%
      Pose mAP@50-95:49.68%

[OK] Pipeline complete.
