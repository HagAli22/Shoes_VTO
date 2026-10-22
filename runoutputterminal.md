/content/Shoes_VTO
===========================================================================
  STAGE A FOOT DETECTOR — PRODUCTION TRAINING PIPELINE                      
  Base weights: /content/Shoes_VTO/deliverables/stage_a_13kp/models/stage-a-256-13kp-v7.pt
  Dataset:      /content/Shoes_VTO/data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/data.yaml
  Device:       0 (Tesla T4)
  Epochs:       60 (patience=15)
  Resolution:   256x256 | Batch: 64 | LR0: 0.0002
  Freeze:       0 layers
  Output:       /content/Shoes_VTO/outputs/stage_a/run_v7_256_merged_with_negatives
===========================================================================
Ultralytics 8.4.183 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
engine/trainer: agnostic_nms=True, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=64, bgr=0.0, box=7.5, cache=False, cfg=None, channels_last=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, cls_remap=True, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=True, cutmix=0.0, data=/content/Shoes_VTO/data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/data.yaml, degrees=10.0, deterministic=True, device=0, dfl=1.5, dgrad=0.5, dis=6.0, distill_model=None, dlam=1.0, dlog=1.0, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=60, erasing=0.3, exist_ok=True, fliplr=0.0, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=256, iou=0.7, kobj=1.0, line_width=None, lr0=0.0002, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=/content/Shoes_VTO/deliverables/stage_a_13kp/models/stage-a-256-13kp-v7.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=run_v7_256_merged_with_negatives, nbs=64, nms=None, opset=None, optimize=False, optimizer=AdamW, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/content/Shoes_VTO/outputs/stage_a, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/content/Shoes_VTO/outputs/stage_a/run_v7_256_merged_with_negatives, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=pose, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None

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
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 795.8±464.9 MB/s, size: 27.5 KB)
train: Scanning /content/Shoes_VTO/data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/train/labels... 6620 images, 649 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 6620/6620 1.5Kit/s 4.3s
train: New cache created: /content/Shoes_VTO/data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/train/labels.cache
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1746.5±922.7 MB/s, size: 72.6 KB)
val: Scanning /content/Shoes_VTO/data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/valid/labels... 827 images, 84 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 827/827 1.5Kit/s 0.5s
val: New cache created: /content/Shoes_VTO/data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/valid/labels.cache
optimizer: AdamW(lr=0.0002, momentum=0.937) with parameter groups 63 weight(decay=0.0), 73 weight(decay=0.0005), 72 bias(decay=0.0)
Plotting labels to /content/Shoes_VTO/outputs/stage_a/run_v7_256_merged_with_negatives/labels.jpg... 
Using 6620 train, 827 val images for fraction=1.0 at imgsz=256
Using 2 dataloader workers
Logging results to /content/Shoes_VTO/outputs/stage_a/run_v7_256_merged_with_negatives
Starting training for 60 epochs...

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       1/60      1.34G      1.298      3.462     0.4108     0.9154      1.193         55        256: 100% ━━━━━━━━━━━━ 104/104 2.0it/s 52.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.5s/it 10.4s
                   all        827       1294      0.898      0.845      0.884      0.513      0.844      0.757      0.747      0.322

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       2/60      1.88G      1.254      3.196     0.4029     0.8786      1.177         83        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.8s
                   all        827       1294      0.924      0.898       0.93      0.623      0.904      0.844      0.857      0.439

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       3/60      1.88G      1.227      3.049     0.3989     0.8489      1.158         72        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0it/s 6.9s
                   all        827       1294      0.926      0.902      0.929      0.639      0.916      0.877      0.877      0.481

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       4/60      1.88G      1.206      2.935     0.3967     0.8387      1.148         76        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 45.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1it/s 6.6s
                   all        827       1294      0.933      0.908      0.937      0.631      0.917      0.871       0.88      0.486

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       5/60      1.88G      1.197      2.901     0.3886     0.8215      1.146         74        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 45.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.2s/it 8.3s
                   all        827       1294      0.941      0.911      0.941       0.65      0.918      0.876      0.883      0.486

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       6/60      1.88G      1.196      2.875     0.3876      0.812      1.144         60        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.7s
                   all        827       1294      0.922       0.91      0.929      0.636      0.909      0.878      0.881      0.496

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       7/60      1.88G      1.185      2.838     0.3887     0.7991      1.137         60        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0s/it 7.2s
                   all        827       1294      0.937      0.906      0.938      0.642      0.923      0.877      0.888      0.497

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       8/60      1.88G      1.173        2.8     0.3875     0.7955      1.135         60        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.2s/it 8.2s
                   all        827       1294      0.927      0.919      0.939      0.638      0.899      0.889      0.886      0.507

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       9/60      1.88G      1.178      2.787     0.3865       0.79      1.133         56        256: 100% ━━━━━━━━━━━━ 104/104 2.4it/s 44.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.2s/it 8.1s
                   all        827       1294      0.947      0.908      0.943      0.654      0.919      0.879      0.886      0.503

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      10/60      1.88G      1.174      2.737      0.387     0.7887      1.134         80        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0it/s 6.7s
                   all        827       1294       0.94      0.916      0.941       0.65      0.916      0.881       0.89      0.503

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      11/60      1.88G      1.182      2.813     0.3849     0.7925      1.136         70        256: 100% ━━━━━━━━━━━━ 104/104 2.4it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.9s
                   all        827       1294      0.933      0.919      0.941      0.655      0.913       0.89      0.895       0.51

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      12/60      1.88G      1.164      2.719      0.384     0.7866      1.129         64        256: 100% ━━━━━━━━━━━━ 104/104 2.4it/s 44.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0it/s 6.9s
                   all        827       1294      0.938      0.911      0.937      0.647       0.91      0.877      0.885      0.505

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      13/60      1.88G      1.162      2.716     0.3831     0.7773      1.124         77        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 45.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1it/s 6.6s
                   all        827       1294      0.934      0.922      0.941      0.651      0.922      0.882      0.889      0.499

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      14/60      1.88G      1.167      2.758     0.3814     0.7861      1.127         68        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 45.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.8s
                   all        827       1294      0.932      0.916      0.936      0.653       0.91       0.88      0.881      0.509

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      15/60      1.88G      1.154      2.702      0.385     0.7744      1.123         64        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 45.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.7s
                   all        827       1294      0.929      0.911       0.93      0.645      0.902      0.877      0.874      0.507

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      16/60      1.88G       1.16      2.666     0.3792     0.7661      1.123         70        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.4s
                   all        827       1294      0.932      0.916      0.941      0.652      0.913      0.887      0.893      0.512

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      17/60      1.88G      1.154      2.695     0.3824     0.7596      1.121         69        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 45.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 8.0s
                   all        827       1294      0.929      0.918      0.935      0.649      0.905      0.884      0.879      0.498

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      18/60      1.88G      1.148      2.632     0.3811     0.7626      1.121         72        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.6s
                   all        827       1294      0.924      0.919       0.94       0.65      0.905      0.889      0.889      0.502

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      19/60      1.88G      1.148      2.646     0.3799     0.7649      1.118         70        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 45.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.9s
                   all        827       1294      0.934      0.921      0.941      0.653      0.923      0.883      0.888      0.511

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      20/60      1.88G      1.148      2.637     0.3797     0.7596      1.117         78        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 46.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.5s
                   all        827       1294      0.937      0.923      0.946      0.656       0.91      0.887      0.887      0.502

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      21/60      1.88G      1.139      2.621     0.3819     0.7465      1.115         50        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 45.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.3s/it 9.1s
                   all        827       1294      0.932      0.926      0.946      0.654      0.912      0.893      0.888      0.514

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      22/60      1.88G      1.134      2.588      0.377     0.7442      1.112         76        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1it/s 6.3s
                   all        827       1294      0.934      0.918      0.945      0.651      0.917      0.892      0.895      0.514

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      23/60      1.88G      1.134      2.631     0.3784     0.7497      1.111         93        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 45.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0s/it 7.1s
                   all        827       1294      0.941      0.917      0.942      0.648      0.918      0.888      0.896      0.517

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      24/60      1.88G      1.134      2.592     0.3772     0.7471      1.113         77        256: 100% ━━━━━━━━━━━━ 104/104 2.4it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.8s
                   all        827       1294      0.943      0.922      0.946      0.653       0.92       0.89      0.894      0.522

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      25/60      1.88G       1.14      2.603     0.3748     0.7504      1.115         58        256: 100% ━━━━━━━━━━━━ 104/104 2.4it/s 43.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.6s
                   all        827       1294      0.936      0.915      0.939      0.653      0.916       0.89      0.893       0.52

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      26/60      1.88G      1.132      2.504     0.3739     0.7389      1.111         74        256: 100% ━━━━━━━━━━━━ 104/104 2.4it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0it/s 6.8s
                   all        827       1294      0.935      0.922      0.942      0.652      0.923      0.889      0.895      0.523

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      27/60      1.88G      1.126      2.577     0.3792     0.7371      1.107         67        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1it/s 6.5s
                   all        827       1294      0.939      0.917       0.94      0.656      0.923      0.886      0.884       0.51

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      28/60      1.88G      1.119      2.538     0.3762     0.7335      1.107         72        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1it/s 6.7s
                   all        827       1294      0.935      0.929      0.943      0.658      0.914      0.881       0.89       0.52

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      29/60      1.88G      1.114      2.509     0.3746     0.7199      1.105         77        256: 100% ━━━━━━━━━━━━ 104/104 2.4it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.2s/it 8.7s
                   all        827       1294      0.933      0.924      0.943      0.656      0.906      0.893      0.885      0.519

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      30/60      1.88G      1.129      2.572     0.3758     0.7411      1.109         57        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 46.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.3s/it 9.0s
                   all        827       1294      0.936      0.923      0.943      0.657      0.916      0.886       0.89      0.524

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      31/60      1.88G      1.124      2.535     0.3738      0.737       1.11         76        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 45.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0s/it 7.2s
                   all        827       1294      0.938       0.93      0.946      0.661      0.911      0.894      0.895      0.523

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      32/60      1.88G      1.115      2.491     0.3748      0.725      1.103         77        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.8s
                   all        827       1294      0.936      0.921      0.947      0.658      0.919      0.899      0.897       0.53

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      33/60      1.88G      1.118      2.472     0.3716     0.7231      1.104         63        256: 100% ━━━━━━━━━━━━ 104/104 2.2it/s 47.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.2s/it 8.3s
                   all        827       1294      0.936      0.929      0.949      0.658      0.913      0.886       0.89      0.524

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      34/60      1.88G      1.112      2.513     0.3735     0.7237      1.104         77        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 46.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.4s
                   all        827       1294      0.942      0.921      0.946      0.661      0.929      0.892      0.893      0.527

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      35/60      1.88G      1.122      2.472     0.3731     0.7253      1.103         73        256: 100% ━━━━━━━━━━━━ 104/104 2.2it/s 48.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.7s
                   all        827       1294      0.947      0.922      0.946      0.661      0.918      0.893      0.891      0.528

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      36/60      1.88G      1.115      2.475     0.3726     0.7206      1.102         69        256: 100% ━━━━━━━━━━━━ 104/104 2.2it/s 47.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0it/s 6.7s
                   all        827       1294      0.945      0.916      0.945      0.662      0.922      0.889      0.892      0.531

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      37/60      1.88G      1.108      2.487     0.3716     0.7134      1.097         73        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 46.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0it/s 6.8s
                   all        827       1294      0.941      0.917      0.945      0.664      0.917       0.89       0.89       0.53

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      38/60      1.88G      1.104      2.486     0.3704     0.7188      1.101         63        256: 100% ━━━━━━━━━━━━ 104/104 2.4it/s 44.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.7s
                   all        827       1294      0.942      0.918      0.948      0.665      0.923      0.893      0.899       0.53

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      39/60      1.88G      1.098      2.442     0.3697     0.7035      1.097         67        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0it/s 6.9s
                   all        827       1294       0.94      0.924      0.942      0.662      0.918      0.891      0.894      0.533

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      40/60      1.88G      1.103      2.463     0.3714      0.712      1.097         75        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0it/s 6.9s
                   all        827       1294      0.948      0.916      0.948      0.666      0.922      0.889      0.899       0.54

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      41/60      1.88G        1.1      2.431     0.3697      0.706      1.095         61        256: 100% ━━━━━━━━━━━━ 104/104 2.4it/s 44.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1it/s 6.5s
                   all        827       1294      0.938      0.928      0.948      0.666      0.924      0.899      0.904      0.542

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      42/60      1.88G      1.099      2.482     0.3723       0.71      1.095         67        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.4s
                   all        827       1294      0.949      0.923      0.951      0.666      0.926        0.9      0.902      0.538

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      43/60      1.88G      1.093      2.425     0.3713     0.6991      1.092         59        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.9s
                   all        827       1294      0.951      0.921      0.947      0.669      0.927      0.896      0.902       0.54

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      44/60      1.88G        1.1       2.42      0.368     0.7104      1.096         72        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0it/s 6.9s
                   all        827       1294      0.947       0.92      0.947      0.667      0.926       0.89      0.895      0.541

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      45/60      1.88G      1.091      2.402     0.3726     0.7006       1.09         70        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 45.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0s/it 7.1s
                   all        827       1294      0.939      0.926      0.948      0.664      0.924      0.892      0.895      0.539

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      46/60      1.88G      1.094      2.415     0.3731     0.7022      1.092         68        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1it/s 6.3s
                   all        827       1294      0.946      0.924      0.948      0.666      0.922      0.901        0.9       0.54

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      47/60      1.88G      1.094      2.416     0.3704     0.7037      1.091         72        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 45.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1it/s 6.6s
                   all        827       1294      0.946      0.925      0.948      0.667      0.923      0.892      0.895       0.54

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      48/60      1.88G      1.092      2.425       0.37     0.7079      1.096         72        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.2s/it 8.2s
                   all        827       1294      0.951       0.92      0.947      0.667      0.926      0.892      0.896      0.539

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      49/60      1.88G       1.09      2.409     0.3707     0.7015      1.088         81        256: 100% ━━━━━━━━━━━━ 104/104 2.4it/s 44.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0s/it 7.3s
                   all        827       1294      0.943      0.923      0.951      0.667      0.924      0.897        0.9      0.541

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      50/60      1.88G      1.086      2.359     0.3683     0.7003      1.092         64        256: 100% ━━━━━━━━━━━━ 104/104 2.4it/s 43.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.9s
                   all        827       1294       0.95       0.92      0.951      0.668      0.924      0.895      0.895      0.539
Closing dataloader mosaic
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      51/60      1.88G     0.9992      1.606     0.3477     0.5431      1.037         39        256: 100% ━━━━━━━━━━━━ 104/104 2.2it/s 47.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0it/s 6.8s
                   all        827       1294      0.941      0.923      0.947      0.666      0.922      0.891      0.891      0.533

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      52/60      1.88G     0.9898       1.58     0.3476     0.5382      1.035         42        256: 100% ━━━━━━━━━━━━ 104/104 2.4it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0it/s 6.8s
                   all        827       1294      0.941      0.926       0.95      0.669      0.928      0.894      0.895      0.537

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      53/60      1.88G     0.9924      1.565     0.3474     0.5345      1.034         42        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 45.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0s/it 7.0s
                   all        827       1294      0.945      0.925      0.951      0.669      0.926      0.895      0.896      0.539

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      54/60      1.88G     0.9893      1.565     0.3466     0.5361      1.033         44        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 46.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0s/it 7.1s
                   all        827       1294      0.945      0.925       0.95      0.667      0.926      0.895      0.894      0.537

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      55/60      1.88G     0.9828      1.566     0.3466     0.5301      1.031         43        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 45.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.5s
                   all        827       1294      0.945      0.926      0.949      0.669      0.927      0.895      0.895      0.538

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      56/60      1.88G     0.9847      1.562      0.347     0.5348      1.033         38        256: 100% ━━━━━━━━━━━━ 104/104 2.4it/s 44.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0s/it 7.3s
                   all        827       1294      0.946      0.924      0.949       0.67      0.926      0.897      0.895      0.539

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      57/60      1.88G     0.9842      1.572     0.3469     0.5316      1.029         43        256: 100% ━━━━━━━━━━━━ 104/104 2.4it/s 44.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0it/s 6.8s
                   all        827       1294      0.944      0.923      0.949      0.669      0.927      0.896      0.895       0.54

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      58/60      1.88G     0.9846      1.551     0.3469     0.5302      1.032         42        256: 100% ━━━━━━━━━━━━ 104/104 2.4it/s 43.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.6s
                   all        827       1294      0.945      0.924      0.949      0.669      0.926      0.897      0.894      0.539

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      59/60      1.88G     0.9789      1.575     0.3468     0.5297       1.03         45        256: 100% ━━━━━━━━━━━━ 104/104 2.4it/s 43.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1s/it 7.4s
                   all        827       1294      0.948      0.923      0.949      0.669      0.928      0.896      0.894      0.538

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      60/60      1.88G     0.9799      1.547     0.3457     0.5294      1.029         42        256: 100% ━━━━━━━━━━━━ 104/104 2.3it/s 44.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.0s/it 7.3s
                   all        827       1294      0.948      0.922      0.949       0.67      0.927      0.896      0.893      0.539

60 epochs completed in 0.890 hours.
Optimizer stripped from /content/Shoes_VTO/outputs/stage_a/run_v7_256_merged_with_negatives/weights/last.pt, 6.6MB
Optimizer stripped from /content/Shoes_VTO/outputs/stage_a/run_v7_256_merged_with_negatives/weights/best.pt, 6.6MB

Validating /content/Shoes_VTO/outputs/stage_a/run_v7_256_merged_with_negatives/weights/best.pt...
Ultralytics 8.4.183 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
YOLOv8n-pose summary (fused): 81 layers, 3,209,267 parameters, 0 gradients, 8.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.4s/it 9.7s
                   all        827       1294      0.946      0.925      0.951      0.671      0.926      0.897      0.895      0.539
             left_foot        643        643      0.946      0.925       0.95       0.66      0.923      0.891      0.882      0.524
            right_foot        648        651      0.947      0.924      0.953      0.682      0.929      0.903      0.908      0.555
Speed: 0.1ms preprocess, 4.7ms inference, 0.0ms loss, 2.1ms postprocess per image
Results saved to /content/Shoes_VTO/outputs/stage_a/run_v7_256_merged_with_negatives

===========================================================================
  Training Completed Successfully!
  Time Elapsed:    53.9 minutes
  Best Checkpoint: /content/Shoes_VTO/outputs/stage_a/run_v7_256_merged_with_negatives/weights/best.pt
===========================================================================

[*] Running final validation on best checkpoint...
Ultralytics 8.4.183 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
YOLOv8n-pose summary (fused): 81 layers, 3,209,267 parameters, 0 gradients, 8.8 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1649.4±780.7 MB/s, size: 55.1 KB)
val: Scanning /content/Shoes_VTO/data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/valid/labels.cache... 827 images, 84 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 827/827 231.2Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 52/52 5.6it/s 9.3s
                   all        827       1294      0.946      0.924      0.948      0.668      0.926      0.897       0.89      0.537
             left_foot        643        643      0.951      0.928      0.949      0.657      0.923      0.896      0.881      0.523
            right_foot        648        651      0.942      0.919      0.948      0.678      0.929      0.899      0.898      0.552
Speed: 0.2ms preprocess, 2.6ms inference, 0.0ms loss, 1.8ms postprocess per image
Results saved to /content/Shoes_VTO/runs/pose/val
  [+] Validation Results:
      Box mAP@50:    94.84%
      Box mAP@50-95: 66.76%
      Pose mAP@50:   88.99%
      Pose mAP@50-95:53.74%

[OK] Pipeline complete.
