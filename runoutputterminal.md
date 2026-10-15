/content/Shoes_VTO
===========================================================================
  STAGE A FOOT DETECTOR — PRODUCTION TRAINING PIPELINE                      
  Base weights: /content/Shoes_VTO/outputs/stage_a/run_v7_merged_v4_1_v4_2_13kp/weights/best.pt
  Dataset:      /content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/data.yaml
  Device:       0 (Tesla T4)
  Epochs:       50 (patience=15)
  Resolution:   320x320 | Batch: 32 | LR0: 0.0003
  Freeze:       0 layers
  Output:       /content/Shoes_VTO/outputs/stage_a/run_v7_merged_full_unfreeze_noflip
===========================================================================
Ultralytics 8.4.177 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
engine/trainer: agnostic_nms=True, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=32, bgr=0.0, box=7.5, cache=False, cfg=None, channels_last=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, cls_remap=True, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=True, cutmix=0.0, data=/content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/data.yaml, degrees=10.0, deterministic=True, device=0, dfl=1.5, dgrad=0.5, dis=6.0, distill_model=None, dlam=1.0, dlog=1.0, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=50, erasing=0.3, exist_ok=True, fliplr=0.0, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=320, iou=0.7, kobj=1.0, line_width=None, lr0=0.0003, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=/content/Shoes_VTO/outputs/stage_a/run_v7_merged_v4_1_v4_2_13kp/weights/best.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=run_v7_merged_full_unfreeze_noflip, nbs=64, nms=None, opset=None, optimize=False, optimizer=AdamW, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/content/Shoes_VTO/outputs/stage_a, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/content/Shoes_VTO/outputs/stage_a/run_v7_merged_full_unfreeze_noflip, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=pose, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None

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
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1500.0±758.1 MB/s, size: 72.7 KB)
train: Scanning /content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/train/labels.cache... 4128 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 4128/4128 597.0Mit/s 0.0s
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2044.8±770.5 MB/s, size: 48.8 KB)
val: Scanning /content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/valid/labels.cache... 516 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 516/516 94.1Mit/s 0.0s
optimizer: AdamW(lr=0.0003, momentum=0.937) with parameter groups 63 weight(decay=0.0), 73 weight(decay=0.0005), 72 bias(decay=0.0)
Plotting labels to /content/Shoes_VTO/outputs/stage_a/run_v7_merged_full_unfreeze_noflip/labels.jpg... 
Using 4128 train, 516 val images for fraction=1.0 at imgsz=320
Using 8 dataloader workers
Logging results to /content/Shoes_VTO/outputs/stage_a/run_v7_merged_full_unfreeze_noflip
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       1/50      1.08G      1.206      3.087     0.4086     0.9283      1.205         86        320: 100% ━━━━━━━━━━━━ 129/129 9.5it/s 13.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 2.3it/s 3.8s
                   all        516        874      0.879      0.871      0.877      0.539      0.843      0.774      0.753      0.338

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       2/50      1.44G      1.193      2.928     0.4048     0.8576        1.2         90        320: 100% ━━━━━━━━━━━━ 129/129 12.1it/s 10.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.923      0.885      0.908      0.587      0.908      0.845      0.849      0.434

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       3/50      1.44G      1.165      2.793     0.4054     0.8263      1.184         91        320: 100% ━━━━━━━━━━━━ 129/129 12.0it/s 10.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.915      0.898      0.908      0.587      0.907      0.845      0.851      0.456

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       4/50      1.44G      1.148       2.69     0.3988     0.8031      1.178         80        320: 100% ━━━━━━━━━━━━ 129/129 11.7it/s 11.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 1.9s
                   all        516        874      0.902      0.903       0.91       0.61      0.909      0.846      0.851      0.453

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       5/50      1.44G       1.15      2.645     0.4023      0.797      1.177         88        320: 100% ━━━━━━━━━━━━ 129/129 11.8it/s 11.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874       0.93      0.906       0.92      0.614      0.899      0.872      0.861       0.47

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       6/50      1.44G      1.147      2.634     0.3996     0.7841      1.174         87        320: 100% ━━━━━━━━━━━━ 129/129 11.9it/s 10.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.923      0.912       0.92       0.61      0.899      0.874      0.864      0.467

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       7/50      1.44G      1.136      2.639     0.3969     0.7763      1.171         83        320: 100% ━━━━━━━━━━━━ 129/129 11.8it/s 10.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.939       0.91       0.92      0.622      0.909      0.877      0.871      0.474

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       8/50      1.44G      1.131      2.597     0.3965     0.7709      1.169         93        320: 100% ━━━━━━━━━━━━ 129/129 11.9it/s 10.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.934      0.886      0.912      0.618       0.92      0.856      0.869      0.461

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       9/50      1.44G      1.127      2.588     0.3964     0.7702      1.165         82        320: 100% ━━━━━━━━━━━━ 129/129 11.9it/s 10.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.937      0.905      0.925      0.624      0.919      0.872      0.875      0.464

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      10/50      1.44G      1.128      2.589     0.3934     0.7733      1.164         89        320: 100% ━━━━━━━━━━━━ 129/129 12.0it/s 10.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874       0.93      0.918      0.921      0.613      0.895      0.883      0.861       0.47

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      11/50      1.44G      1.128      2.589     0.3998     0.7577      1.163        108        320: 100% ━━━━━━━━━━━━ 129/129 12.0it/s 10.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.937      0.914      0.923      0.618      0.903      0.874      0.859      0.464

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      12/50      1.44G      1.114      2.533     0.3912     0.7569      1.153         94        320: 100% ━━━━━━━━━━━━ 129/129 12.3it/s 10.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.938      0.914      0.927      0.625      0.916      0.888      0.882      0.478

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      13/50      1.44G      1.126      2.555     0.3925     0.7598      1.163         97        320: 100% ━━━━━━━━━━━━ 129/129 12.2it/s 10.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.936      0.919      0.923      0.627      0.914      0.889      0.879      0.489

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      14/50      1.44G      1.113      2.549     0.3908     0.7429      1.153        105        320: 100% ━━━━━━━━━━━━ 129/129 12.2it/s 10.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 1.9s
                   all        516        874      0.935      0.914      0.918      0.627      0.919      0.879      0.865       0.48

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      15/50      1.44G      1.111      2.509     0.3937     0.7509      1.157         99        320: 100% ━━━━━━━━━━━━ 129/129 12.1it/s 10.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.937      0.921      0.929      0.632      0.922      0.876      0.876       0.48

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      16/50      1.44G      1.101      2.467     0.3887     0.7383      1.151         87        320: 100% ━━━━━━━━━━━━ 129/129 12.1it/s 10.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.7it/s 1.9s
                   all        516        874       0.93      0.923      0.935      0.634      0.925      0.881      0.883      0.489

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      17/50      1.44G      1.103      2.462     0.3896     0.7398      1.154         83        320: 100% ━━━━━━━━━━━━ 129/129 11.9it/s 10.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.935      0.915       0.92      0.629      0.913      0.871      0.867      0.482

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      18/50      1.44G      1.096       2.47      0.389     0.7319      1.147        104        320: 100% ━━━━━━━━━━━━ 129/129 12.0it/s 10.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.7it/s 1.9s
                   all        516        874      0.924      0.916      0.924      0.617      0.927      0.879      0.882      0.485

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      19/50      1.44G      1.094      2.446     0.3925     0.7332      1.146         82        320: 100% ━━━━━━━━━━━━ 129/129 12.0it/s 10.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 1.9s
                   all        516        874      0.929      0.933      0.935      0.637      0.917      0.892      0.895      0.495

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      20/50      1.44G      1.095      2.433     0.3857     0.7144      1.141         88        320: 100% ━━━━━━━━━━━━ 129/129 11.8it/s 10.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.948      0.927      0.938      0.628       0.92      0.887       0.89       0.49

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      21/50      1.44G      1.089      2.387     0.3883     0.7229      1.142         80        320: 100% ━━━━━━━━━━━━ 129/129 12.0it/s 10.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.939      0.924      0.936      0.641      0.919      0.901      0.899      0.497

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      22/50      1.44G      1.086       2.39     0.3897     0.7227      1.142         80        320: 100% ━━━━━━━━━━━━ 129/129 12.0it/s 10.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.937      0.926      0.935      0.635      0.916        0.9      0.895      0.504

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      23/50      1.44G      1.097      2.435     0.3892     0.7212      1.146         70        320: 100% ━━━━━━━━━━━━ 129/129 11.9it/s 10.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874       0.94      0.921      0.932      0.623       0.93      0.879      0.887      0.504

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      24/50      1.44G      1.084      2.351     0.3859     0.7072       1.14        105        320: 100% ━━━━━━━━━━━━ 129/129 12.1it/s 10.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.936       0.92      0.932      0.633      0.932      0.882      0.892      0.494

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      25/50      1.44G      1.074      2.372     0.3844     0.7056      1.137         98        320: 100% ━━━━━━━━━━━━ 129/129 12.2it/s 10.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.936      0.915      0.934      0.641      0.915      0.883      0.884      0.499

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      26/50      1.44G      1.079      2.357     0.3849     0.7058      1.136         84        320: 100% ━━━━━━━━━━━━ 129/129 12.0it/s 10.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 1.9s
                   all        516        874      0.939       0.92      0.932      0.639      0.939      0.877      0.896      0.505

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      27/50      1.44G      1.071      2.382     0.3828     0.7064      1.134         87        320: 100% ━━━━━━━━━━━━ 129/129 12.2it/s 10.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 1.9s
                   all        516        874      0.935       0.93      0.937      0.644      0.924        0.9        0.9      0.509

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      28/50      1.44G      1.074       2.33     0.3851     0.7046      1.132         95        320: 100% ━━━━━━━━━━━━ 129/129 11.9it/s 10.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.942      0.922      0.932      0.634      0.919      0.893      0.886      0.502

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      29/50      1.44G      1.074      2.371     0.3865     0.7151      1.134        103        320: 100% ━━━━━━━━━━━━ 129/129 12.0it/s 10.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874       0.94      0.926      0.939      0.637      0.937      0.878      0.893      0.507

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      30/50      1.44G      1.066      2.304     0.3844     0.6844       1.13         77        320: 100% ━━━━━━━━━━━━ 129/129 12.0it/s 10.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.7it/s 1.9s
                   all        516        874      0.943      0.932       0.94      0.634      0.923      0.905        0.9      0.513

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      31/50      1.44G       1.07      2.301     0.3818     0.6995      1.132         84        320: 100% ━━━━━━━━━━━━ 129/129 12.6it/s 10.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.7it/s 1.9s
                   all        516        874      0.942      0.921      0.934      0.646      0.923      0.881       0.89      0.514

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      32/50      1.44G      1.061      2.307      0.384     0.6952      1.133         93        320: 100% ━━━━━━━━━━━━ 129/129 11.8it/s 10.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.941      0.928      0.934      0.642      0.921      0.882      0.885      0.513

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      33/50      1.44G       1.06      2.256     0.3823     0.6839      1.131         81        320: 100% ━━━━━━━━━━━━ 129/129 12.3it/s 10.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.7it/s 1.9s
                   all        516        874      0.937      0.937       0.94      0.643      0.934      0.881      0.892      0.516

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      34/50      1.44G      1.056      2.276     0.3889     0.6883      1.127         77        320: 100% ━━━━━━━━━━━━ 129/129 12.2it/s 10.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.942      0.929      0.938      0.642      0.914      0.894      0.888      0.517

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      35/50      1.44G      1.055      2.261     0.3816     0.6873      1.125         82        320: 100% ━━━━━━━━━━━━ 129/129 12.1it/s 10.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.7it/s 1.9s
                   all        516        874      0.943       0.93      0.934      0.643      0.921      0.888      0.884      0.515

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      36/50      1.44G      1.053      2.258     0.3835     0.6846      1.124         78        320: 100% ━━━━━━━━━━━━ 129/129 11.9it/s 10.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 1.9s
                   all        516        874      0.931      0.931      0.936      0.642      0.924      0.891      0.894      0.514

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      37/50      1.44G      1.051      2.256     0.3831     0.6827      1.121        124        320: 100% ━━━━━━━━━━━━ 129/129 12.2it/s 10.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.939      0.928      0.933      0.641      0.929      0.881      0.889      0.517

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      38/50      1.44G      1.062      2.216     0.3841     0.6872      1.122         92        320: 100% ━━━━━━━━━━━━ 129/129 12.0it/s 10.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.941      0.936      0.943      0.646       0.92      0.902      0.896      0.515

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      39/50      1.44G      1.047      2.223     0.3806     0.6791      1.122         89        320: 100% ━━━━━━━━━━━━ 129/129 12.0it/s 10.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.7it/s 1.9s
                   all        516        874      0.941      0.927      0.935      0.643      0.919      0.891      0.891      0.517

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      40/50      1.44G      1.055      2.268     0.3853     0.6861      1.122         90        320: 100% ━━━━━━━━━━━━ 129/129 12.1it/s 10.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874       0.94      0.932      0.937      0.641      0.945      0.876      0.892      0.515
Closing dataloader mosaic
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      41/50      1.44G     0.9643      1.486     0.3663     0.5294      1.065         54        320: 100% ━━━━━━━━━━━━ 129/129 10.1it/s 12.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.938      0.925       0.94      0.644      0.919      0.888      0.893       0.52

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      42/50      1.44G      0.957      1.467     0.3645     0.5193      1.063         52        320: 100% ━━━━━━━━━━━━ 129/129 12.4it/s 10.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.7it/s 1.9s
                   all        516        874      0.933      0.931      0.937      0.641      0.937       0.88      0.892      0.519

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      43/50      1.44G     0.9514       1.46     0.3643     0.5142      1.061         54        320: 100% ━━━━━━━━━━━━ 129/129 12.5it/s 10.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.938      0.928      0.942      0.645       0.92      0.889      0.893      0.519

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      44/50      1.44G     0.9537      1.456     0.3622     0.5143      1.062         56        320: 100% ━━━━━━━━━━━━ 129/129 12.5it/s 10.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 1.9s
                   all        516        874      0.942      0.924      0.938      0.643      0.931      0.884      0.891      0.519

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      45/50      1.44G     0.9484      1.439      0.363      0.508      1.061         53        320: 100% ━━━━━━━━━━━━ 129/129 12.0it/s 10.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.942      0.926      0.939      0.645      0.933      0.883      0.895       0.52

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      46/50      1.44G     0.9497       1.44     0.3636     0.5096      1.058         57        320: 100% ━━━━━━━━━━━━ 129/129 12.4it/s 10.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 1.9s
                   all        516        874      0.944      0.922      0.938      0.647      0.933      0.886      0.895       0.52

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      47/50      1.44G     0.9532      1.437     0.3613     0.5137      1.065         51        320: 100% ━━━━━━━━━━━━ 129/129 12.4it/s 10.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.5it/s 2.0s
                   all        516        874      0.943      0.927      0.939      0.648      0.927      0.888      0.893      0.522

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      48/50      1.44G     0.9467      1.433     0.3616     0.5098      1.059         52        320: 100% ━━━━━━━━━━━━ 129/129 12.2it/s 10.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.6it/s 2.0s
                   all        516        874      0.942      0.925      0.937      0.644      0.933      0.883      0.896      0.523

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      49/50      1.44G     0.9531      1.447     0.3636     0.5143      1.065         57        320: 100% ━━━━━━━━━━━━ 129/129 12.2it/s 10.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.7it/s 1.9s
                   all        516        874      0.939      0.925      0.938      0.646      0.933      0.883      0.892       0.52

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      50/50      1.44G     0.9388      1.436     0.3628     0.5112       1.06         57        320: 100% ━━━━━━━━━━━━ 129/129 12.2it/s 10.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 4.7it/s 1.9s
                   all        516        874      0.942      0.927      0.938      0.645      0.933      0.883      0.892      0.521

50 epochs completed in 0.182 hours.
Optimizer stripped from /content/Shoes_VTO/outputs/stage_a/run_v7_merged_full_unfreeze_noflip/weights/last.pt, 6.6MB
Optimizer stripped from /content/Shoes_VTO/outputs/stage_a/run_v7_merged_full_unfreeze_noflip/weights/best.pt, 6.6MB

Validating /content/Shoes_VTO/outputs/stage_a/run_v7_merged_full_unfreeze_noflip/weights/best.pt...
Ultralytics 8.4.177 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
YOLOv8n-pose summary (fused): 81 layers, 3,209,267 parameters, 0 gradients, 8.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 2.2it/s 4.1s
                   all        516        874      0.943      0.927      0.938      0.647      0.926       0.89      0.893      0.522
             left_foot        424        425      0.944      0.916      0.927      0.643      0.921      0.873      0.875       0.51
            right_foot        447        449      0.942      0.937       0.95      0.651      0.931      0.906      0.911      0.534
Speed: 0.2ms preprocess, 2.1ms inference, 0.0ms loss, 1.7ms postprocess per image
Results saved to /content/Shoes_VTO/outputs/stage_a/run_v7_merged_full_unfreeze_noflip

===========================================================================
  Training Completed Successfully!
  Time Elapsed:    11.2 minutes
  Best Checkpoint: /content/Shoes_VTO/outputs/stage_a/run_v7_merged_full_unfreeze_noflip/weights/best.pt
===========================================================================

[*] Running final validation on best checkpoint...
Ultralytics 8.4.177 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
YOLOv8n-pose summary (fused): 81 layers, 3,209,267 parameters, 0 gradients, 8.8 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1375.7±717.9 MB/s, size: 29.1 KB)
val: Scanning /content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/valid/labels.cache... 516 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 516/516 196.8Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 33/33 7.2it/s 4.6s
                   all        516        874      0.943       0.92      0.934      0.645      0.925       0.89      0.892      0.521
             left_foot        424        425      0.946      0.908      0.921       0.64      0.923      0.874      0.875       0.51
            right_foot        447        449       0.94      0.931      0.948       0.65      0.928      0.906      0.908      0.532
Speed: 0.2ms preprocess, 2.4ms inference, 0.0ms loss, 1.3ms postprocess per image
Results saved to /content/Shoes_VTO/runs/pose/val-2
  [+] Validation Results:
      Box mAP@50:    93.43%
      Box mAP@50-95: 64.46%
      Pose mAP@50:   89.15%
      Pose mAP@50-95:52.13%

[OK] Pipeline complete.
