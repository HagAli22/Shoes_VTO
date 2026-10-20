/content/Shoes_VTO
Creating new Ultralytics Settings v0.0.8 file ✅ 
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/usage/settings.
===========================================================================
  STAGE A FOOT DETECTOR — PRODUCTION TRAINING PIPELINE                      
  Base weights: /content/Shoes_VTO/deliverables/stage_a_13kp/models/stage-a-320-13kp-v7.pt
  Dataset:      /content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/data.yaml
  Device:       0 (Tesla T4)
  Epochs:       50 (patience=15)
  Resolution:   256x256 | Batch: 32 | LR0: 0.0003
  Freeze:       0 layers
  Output:       /content/Shoes_VTO/outputs/stage_a/run_v7_256_13kp
===========================================================================
Ultralytics 8.4.182 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
engine/trainer: agnostic_nms=True, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=32, bgr=0.0, box=7.5, cache=False, cfg=None, channels_last=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, cls_remap=True, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=True, cutmix=0.0, data=/content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/data.yaml, degrees=10.0, deterministic=True, device=0, dfl=1.5, dgrad=0.5, dis=6.0, distill_model=None, dlam=1.0, dlog=1.0, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=50, erasing=0.3, exist_ok=True, fliplr=0.0, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=256, iou=0.7, kobj=1.0, line_width=None, lr0=0.0003, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=/content/Shoes_VTO/deliverables/stage_a_13kp/models/stage-a-320-13kp-v7.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=run_v7_256_13kp, nbs=64, nms=None, opset=None, optimize=False, optimizer=AdamW, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/content/Shoes_VTO/outputs/stage_a, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/content/Shoes_VTO/outputs/stage_a/run_v7_256_13kp, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=pose, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf': 100% ━━━━━━━━━━━━ 755.1KB 30.6MB/s 0.0s

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
AMP: downloading yolo26n.pt for AMP checks (one-time, not used for training)...
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'weights/yolo26n.pt': 100% ━━━━━━━━━━━━ 5.3MB 90.6MB/s 0.1s
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1070.4±431.1 MB/s, size: 70.0 KB)
train: Scanning /content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/train/labels... 1286 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 1286/1286 1.3Kit/s 1.0s
train: New cache created: /content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/train/labels.cache
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1818.9±1073.2 MB/s, size: 50.6 KB)
val: Scanning /content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/valid/labels... 160 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 160/160 1.6Kit/s 0.1s
val: New cache created: /content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/valid/labels.cache
optimizer: AdamW(lr=0.0003, momentum=0.937) with parameter groups 63 weight(decay=0.0), 73 weight(decay=0.0005), 72 bias(decay=0.0)
Plotting labels to /content/Shoes_VTO/outputs/stage_a/run_v7_256_13kp/labels.jpg... 
Using 1286 train, 160 val images for fraction=1.0 at imgsz=256
Using 2 dataloader workers
Logging results to /content/Shoes_VTO/outputs/stage_a/run_v7_256_13kp
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       1/50     0.715G      1.342        3.5     0.4623     0.9361      1.207         12        256: 100% ━━━━━━━━━━━━ 41/41 2.9it/s 14.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.0s/it 3.1s
                   all        160        279      0.865      0.794      0.834      0.445      0.822      0.689      0.693      0.273

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       2/50     0.922G      1.333      3.146     0.4466     0.8884      1.192         19        256: 100% ━━━━━━━━━━━━ 41/41 4.8it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.8it/s 1.6s
                   all        160        279       0.97      0.938       0.97      0.628      0.929      0.895      0.908      0.478

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       3/50     0.922G      1.259      2.917     0.4468     0.8471      1.167         14        256: 100% ━━━━━━━━━━━━ 41/41 4.7it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.8s
                   all        160        279      0.963      0.954      0.975       0.65      0.919      0.911      0.926      0.526

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       4/50     0.922G      1.279      2.813      0.435      0.841      1.164         12        256: 100% ━━━━━━━━━━━━ 41/41 5.0it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.8it/s 1.6s
                   all        160        279      0.962      0.957       0.98      0.636      0.942      0.928       0.94      0.534

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       5/50     0.922G      1.264      2.799     0.4336      0.822      1.157         16        256: 100% ━━━━━━━━━━━━ 41/41 4.8it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all        160        279      0.969      0.962      0.975      0.628      0.943      0.945      0.951      0.544

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       6/50     0.922G      1.238      2.676     0.4286     0.8007      1.152         12        256: 100% ━━━━━━━━━━━━ 41/41 4.8it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.8s
                   all        160        279      0.967      0.958      0.977      0.643      0.945      0.937      0.942      0.545

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       7/50     0.922G      1.256      2.708     0.4323     0.8154      1.152         17        256: 100% ━━━━━━━━━━━━ 41/41 4.7it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.6it/s 1.8s
                   all        160        279      0.962      0.943      0.975      0.647      0.939      0.921      0.945      0.544

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       8/50     0.922G      1.228      2.695     0.4404     0.8044      1.146         17        256: 100% ━━━━━━━━━━━━ 41/41 4.8it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.8s
                   all        160        279      0.962       0.96      0.978       0.65      0.956       0.94      0.957       0.55

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
       9/50     0.922G      1.241      2.708     0.4321     0.8084      1.148         13        256: 100% ━━━━━━━━━━━━ 41/41 4.7it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.9it/s 1.6s
                   all        160        279      0.959      0.941      0.967      0.637      0.945      0.914      0.927      0.532

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      10/50     0.922G      1.229      2.715      0.434     0.8015      1.132         23        256: 100% ━━━━━━━━━━━━ 41/41 4.6it/s 8.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.8it/s 1.7s
                   all        160        279      0.963      0.961       0.98      0.645      0.936      0.928      0.946      0.539

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      11/50     0.922G      1.216      2.563     0.4286      0.775      1.136         22        256: 100% ━━━━━━━━━━━━ 41/41 4.8it/s 8.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.8it/s 1.7s
                   all        160        279      0.967      0.961      0.981      0.648      0.942      0.932      0.935      0.538

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      12/50     0.924G      1.215      2.602      0.426     0.7769      1.137         18        256: 100% ━━━━━━━━━━━━ 41/41 5.0it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.7s
                   all        160        279       0.95      0.954      0.971      0.634       0.93      0.928      0.936       0.54

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      13/50     0.924G      1.208      2.596     0.4333     0.7915      1.135         17        256: 100% ━━━━━━━━━━━━ 41/41 4.8it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.8s
                   all        160        279      0.961      0.957      0.978      0.639      0.955      0.929      0.948      0.533

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      14/50     0.924G      1.215      2.603     0.4255     0.7805      1.141         12        256: 100% ━━━━━━━━━━━━ 41/41 4.6it/s 8.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.1it/s 1.4s
                   all        160        279       0.96       0.96      0.972      0.638      0.944      0.918      0.943      0.542

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      15/50     0.924G      1.215      2.542     0.4282     0.7616      1.129         11        256: 100% ━━━━━━━━━━━━ 41/41 4.7it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all        160        279      0.968      0.944      0.975      0.634      0.945      0.926       0.94      0.549

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      16/50     0.924G      1.199      2.519     0.4208     0.7738      1.129          9        256: 100% ━━━━━━━━━━━━ 41/41 4.8it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.8it/s 1.6s
                   all        160        279       0.98       0.94      0.975      0.643      0.954       0.92      0.937      0.545

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      17/50     0.924G      1.217      2.586     0.4236      0.776      1.137         27        256: 100% ━━━━━━━━━━━━ 41/41 4.7it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all        160        279      0.967      0.946      0.976       0.64       0.96      0.913      0.937      0.536

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      18/50     0.924G      1.198      2.515      0.419     0.7895      1.126         21        256: 100% ━━━━━━━━━━━━ 41/41 4.8it/s 8.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.9it/s 1.6s
                   all        160        279      0.964      0.959      0.979      0.649      0.931      0.919      0.938      0.551

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      19/50     0.924G      1.187      2.522     0.4267     0.7677      1.123         24        256: 100% ━━━━━━━━━━━━ 41/41 4.7it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.6it/s 1.9s
                   all        160        279      0.973      0.932      0.971      0.645      0.926      0.903      0.927      0.535

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      20/50     0.924G      1.186      2.555     0.4247     0.7588      1.124         15        256: 100% ━━━━━━━━━━━━ 41/41 4.7it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.3it/s 1.3s
                   all        160        279      0.976      0.939      0.972      0.655       0.93      0.909      0.913      0.542

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      21/50     0.924G       1.18      2.441     0.4236     0.7466      1.118         13        256: 100% ━━━━━━━━━━━━ 41/41 4.9it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.8s
                   all        160        279      0.976      0.968      0.978      0.656      0.955      0.928      0.942       0.55

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      22/50     0.924G      1.167      2.451     0.4242     0.7454      1.118         12        256: 100% ━━━━━━━━━━━━ 41/41 4.9it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.8s
                   all        160        279      0.976      0.951      0.983      0.657      0.948      0.921      0.939       0.55

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      23/50     0.924G      1.186      2.507     0.4176     0.7565      1.128         28        256: 100% ━━━━━━━━━━━━ 41/41 4.8it/s 8.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.8s
                   all        160        279      0.973       0.95      0.981      0.657      0.952      0.929      0.939       0.56

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      24/50     0.924G      1.162      2.484     0.4284     0.7425      1.113         13        256: 100% ━━━━━━━━━━━━ 41/41 4.8it/s 8.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.9it/s 1.5s
                   all        160        279      0.976      0.948      0.975      0.647      0.949      0.939       0.95      0.565

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      25/50     0.924G      1.169      2.422     0.4221     0.7312      1.117         20        256: 100% ━━━━━━━━━━━━ 41/41 4.6it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.8s
                   all        160        279      0.969      0.947      0.975       0.64       0.95      0.929      0.946      0.564

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      26/50     0.924G      1.175      2.423     0.4164     0.7363      1.113         13        256: 100% ━━━━━━━━━━━━ 41/41 4.8it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.9it/s 1.6s
                   all        160        279      0.979      0.942      0.975      0.645      0.955      0.923       0.94       0.56

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      27/50     0.924G       1.16      2.477     0.4297     0.7375      1.115         16        256: 100% ━━━━━━━━━━━━ 41/41 4.7it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.2it/s 1.4s
                   all        160        279      0.957      0.948      0.973      0.648      0.944      0.925      0.934      0.558

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      28/50     0.924G      1.162      2.414     0.4317     0.7476      1.114         21        256: 100% ━━━━━━━━━━━━ 41/41 4.8it/s 8.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.6it/s 1.9s
                   all        160        279      0.964      0.956      0.975      0.649      0.953      0.917      0.939      0.554

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      29/50     0.924G      1.155      2.465     0.4221     0.7338      1.115         15        256: 100% ━━━━━━━━━━━━ 41/41 4.7it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all        160        279      0.985       0.95      0.977      0.655      0.939      0.932      0.946      0.565

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      30/50     0.924G       1.16      2.396     0.4224     0.7526      1.115         19        256: 100% ━━━━━━━━━━━━ 41/41 4.7it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.8s
                   all        160        279      0.986      0.945      0.978      0.653      0.969      0.925      0.949      0.573

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      31/50     0.924G      1.163      2.442     0.4211     0.7367      1.107         22        256: 100% ━━━━━━━━━━━━ 41/41 4.7it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.9it/s 1.6s
                   all        160        279      0.989       0.95      0.979      0.654      0.962      0.925      0.944      0.566

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      32/50     0.924G      1.142      2.468     0.4267     0.7279      1.104         23        256: 100% ━━━━━━━━━━━━ 41/41 4.7it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all        160        279      0.989      0.944      0.978      0.653      0.964       0.92      0.944       0.56

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      33/50     0.924G      1.133      2.278     0.4126     0.6951      1.096         11        256: 100% ━━━━━━━━━━━━ 41/41 4.9it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.5it/s 2.0s
                   all        160        279      0.964      0.968      0.984      0.657      0.969      0.933       0.95      0.564

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      34/50     0.924G      1.132      2.365      0.418     0.7213      1.103         13        256: 100% ━━━━━━━━━━━━ 41/41 4.6it/s 8.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.1it/s 1.5s
                   all        160        279      0.972      0.968      0.984      0.659      0.955      0.928      0.942      0.566

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      35/50     0.924G      1.129      2.286      0.423     0.7132      1.103         15        256: 100% ━━━━━━━━━━━━ 41/41 4.9it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.6it/s 1.8s
                   all        160        279      0.972      0.968      0.982      0.655       0.96      0.933      0.951      0.567

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      36/50     0.924G      1.152      2.426       0.42     0.7221      1.102         14        256: 100% ━━━━━━━━━━━━ 41/41 4.8it/s 8.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.8s
                   all        160        279      0.975      0.968      0.981      0.661      0.954      0.929      0.949      0.563

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      37/50     0.924G      1.143      2.332     0.4245     0.7217      1.106         20        256: 100% ━━━━━━━━━━━━ 41/41 4.9it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.7s
                   all        160        279      0.979      0.953      0.983      0.652      0.963      0.928      0.947      0.568

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      38/50     0.924G      1.142      2.334     0.4202     0.7209      1.108         17        256: 100% ━━━━━━━━━━━━ 41/41 4.8it/s 8.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.8it/s 1.7s
                   all        160        279      0.983      0.947      0.981      0.656      0.964      0.928       0.94      0.565

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      39/50     0.924G      1.125       2.31     0.4196     0.7062      1.093         26        256: 100% ━━━━━━━━━━━━ 41/41 4.7it/s 8.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.8it/s 1.7s
                   all        160        279      0.979      0.949      0.978      0.655      0.962       0.93      0.944      0.569

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      40/50     0.924G      1.126      2.289     0.4211      0.724      1.098         19        256: 100% ━━━━━━━━━━━━ 41/41 4.7it/s 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.1it/s 1.4s
                   all        160        279      0.971       0.96      0.979      0.657      0.957      0.927      0.948       0.57
Closing dataloader mosaic
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      41/50     0.924G      1.031       1.56     0.4058      0.556      1.039         12        256: 100% ━━━━━━━━━━━━ 41/41 3.7it/s 11.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.2it/s 1.4s
                   all        160        279      0.975      0.955       0.98      0.652       0.96      0.932      0.944      0.567

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      42/50     0.924G       1.03      1.562     0.4055     0.5542      1.045         11        256: 100% ━━━━━━━━━━━━ 41/41 4.9it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.8s
                   all        160        279      0.974      0.956      0.977       0.66      0.959      0.934       0.94      0.565

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      43/50     0.924G       1.01      1.548     0.4076     0.5498      1.038         11        256: 100% ━━━━━━━━━━━━ 41/41 4.9it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.0it/s 1.5s
                   all        160        279      0.979       0.95      0.977      0.658      0.969      0.931      0.945      0.566

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      44/50     0.924G      1.015      1.504     0.4063      0.547       1.04         12        256: 100% ━━━━━━━━━━━━ 41/41 5.0it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.7s
                   all        160        279      0.974      0.956      0.977       0.66      0.965      0.932      0.945      0.571

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      45/50     0.924G      1.006      1.514     0.4063     0.5406      1.034          8        256: 100% ━━━━━━━━━━━━ 41/41 4.9it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.8it/s 1.7s
                   all        160        279      0.973      0.958       0.98      0.657      0.965      0.932      0.953      0.573

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      46/50     0.924G      1.017      1.557      0.407      0.553      1.045         11        256: 100% ━━━━━━━━━━━━ 41/41 5.2it/s 7.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.8s
                   all        160        279      0.973      0.959      0.977      0.659      0.965      0.932       0.95      0.573

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      47/50     0.924G      1.008      1.503     0.4037     0.5395      1.039         12        256: 100% ━━━━━━━━━━━━ 41/41 4.9it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.7s
                   all        160        279      0.973      0.958       0.98       0.66      0.965      0.932      0.953      0.573

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      48/50     0.924G      1.015      1.541      0.408     0.5474      1.039         11        256: 100% ━━━━━━━━━━━━ 41/41 5.3it/s 7.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.6it/s 1.8s
                   all        160        279      0.973      0.958       0.98      0.657      0.962      0.928      0.952      0.574

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      49/50     0.924G      1.008      1.523     0.4063     0.5458      1.039         11        256: 100% ━━━━━━━━━━━━ 41/41 5.0it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.2it/s 1.4s
                   all        160        279      0.973      0.959       0.98      0.658      0.962      0.928      0.952      0.575

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      50/50     0.924G     0.9897      1.519     0.4071     0.5355      1.027         10        256: 100% ━━━━━━━━━━━━ 41/41 4.9it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.7it/s 1.8s
                   all        160        279      0.974      0.959       0.98      0.656      0.961      0.928      0.952      0.572

50 epochs completed in 0.152 hours.
Optimizer stripped from /content/Shoes_VTO/outputs/stage_a/run_v7_256_13kp/weights/last.pt, 6.6MB
Optimizer stripped from /content/Shoes_VTO/outputs/stage_a/run_v7_256_13kp/weights/best.pt, 6.6MB

Validating /content/Shoes_VTO/outputs/stage_a/run_v7_256_13kp/weights/best.pt...
Ultralytics 8.4.182 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
YOLOv8n-pose summary (fused): 81 layers, 3,209,267 parameters, 0 gradients, 8.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 1.3it/s 2.4s
                   all        160        279      0.973      0.958       0.98      0.659      0.965      0.932      0.953      0.573
             left_foot        137        138      0.968      0.957      0.979      0.669      0.976      0.949      0.962       0.56
            right_foot        141        141      0.978      0.959      0.981      0.649      0.954      0.915      0.943      0.587
Speed: 0.2ms preprocess, 7.1ms inference, 0.0ms loss, 2.7ms postprocess per image
Results saved to /content/Shoes_VTO/outputs/stage_a/run_v7_256_13kp

===========================================================================
  Training Completed Successfully!
  Time Elapsed:    9.5 minutes
  Best Checkpoint: /content/Shoes_VTO/outputs/stage_a/run_v7_256_13kp/weights/best.pt
===========================================================================

[*] Running final validation on best checkpoint...
Ultralytics 8.4.182 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
YOLOv8n-pose summary (fused): 81 layers, 3,209,267 parameters, 0 gradients, 8.8 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2310.3±750.6 MB/s, size: 60.9 KB)
val: Scanning /content/Shoes_VTO/data/merged_v4_1_v4_2_raw_13kp/valid/labels.cache... 160 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 160/160 55.9Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.7it/s 2.7s
                   all        160        279      0.981      0.942       0.98      0.655      0.961      0.918       0.95      0.579
             left_foot        137        138      0.983      0.949      0.978      0.665       0.97      0.935      0.957      0.561
            right_foot        141        141      0.978      0.934      0.981      0.644      0.953      0.901      0.942      0.597
Speed: 0.4ms preprocess, 5.5ms inference, 0.0ms loss, 2.4ms postprocess per image
Results saved to /content/Shoes_VTO/runs/pose/val
  [+] Validation Results:
      Box mAP@50:    97.97%
      Box mAP@50-95: 65.46%
      Pose mAP@50:   94.96%
      Pose mAP@50-95:57.92%

[OK] Pipeline complete. 
