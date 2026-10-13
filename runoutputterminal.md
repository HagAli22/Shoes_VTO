Creating new Ultralytics Settings v0.0.8 file ✅ 
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/usage/settings.
===========================================================================
  STAGE A FOOT DETECTOR — PRODUCTION TRAINING PIPELINE                      
  Base weights: /content/Shoes_VTO/checkpoints/stage_a_13kp_adapted.pt
  Dataset:      /content/Shoes_VTO/data/shuffled_v4_13kp/data.yaml
  Device:       0 (Tesla T4)
  Epochs:       120 (patience=40)
  Resolution:   320x320 | Batch: 16 | LR0: 0.001
  Freeze:       0 layers
  Output:       /content/Shoes_VTO/outputs/stage_a/run_v4_13kp
===========================================================================
Ultralytics 8.4.176 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, channels_last=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, cls_remap=True, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=True, cutmix=0.0, data=/content/Shoes_VTO/data/shuffled_v4_13kp/data.yaml, degrees=10.0, deterministic=True, device=0, dfl=1.5, dgrad=0.5, dis=6.0, distill_model=None, dlam=1.0, dlog=1.0, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=120, erasing=0.3, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=320, iou=0.7, kobj=1.0, line_width=None, lr0=0.001, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=/content/Shoes_VTO/checkpoints/stage_a_13kp_adapted.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=run_v4_13kp, nbs=64, nms=None, opset=None, optimize=False, optimizer=AdamW, overlap_mask=True, patience=40, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/content/Shoes_VTO/outputs/stage_a, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/content/Shoes_VTO/outputs/stage_a/run_v4_13kp, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=pose, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf': 100% ━━━━━━━━━━━━ 755.1KB 13.3MB/s 0.1s

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

Transferred 364/397 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: downloading yolo26n.pt for AMP checks (one-time, not used for training)...
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'weights/yolo26n.pt': 100% ━━━━━━━━━━━━ 5.3MB 57.1MB/s 0.1s
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1968.8±906.8 MB/s, size: 133.5 KB)
train: Scanning /content/Shoes_VTO/data/shuffled_v4_13kp/train/labels... 643 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 643/643 988.1it/s 0.7s
train: New cache created: /content/Shoes_VTO/data/shuffled_v4_13kp/train/labels.cache
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2704.5±825.0 MB/s, size: 168.4 KB)
val: Scanning /content/Shoes_VTO/data/shuffled_v4_13kp/valid/labels... 79 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 79/79 942.1it/s 0.1s
val: New cache created: /content/Shoes_VTO/data/shuffled_v4_13kp/valid/labels.cache
optimizer: AdamW(lr=0.001, momentum=0.937) with parameter groups 63 weight(decay=0.0), 73 weight(decay=0.0005), 72 bias(decay=0.0)
Plotting labels to /content/Shoes_VTO/outputs/stage_a/run_v4_13kp/labels.jpg... 
Using 643 train, 79 val images for fraction=1.0 at imgsz=320
Using 8 dataloader workers
Logging results to /content/Shoes_VTO/outputs/stage_a/run_v4_13kp
Starting training for 120 epochs...

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      1/120     0.609G      1.524      9.432     0.6448      1.051      1.365          5        320: 100% ━━━━━━━━━━━━ 41/41 5.3it/s 7.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                   all         79        129      0.635      0.688      0.634       0.28   0.000131    0.00725   3.81e-06   1.14e-06

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      2/120     0.764G       1.48      8.219     0.6298      1.102      1.355          9        320: 100% ━━━━━━━━━━━━ 41/41 13.4it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.8it/s 0.4s
                   all         79        129      0.761      0.865      0.825       0.46     0.0069    0.00833    0.00011   2.21e-05

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      3/120     0.764G      1.487      7.528     0.6286      1.089      1.344          6        320: 100% ━━━━━━━━━━━━ 41/41 12.8it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.5it/s 0.4s
                   all         79        129      0.906        0.9      0.948      0.518      0.038     0.0322    0.00704   0.000807

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      4/120     0.764G      1.467      6.713     0.6268      1.094      1.348          8        320: 100% ━━━━━━━━━━━━ 41/41 13.6it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.9it/s 0.3s
                   all         79        129      0.917      0.875      0.923       0.51       0.14      0.111     0.0277    0.00395

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      5/120     0.764G      1.492      6.217     0.6167      1.072      1.356          5        320: 100% ━━━━━━━━━━━━ 41/41 14.8it/s 2.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.6it/s 0.3s
                   all         79        129      0.828       0.89      0.931      0.523      0.314      0.354      0.184     0.0287

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      6/120     0.764G      1.454      5.887     0.6176      1.086      1.345          5        320: 100% ━━━━━━━━━━━━ 41/41 12.9it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.5it/s 0.4s
                   all         79        129      0.902      0.908      0.944      0.533      0.465      0.387      0.294     0.0607

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      7/120     0.764G      1.457      5.472     0.6122      1.033      1.341          6        320: 100% ━━━━━━━━━━━━ 41/41 13.8it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.5it/s 0.4s
                   all         79        129      0.932      0.938      0.967      0.525      0.579      0.582       0.42     0.0785

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      8/120     0.764G      1.437      5.129      0.612      1.056      1.344          9        320: 100% ━━━━━━━━━━━━ 41/41 14.5it/s 2.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.4it/s 0.4s
                   all         79        129      0.903      0.928      0.947      0.499      0.529      0.562      0.386     0.0853

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      9/120     0.764G      1.429      4.978     0.6065       1.03      1.317          5        320: 100% ━━━━━━━━━━━━ 41/41 14.3it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.6it/s 0.4s
                   all         79        129      0.936      0.913      0.961      0.512      0.692      0.656      0.548      0.118

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     10/120     0.764G      1.445      4.628     0.6055      1.014       1.32          6        320: 100% ━━━━━━━━━━━━ 41/41 13.0it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.8it/s 0.4s
                   all         79        129      0.965      0.896      0.936      0.533       0.75      0.696      0.614      0.178

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     11/120     0.764G      1.406       4.41     0.5955      1.007      1.302          4        320: 100% ━━━━━━━━━━━━ 41/41 14.2it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.9it/s 0.4s
                   all         79        129      0.909      0.929      0.941       0.51      0.766      0.784      0.673      0.189

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     12/120     0.764G      1.421      4.317     0.5964       1.01      1.312          4        320: 100% ━━━━━━━━━━━━ 41/41 13.2it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.5it/s 0.4s
                   all         79        129       0.92      0.898      0.951      0.533       0.75      0.731      0.649      0.194

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     13/120     0.764G      1.399      4.249     0.5908     0.9774      1.298         11        320: 100% ━━━━━━━━━━━━ 41/41 12.6it/s 3.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         79        129      0.928      0.907      0.946      0.516      0.791      0.768      0.703      0.234

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     14/120     0.764G      1.451      4.347     0.6036      1.061      1.334          7        320: 100% ━━━━━━━━━━━━ 41/41 13.8it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.3it/s 0.4s
                   all         79        129      0.855      0.879      0.921      0.498      0.707      0.733      0.643       0.22

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     15/120     0.764G      1.447      4.034     0.5923      1.011      1.323          7        320: 100% ━━━━━━━━━━━━ 41/41 14.1it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.8it/s 0.3s
                   all         79        129      0.914      0.904      0.942      0.493      0.754      0.746      0.624      0.224

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     16/120     0.764G      1.406      4.106     0.5924      1.019       1.31          6        320: 100% ━━━━━━━━━━━━ 41/41 13.2it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.8it/s 0.4s
                   all         79        129      0.906      0.845      0.902      0.454      0.784      0.731      0.654      0.223

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     17/120     0.764G      1.438      3.972     0.5948     0.9985      1.326          7        320: 100% ━━━━━━━━━━━━ 41/41 13.1it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.5it/s 0.4s
                   all         79        129      0.897      0.886      0.927      0.537      0.777       0.76      0.737      0.264

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     18/120     0.764G      1.422      3.919      0.595      1.008      1.332         10        320: 100% ━━━━━━━━━━━━ 41/41 14.4it/s 2.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.2it/s 0.4s
                   all         79        129      0.876      0.893       0.93      0.519      0.787      0.838      0.788      0.295

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     19/120     0.764G       1.41      3.738     0.5897     0.9812      1.316          4        320: 100% ━━━━━━━━━━━━ 41/41 14.3it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.2it/s 0.4s
                   all         79        129      0.951      0.907      0.954      0.526      0.841      0.805      0.776      0.291

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     20/120     0.764G      1.387      3.667      0.588     0.9795        1.3          6        320: 100% ━━━━━━━━━━━━ 41/41 13.6it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.0it/s 0.4s
                   all         79        129      0.938      0.897      0.946      0.548      0.867       0.82      0.814      0.348

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     21/120     0.764G      1.384      3.653     0.5823     0.9466      1.289          7        320: 100% ━━━━━━━━━━━━ 41/41 13.5it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.5it/s 0.4s
                   all         79        129      0.909      0.899       0.93      0.504      0.827       0.83      0.802      0.321

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     22/120     0.764G      1.398       3.47     0.5826     0.9435      1.305          8        320: 100% ━━━━━━━━━━━━ 41/41 14.2it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.5it/s 0.4s
                   all         79        129      0.898      0.885      0.931      0.527      0.844      0.834       0.84      0.327

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     23/120     0.764G      1.386      3.716     0.5899     0.9835       1.29          7        320: 100% ━━━━━━━━━━━━ 41/41 13.9it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.3it/s 0.4s
                   all         79        129      0.904      0.907      0.934      0.527      0.859      0.849      0.827      0.345

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     24/120     0.764G       1.37      3.515     0.5815     0.9199      1.299          7        320: 100% ━━━━━━━━━━━━ 41/41 12.7it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.6it/s 0.3s
                   all         79        129      0.882      0.943      0.931       0.51      0.889      0.843      0.826      0.333

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     25/120     0.764G      1.378      3.618     0.5791     0.9366      1.287         12        320: 100% ━━━━━━━━━━━━ 41/41 14.1it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.0it/s 0.4s
                   all         79        129      0.884      0.898       0.92      0.523      0.837      0.839      0.815      0.302

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     26/120     0.764G      1.397      3.626     0.5811     0.9345      1.295          7        320: 100% ━━━━━━━━━━━━ 41/41 14.0it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.0it/s 0.4s
                   all         79        129      0.897      0.887      0.936      0.503      0.844      0.806      0.816      0.306

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     27/120     0.764G      1.375      3.454     0.5802     0.9405      1.303          4        320: 100% ━━━━━━━━━━━━ 41/41 13.5it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.4it/s 0.4s
                   all         79        129      0.952      0.934      0.965      0.544      0.914      0.885      0.901      0.349

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     28/120     0.764G      1.399        3.3     0.5709     0.9305      1.296          9        320: 100% ━━━━━━━━━━━━ 41/41 13.1it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.5it/s 0.4s
                   all         79        129      0.854      0.915      0.926       0.54      0.819      0.853      0.821      0.347

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     29/120     0.764G      1.363      3.369     0.5752     0.9247      1.277          7        320: 100% ━━━━━━━━━━━━ 41/41 13.4it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.2it/s 0.4s
                   all         79        129      0.886      0.847      0.902      0.527      0.884      0.799      0.844      0.367

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     30/120     0.764G      1.375      3.454     0.5765     0.9483      1.305         10        320: 100% ━━━━━━━━━━━━ 41/41 13.6it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.7it/s 0.4s
                   all         79        129      0.908      0.907      0.945      0.546      0.839      0.839      0.848      0.348

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     31/120     0.764G      1.381      3.339     0.5748      0.946      1.279          5        320: 100% ━━━━━━━━━━━━ 41/41 13.0it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.6it/s 0.4s
                   all         79        129      0.893      0.913      0.931      0.436      0.836       0.85      0.814      0.321

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     32/120     0.764G      1.401      3.321     0.5806     0.9389      1.305          7        320: 100% ━━━━━━━━━━━━ 41/41 13.9it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.5it/s 0.4s
                   all         79        129      0.928      0.868      0.917      0.476      0.895      0.836      0.835       0.34

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     33/120     0.764G      1.382      3.314     0.5722     0.8952      1.283          4        320: 100% ━━━━━━━━━━━━ 41/41 13.4it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.1it/s 0.4s
                   all         79        129      0.946      0.868      0.944      0.499      0.897      0.822      0.842      0.354

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     34/120     0.764G      1.354      3.461     0.5695      0.953      1.279          7        320: 100% ━━━━━━━━━━━━ 41/41 13.3it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.3it/s 0.4s
                   all         79        129      0.946      0.892      0.936      0.482      0.906      0.866      0.852      0.338

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     35/120     0.764G      1.337       3.24     0.5684     0.8992      1.277          5        320: 100% ━━━━━━━━━━━━ 41/41 13.2it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.4it/s 0.4s
                   all         79        129      0.905      0.911      0.937      0.508      0.839      0.822      0.798      0.336

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     36/120     0.764G      1.322      3.195     0.5676     0.8907      1.253         11        320: 100% ━━━━━━━━━━━━ 41/41 13.1it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.8it/s 0.4s
                   all         79        129      0.934      0.915      0.939      0.504      0.886      0.858      0.859      0.386

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     37/120     0.764G       1.34      3.292     0.5683     0.9029       1.27          9        320: 100% ━━━━━━━━━━━━ 41/41 14.1it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.5it/s 0.4s
                   all         79        129      0.949      0.907      0.942      0.504      0.877      0.841       0.82      0.387

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     38/120     0.764G      1.375      3.201      0.563      0.894      1.283          5        320: 100% ━━━━━━━━━━━━ 41/41 13.3it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         79        129       0.91      0.926      0.951      0.476      0.869      0.886       0.85      0.381

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     39/120     0.764G      1.321      3.166     0.5692     0.8914      1.258          6        320: 100% ━━━━━━━━━━━━ 41/41 13.2it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.5it/s 0.4s
                   all         79        129      0.908      0.894      0.924      0.506      0.878      0.864      0.842      0.394

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     40/120     0.764G      1.343      3.057     0.5631     0.8999      1.272          4        320: 100% ━━━━━━━━━━━━ 41/41 13.4it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.3it/s 0.4s
                   all         79        129       0.91      0.906      0.927       0.51      0.854      0.855      0.835      0.401

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     41/120     0.764G      1.348       2.98     0.5629      0.893      1.287          5        320: 100% ━━━━━━━━━━━━ 41/41 14.1it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.8it/s 0.4s
                   all         79        129      0.944      0.923      0.932       0.52      0.903       0.89      0.865      0.432

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     42/120     0.764G      1.311      3.114      0.564     0.8846      1.255          6        320: 100% ━━━━━━━━━━━━ 41/41 12.6it/s 3.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.6it/s 0.4s
                   all         79        129      0.969      0.887      0.932      0.482       0.92      0.843      0.826      0.372

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     43/120     0.764G       1.34      3.094     0.5617     0.8706      1.257         10        320: 100% ━━━━━━━━━━━━ 41/41 13.1it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.3it/s 0.4s
                   all         79        129      0.927       0.92      0.936      0.481      0.887      0.883      0.871      0.388

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     44/120     0.764G      1.341       3.13     0.5661     0.9024       1.27          8        320: 100% ━━━━━━━━━━━━ 41/41 13.8it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.2it/s 0.4s
                   all         79        129      0.917      0.876      0.941      0.537      0.881      0.849      0.895      0.412

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     45/120     0.764G      1.357      2.992     0.5602     0.9026      1.284          4        320: 100% ━━━━━━━━━━━━ 41/41 12.8it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.1it/s 0.4s
                   all         79        129      0.896      0.932      0.957      0.543      0.865      0.901      0.917      0.442

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     46/120     0.764G      1.323      2.882     0.5561     0.8649      1.252          6        320: 100% ━━━━━━━━━━━━ 41/41 13.2it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.2it/s 0.4s
                   all         79        129      0.937      0.911      0.949      0.531      0.887      0.865      0.854      0.407

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     47/120     0.764G      1.313      3.006     0.5638     0.8601      1.259          8        320: 100% ━━━━━━━━━━━━ 41/41 13.9it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.0it/s 0.4s
                   all         79        129      0.955      0.913      0.954      0.518      0.915      0.872      0.864      0.425

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     48/120     0.764G       1.33      2.881     0.5531     0.8673      1.268          4        320: 100% ━━━━━━━━━━━━ 41/41 13.9it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.6it/s 0.4s
                   all         79        129      0.949      0.893      0.957      0.501      0.918      0.834      0.871      0.413

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     49/120     0.764G      1.324      2.964     0.5601     0.8839      1.269          4        320: 100% ━━━━━━━━━━━━ 41/41 12.4it/s 3.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.1it/s 0.4s
                   all         79        129      0.935      0.893      0.953      0.515      0.918      0.885       0.92       0.45

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     50/120     0.764G      1.318      2.996     0.5521     0.8825      1.272          7        320: 100% ━━━━━━━━━━━━ 41/41 13.3it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.1it/s 0.4s
                   all         79        129      0.929      0.908      0.946      0.535      0.899      0.868      0.881      0.416

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     51/120     0.764G      1.304      2.926     0.5557     0.8629      1.246          4        320: 100% ━━━━━━━━━━━━ 41/41 13.9it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.0it/s 0.4s
                   all         79        129      0.937      0.892      0.954      0.551      0.915      0.868       0.91      0.438

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     52/120     0.764G      1.303       2.91     0.5531     0.8386      1.256          4        320: 100% ━━━━━━━━━━━━ 41/41 12.5it/s 3.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         79        129      0.948       0.92      0.952      0.541      0.907      0.881      0.882      0.438

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     53/120     0.764G      1.304      2.823     0.5506     0.8511      1.254          8        320: 100% ━━━━━━━━━━━━ 41/41 12.9it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.7it/s 0.3s
                   all         79        129      0.941      0.928      0.953       0.52      0.911      0.898      0.891      0.453

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     54/120     0.764G      1.315      2.816     0.5499     0.8794      1.243          3        320: 100% ━━━━━━━━━━━━ 41/41 14.0it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.9it/s 0.4s
                   all         79        129      0.933      0.937       0.96       0.53      0.923      0.892      0.913      0.455

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     55/120     0.764G      1.289      2.816      0.543     0.8203      1.248          4        320: 100% ━━━━━━━━━━━━ 41/41 14.1it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.5it/s 0.4s
                   all         79        129      0.935      0.911      0.947      0.527      0.898      0.877      0.875      0.421

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     56/120     0.764G      1.267      2.724     0.5461     0.8391      1.237         11        320: 100% ━━━━━━━━━━━━ 41/41 12.2it/s 3.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.7it/s 0.4s
                   all         79        129      0.936      0.908       0.95       0.54      0.921      0.915      0.914      0.453

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     57/120     0.764G      1.265        2.7     0.5475     0.8348      1.231          9        320: 100% ━━━━━━━━━━━━ 41/41 13.0it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.1it/s 0.4s
                   all         79        129      0.957      0.936      0.956      0.542      0.932      0.912      0.911      0.455

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     58/120     0.764G      1.257      2.841     0.5452     0.8266      1.238          6        320: 100% ━━━━━━━━━━━━ 41/41 13.8it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.1it/s 0.4s
                   all         79        129      0.956      0.902      0.951      0.545      0.939      0.883      0.913      0.483

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     59/120     0.764G      1.255      2.643     0.5461     0.8228      1.221          5        320: 100% ━━━━━━━━━━━━ 41/41 12.5it/s 3.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.2it/s 0.4s
                   all         79        129      0.942      0.917      0.955      0.505      0.935       0.89      0.917      0.487

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     60/120     0.764G      1.233      2.635     0.5453     0.8075      1.211          3        320: 100% ━━━━━━━━━━━━ 41/41 13.8it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.3it/s 0.4s
                   all         79        129      0.938       0.92      0.942      0.489      0.924      0.906      0.918      0.475

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     61/120     0.764G      1.271      2.803      0.549     0.8067      1.236         10        320: 100% ━━━━━━━━━━━━ 41/41 14.1it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.3it/s 0.4s
                   all         79        129      0.953      0.904      0.944      0.506      0.946      0.897      0.938      0.479

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     62/120     0.764G      1.266      2.682     0.5481     0.7995      1.224          6        320: 100% ━━━━━━━━━━━━ 41/41 13.2it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.2it/s 0.4s
                   all         79        129       0.95      0.914       0.95      0.532      0.925       0.89      0.916      0.489

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     63/120     0.764G      1.277      2.676      0.534     0.8213      1.233          9        320: 100% ━━━━━━━━━━━━ 41/41 12.1it/s 3.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.1it/s 0.4s
                   all         79        129      0.942      0.931      0.954      0.529      0.925      0.916      0.926      0.462

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     64/120     0.764G       1.29       2.65     0.5406     0.8046      1.237         11        320: 100% ━━━━━━━━━━━━ 41/41 13.4it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.3it/s 0.4s
                   all         79        129      0.983      0.921      0.952       0.55      0.951      0.908      0.931      0.475

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     65/120     0.764G      1.259      2.752     0.5403     0.8095      1.225          6        320: 100% ━━━━━━━━━━━━ 41/41 13.9it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.8it/s 0.4s
                   all         79        129      0.945      0.919      0.943      0.526      0.921      0.898      0.905      0.469

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     66/120     0.764G      1.247      2.602     0.5362     0.8287      1.222          5        320: 100% ━━━━━━━━━━━━ 41/41 12.7it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.5it/s 0.4s
                   all         79        129      0.953      0.875      0.924      0.529      0.887      0.867      0.868      0.438

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     67/120     0.764G        1.3      2.636     0.5327     0.8244      1.248          8        320: 100% ━━━━━━━━━━━━ 41/41 13.6it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.9it/s 0.4s
                   all         79        129      0.963      0.916      0.955      0.561      0.934      0.893      0.911      0.505

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     68/120     0.764G      1.245      2.668     0.5417     0.8025      1.212          9        320: 100% ━━━━━━━━━━━━ 41/41 13.8it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.9it/s 0.4s
                   all         79        129      0.947      0.905      0.951      0.568      0.931       0.89      0.923      0.482

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     69/120     0.764G      1.218      2.631      0.539     0.7839      1.211          4        320: 100% ━━━━━━━━━━━━ 41/41 13.6it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.0it/s 0.4s
                   all         79        129      0.966      0.914      0.951      0.559      0.942      0.898      0.931      0.492

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     70/120     0.764G      1.257       2.52     0.5336     0.7869      1.209          4        320: 100% ━━━━━━━━━━━━ 41/41 12.7it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.1it/s 0.4s
                   all         79        129      0.945      0.914      0.949      0.553      0.927      0.898       0.93      0.488

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     71/120     0.764G      1.219       2.55     0.5339     0.7925      1.206          5        320: 100% ━━━━━━━━━━━━ 41/41 13.6it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.7it/s 0.4s
                   all         79        129       0.95      0.921      0.948      0.547      0.934      0.906      0.928       0.49

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     72/120     0.764G      1.219      2.509     0.5346     0.7759      1.205          7        320: 100% ━━━━━━━━━━━━ 41/41 13.9it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.0it/s 0.4s
                   all         79        129      0.957       0.92      0.944      0.544      0.933      0.897      0.916      0.489

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     73/120     0.764G      1.267      2.614     0.5357     0.8335      1.244          5        320: 100% ━━━━━━━━━━━━ 41/41 13.2it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.5it/s 0.4s
                   all         79        129      0.962      0.911      0.952      0.544      0.938      0.888      0.917      0.485

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     74/120     0.764G      1.188      2.531     0.5306     0.7622      1.182          8        320: 100% ━━━━━━━━━━━━ 41/41 13.1it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.4it/s 0.4s
                   all         79        129      0.944      0.909      0.954      0.554      0.941      0.883      0.918      0.506

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     75/120     0.764G      1.232      2.414     0.5258     0.7717      1.199         11        320: 100% ━━━━━━━━━━━━ 41/41 14.4it/s 2.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.7it/s 0.4s
                   all         79        129      0.955      0.897       0.95      0.537      0.898      0.945      0.947      0.513

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     76/120     0.764G      1.254      2.488     0.5227     0.8133      1.208          7        320: 100% ━━━━━━━━━━━━ 41/41 13.1it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         79        129      0.933      0.937      0.947      0.537      0.915      0.918      0.932      0.514

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     77/120     0.764G      1.198      2.417     0.5195     0.7676      1.193          5        320: 100% ━━━━━━━━━━━━ 41/41 12.5it/s 3.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.3it/s 0.4s
                   all         79        129      0.951      0.923      0.944      0.547      0.941      0.899      0.914        0.5

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     78/120     0.764G      1.224      2.456     0.5316     0.7832      1.219          5        320: 100% ━━━━━━━━━━━━ 41/41 13.0it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.5it/s 0.4s
                   all         79        129      0.949      0.919      0.942      0.553      0.959      0.903      0.931      0.522

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     79/120     0.764G      1.224      2.447     0.5244     0.7667      1.196         14        320: 100% ━━━━━━━━━━━━ 41/41 13.8it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.3it/s 0.4s
                   all         79        129      0.918      0.922      0.949      0.585      0.909      0.875      0.886      0.504

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     80/120     0.764G      1.203      2.485     0.5194      0.769      1.187          8        320: 100% ━━━━━━━━━━━━ 41/41 12.8it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.8it/s 0.4s
                   all         79        129      0.944       0.92      0.949      0.548      0.919      0.897      0.898      0.504

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     81/120     0.764G      1.202      2.416     0.5214     0.7622      1.192          5        320: 100% ━━━━━━━━━━━━ 41/41 13.5it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.9it/s 0.4s
                   all         79        129      0.967      0.884       0.95      0.582      0.943       0.86      0.908      0.503

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     82/120     0.764G      1.176      2.333     0.5287     0.7375      1.184          5        320: 100% ━━━━━━━━━━━━ 41/41 13.8it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.9it/s 0.4s
                   all         79        129      0.941      0.915      0.946       0.57       0.94      0.914      0.927      0.515

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     83/120     0.764G      1.202      2.364     0.5282     0.7713      1.206          6        320: 100% ━━━━━━━━━━━━ 41/41 13.8it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.8it/s 0.4s
                   all         79        129      0.939      0.922      0.955       0.57      0.931      0.924       0.94      0.535

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     84/120     0.764G      1.166      2.394     0.5233     0.7485      1.185          5        320: 100% ━━━━━━━━━━━━ 41/41 12.5it/s 3.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.3it/s 0.4s
                   all         79        129      0.929      0.944      0.949      0.574      0.921      0.936      0.927      0.529

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     85/120     0.764G      1.158      2.401      0.525     0.7405      1.178          7        320: 100% ━━━━━━━━━━━━ 41/41 13.9it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.1it/s 0.4s
                   all         79        129      0.931      0.915      0.934      0.564      0.926      0.898      0.902      0.511

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     86/120     0.764G      1.178      2.329      0.521     0.7449      1.186          4        320: 100% ━━━━━━━━━━━━ 41/41 12.7it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.9it/s 0.4s
                   all         79        129      0.923       0.94      0.948      0.568       0.93      0.892      0.932      0.517

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     87/120     0.764G       1.17      2.329     0.5241      0.739       1.19          8        320: 100% ━━━━━━━━━━━━ 41/41 12.9it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         79        129      0.948      0.902      0.949      0.566      0.931      0.887      0.922      0.498

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     88/120     0.764G      1.187      2.349     0.5227     0.7503      1.183          6        320: 100% ━━━━━━━━━━━━ 41/41 13.3it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.3it/s 0.4s
                   all         79        129      0.937      0.924      0.954      0.574      0.921      0.909      0.928      0.518

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     89/120     0.764G      1.141      2.277     0.5209     0.7122      1.168          7        320: 100% ━━━━━━━━━━━━ 41/41 13.8it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.9it/s 0.4s
                   all         79        129      0.946      0.916      0.952      0.567       0.93      0.901      0.929      0.518

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     90/120     0.764G      1.166      2.319     0.5219     0.7364      1.177          4        320: 100% ━━━━━━━━━━━━ 41/41 13.6it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.8it/s 0.4s
                   all         79        129      0.946      0.905      0.949      0.563      0.914      0.903      0.913      0.512

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     91/120     0.764G      1.178      2.423     0.5184     0.7567      1.183          9        320: 100% ━━━━━━━━━━━━ 41/41 12.4it/s 3.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.0it/s 0.4s
                   all         79        129      0.931      0.937      0.946       0.56      0.921      0.928      0.918      0.525

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     92/120     0.764G       1.15      2.247     0.5221     0.7123      1.164          8        320: 100% ━━━━━━━━━━━━ 41/41 13.0it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.6it/s 0.4s
                   all         79        129      0.934      0.921      0.939      0.561      0.906      0.923      0.917      0.532

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     93/120     0.764G      1.164      2.273     0.5171     0.7344      1.173          4        320: 100% ━━━━━━━━━━━━ 41/41 13.6it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.3it/s 0.4s
                   all         79        129      0.938      0.912      0.942      0.551      0.908      0.898      0.909      0.513

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     94/120     0.764G      1.157      2.305     0.5219     0.7376      1.167          9        320: 100% ━━━━━━━━━━━━ 41/41 12.4it/s 3.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.4it/s 0.4s
                   all         79        129      0.933      0.908      0.947      0.561      0.895      0.915      0.911      0.525

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     95/120     0.764G      1.143      2.237     0.5095     0.7042      1.156          7        320: 100% ━━━━━━━━━━━━ 41/41 13.0it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.2it/s 0.4s
                   all         79        129      0.934      0.908      0.951      0.568       0.92      0.899      0.923      0.526

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     96/120     0.764G       1.15      2.219     0.5211     0.7024      1.167          5        320: 100% ━━━━━━━━━━━━ 41/41 13.3it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.4it/s 0.4s
                   all         79        129      0.946      0.908      0.953      0.565      0.928      0.904      0.924      0.523

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     97/120     0.764G      1.148      2.275     0.5148     0.7207      1.169          4        320: 100% ━━━━━━━━━━━━ 41/41 13.5it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.7it/s 0.4s
                   all         79        129      0.951      0.915      0.951      0.567      0.935      0.899      0.923      0.544

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     98/120     0.764G      1.175      2.342     0.5274     0.7379      1.183          9        320: 100% ━━━━━━━━━━━━ 41/41 13.6it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.1it/s 0.4s
                   all         79        129      0.959      0.915      0.955      0.564      0.937      0.899      0.925      0.529

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     99/120     0.764G       1.15      2.297     0.5224     0.7255      1.174          8        320: 100% ━━━━━━━━━━━━ 41/41 13.4it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.7it/s 0.3s
                   all         79        129      0.952      0.915      0.951      0.572      0.936      0.899      0.927      0.536

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    100/120     0.764G      1.141      2.252     0.5261     0.7231      1.154          4        320: 100% ━━━━━━━━━━━━ 41/41 13.3it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.3it/s 0.4s
                   all         79        129      0.956      0.915      0.952      0.577      0.908      0.933      0.925      0.541

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    101/120     0.764G      1.131      2.169     0.5142     0.7063      1.155         11        320: 100% ━━━━━━━━━━━━ 41/41 12.6it/s 3.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.1it/s 0.4s
                   all         79        129      0.945      0.915      0.952      0.565      0.906      0.933      0.924      0.541

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    102/120     0.764G      1.148      2.176     0.5213     0.7326      1.163          6        320: 100% ━━━━━━━━━━━━ 41/41 13.3it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.3it/s 0.4s
                   all         79        129      0.958      0.915      0.958      0.573       0.91      0.925      0.926      0.539

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    103/120     0.764G      1.138      2.258     0.5182      0.718      1.157          4        320: 100% ━━━━━━━━━━━━ 41/41 13.5it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.2it/s 0.4s
                   all         79        129      0.948      0.915      0.956      0.575      0.904      0.925      0.925      0.533

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    104/120     0.764G      1.146      2.146     0.5166     0.7259      1.174          5        320: 100% ━━━━━━━━━━━━ 41/41 13.4it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.3it/s 0.4s
                   all         79        129      0.942      0.915      0.953      0.571      0.925      0.899      0.923      0.528

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    105/120     0.764G      1.111      2.093     0.5164     0.6991      1.148          5        320: 100% ━━━━━━━━━━━━ 41/41 13.1it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.0it/s 0.4s
                   all         79        129      0.937      0.915      0.953      0.569      0.908      0.918      0.923      0.528

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    106/120     0.764G      1.144      2.284     0.5166     0.7235      1.162         16        320: 100% ━━━━━━━━━━━━ 41/41 14.2it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.9it/s 0.4s
                   all         79        129      0.938      0.915      0.946      0.571      0.908      0.916      0.924      0.534

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    107/120     0.764G      1.139       2.29      0.518     0.7183      1.163          5        320: 100% ━━━━━━━━━━━━ 41/41 13.6it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.5it/s 0.4s
                   all         79        129       0.94      0.915      0.943      0.568      0.922      0.899      0.924      0.539

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    108/120     0.764G      1.147      2.261     0.5272     0.7199      1.166         11        320: 100% ━━━━━━━━━━━━ 41/41 12.2it/s 3.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.1it/s 0.4s
                   all         79        129      0.939      0.915      0.948      0.569      0.922      0.899       0.92       0.54

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    109/120     0.764G      1.162      2.188      0.521     0.7046      1.171          9        320: 100% ━━━━━━━━━━━━ 41/41 13.9it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.4it/s 0.4s
                   all         79        129      0.946      0.914      0.948      0.568      0.893      0.937      0.924      0.542

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    110/120     0.764G       1.12      2.203      0.521     0.7008      1.142          8        320: 100% ━━━━━━━━━━━━ 41/41 13.5it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.4it/s 0.4s
                   all         79        129      0.945      0.914      0.949      0.569        0.9      0.937      0.925      0.544
Closing dataloader mosaic
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    111/120     0.764G      1.033      1.482      0.499     0.5475      1.111          6        320: 100% ━━━━━━━━━━━━ 41/41 8.0it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.8it/s 0.4s
                   all         79        129      0.953      0.915       0.95      0.572      0.935      0.902       0.93      0.537

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    112/120     0.764G      1.044       1.49        0.5     0.5623      1.115          5        320: 100% ━━━━━━━━━━━━ 41/41 13.2it/s 3.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.2it/s 0.4s
                   all         79        129      0.955      0.912      0.952      0.569      0.901      0.933      0.932      0.538

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    113/120     0.764G       1.03        1.5      0.497     0.5492      1.105          5        320: 100% ━━━━━━━━━━━━ 41/41 14.1it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.9it/s 0.4s
                   all         79        129      0.942      0.911      0.949      0.567      0.891      0.934      0.928       0.53

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    114/120     0.764G      1.033      1.473     0.4997     0.5467      1.107          5        320: 100% ━━━━━━━━━━━━ 41/41 13.5it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.1it/s 0.4s
                   all         79        129      0.938      0.908       0.95      0.572      0.889      0.934      0.929      0.537

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    115/120     0.764G      1.033      1.495     0.4939      0.552        1.1          6        320: 100% ━━━━━━━━━━━━ 41/41 13.0it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 7.9it/s 0.4s
                   all         79        129      0.945      0.908      0.952      0.578        0.9      0.932      0.931      0.544

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    116/120     0.764G      1.024      1.467     0.4982     0.5436      1.104          4        320: 100% ━━━━━━━━━━━━ 41/41 14.4it/s 2.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.2it/s 0.4s
                   all         79        129      0.947      0.911      0.949      0.577        0.9       0.93      0.931      0.544

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    117/120     0.764G      1.022      1.452     0.4986      0.554      1.096          4        320: 100% ━━━━━━━━━━━━ 41/41 13.9it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.3it/s 0.4s
                   all         79        129      0.951      0.908       0.95      0.577      0.907      0.927      0.932      0.542

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    118/120     0.764G      1.014      1.557     0.4918     0.5471      1.107          6        320: 100% ━━━━━━━━━━━━ 41/41 12.7it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.0it/s 0.4s
                   all         79        129      0.947      0.911      0.949      0.577        0.9      0.926      0.932      0.546

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    119/120     0.764G      1.016      1.481     0.4988     0.5312      1.096          4        320: 100% ━━━━━━━━━━━━ 41/41 13.8it/s 3.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.1it/s 0.4s
                   all         79        129      0.944      0.908      0.949      0.578        0.9      0.925      0.932      0.544

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    120/120     0.764G       1.03      1.476     0.4949     0.5431      1.099          4        320: 100% ━━━━━━━━━━━━ 41/41 14.8it/s 2.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 8.6it/s 0.3s
                   all         79        129      0.948      0.911      0.949      0.578        0.9      0.926      0.931      0.545

120 epochs completed in 0.128 hours.
Optimizer stripped from /content/Shoes_VTO/outputs/stage_a/run_v4_13kp/weights/last.pt, 6.7MB
Optimizer stripped from /content/Shoes_VTO/outputs/stage_a/run_v4_13kp/weights/best.pt, 6.7MB

Validating /content/Shoes_VTO/outputs/stage_a/run_v4_13kp/weights/best.pt...
Ultralytics 8.4.176 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
YOLOv8n-pose summary (fused): 81 layers, 3,209,267 parameters, 0 gradients, 8.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.2it/s 1.4s
                   all         79        129      0.943      0.908      0.949      0.577        0.9      0.926      0.932      0.544
             left_foot         69         69      0.961      0.899      0.955      0.547      0.943      0.951      0.949      0.525
            right_foot         60         60      0.925      0.917      0.944      0.606      0.857      0.901      0.915      0.562
Speed: 0.2ms preprocess, 7.1ms inference, 0.0ms loss, 1.7ms postprocess per image
Results saved to /content/Shoes_VTO/outputs/stage_a/run_v4_13kp

===========================================================================
  Training Completed Successfully!
  Time Elapsed:    8.1 minutes
  Best Checkpoint: /content/Shoes_VTO/outputs/stage_a/run_v4_13kp/weights/best.pt
===========================================================================

[*] Running final validation on best checkpoint...
Ultralytics 8.4.176 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
YOLOv8n-pose summary (fused): 81 layers, 3,209,267 parameters, 0 gradients, 8.8 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3308.1±1047.3 MB/s, size: 141.1 KB)
val: Scanning /content/Shoes_VTO/data/shuffled_v4_13kp/valid/labels.cache... 79 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 79/79 27.6Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 2.3it/s 2.2s
                   all         79        129      0.951      0.908       0.95      0.577      0.907      0.926      0.933      0.546
             left_foot         69         69      0.961      0.899      0.955      0.548      0.943      0.951      0.949      0.529
            right_foot         60         60      0.941      0.917      0.945      0.607      0.871      0.901      0.916      0.562
Speed: 0.6ms preprocess, 4.4ms inference, 0.0ms loss, 3.2ms postprocess per image
Results saved to /content/Shoes_VTO/runs/pose/val
  [+] Validation Results:
      Box mAP@50:    94.98%
      Box mAP@50-95: 57.73%
      Pose mAP@50:   93.26%
      Pose mAP@50-95:54.59%

[OK] Pipeline complete.
