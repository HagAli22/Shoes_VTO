===========================================================================
  STAGE A FOOT DETECTOR — PRODUCTION TRAINING PIPELINE                      
  Base weights: /content/Shoes_VTO/outputs/stage_a/run_v4_13kp/weights/best.pt
  Dataset:      /content/Shoes_VTO/data/merged_v3_v4_13kp/data.yaml
  Device:       0 (Tesla T4)
  Epochs:       100 (patience=40)
  Resolution:   320x320 | Batch: 16 | LR0: 0.0008
  Freeze:       0 layers
  Output:       /content/Shoes_VTO/outputs/stage_a/run_v5_merged_13kp
===========================================================================
Ultralytics 8.4.176 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, channels_last=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, cls_remap=True, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=True, cutmix=0.0, data=/content/Shoes_VTO/data/merged_v3_v4_13kp/data.yaml, degrees=10.0, deterministic=True, device=0, dfl=1.5, dgrad=0.5, dis=6.0, distill_model=None, dlam=1.0, dlog=1.0, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=100, erasing=0.3, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=320, iou=0.7, kobj=1.0, line_width=None, lr0=0.0008, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=/content/Shoes_VTO/outputs/stage_a/run_v4_13kp/weights/best.pt, momentum=0.937, mosaic=0.5, multi_scale=0.0, name=run_v5_merged_13kp, nbs=64, nms=None, opset=None, optimize=False, optimizer=AdamW, overlap_mask=True, patience=40, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/content/Shoes_VTO/outputs/stage_a, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/content/Shoes_VTO/outputs/stage_a/run_v5_merged_13kp, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=pose, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None

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
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1096.8±390.4 MB/s, size: 137.1 KB)
train: Scanning /content/Shoes_VTO/data/merged_v3_v4_13kp/train/labels... 1797 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 1797/1797 960.4it/s 1.9s
train: New cache created: /content/Shoes_VTO/data/merged_v3_v4_13kp/train/labels.cache
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2157.7±804.3 MB/s, size: 113.9 KB)
val: Scanning /content/Shoes_VTO/data/merged_v3_v4_13kp/valid/labels... 224 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 224/224 998.4it/s 0.2s
val: New cache created: /content/Shoes_VTO/data/merged_v3_v4_13kp/valid/labels.cache
optimizer: AdamW(lr=0.0008, momentum=0.937) with parameter groups 63 weight(decay=0.0), 73 weight(decay=0.0005), 72 bias(decay=0.0)
Plotting labels to /content/Shoes_VTO/outputs/stage_a/run_v5_merged_13kp/labels.jpg... 
Using 1797 train, 224 val images for fraction=1.0 at imgsz=320
Using 8 dataloader workers
Logging results to /content/Shoes_VTO/outputs/stage_a/run_v5_merged_13kp
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      1/100     0.623G      1.692        5.2     0.5668       1.58      1.486         11        320: 100% ━━━━━━━━━━━━ 113/113 10.1it/s 11.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 3.1it/s 2.2s
                   all        224        390      0.608      0.719      0.717      0.407      0.641      0.514      0.531      0.229

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      2/100     0.783G      1.585      4.539     0.5463      1.381      1.417         19        320: 100% ━━━━━━━━━━━━ 113/113 13.7it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.2it/s 1.0s
                   all        224        390      0.683      0.743      0.784      0.433      0.649      0.592      0.603      0.278

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      3/100     0.783G      1.579      4.475     0.5422      1.371      1.417         12        320: 100% ━━━━━━━━━━━━ 113/113 14.6it/s 7.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 0.9s
                   all        224        390      0.664      0.787      0.786      0.444      0.609      0.615        0.6      0.247

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      4/100     0.783G      1.556      4.238     0.5355      1.326      1.398         12        320: 100% ━━━━━━━━━━━━ 113/113 13.4it/s 8.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.1it/s 1.0s
                   all        224        390        0.7      0.752      0.784      0.413      0.682      0.644      0.638      0.273

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      5/100     0.783G      1.536      4.145     0.5374      1.313       1.39         14        320: 100% ━━━━━━━━━━━━ 113/113 13.7it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390      0.772      0.767      0.835       0.45      0.713      0.661      0.679      0.287

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      6/100     0.783G      1.537      4.141     0.5319      1.299       1.39         15        320: 100% ━━━━━━━━━━━━ 113/113 14.0it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.9it/s 1.0s
                   all        224        390      0.752      0.696      0.802      0.416      0.699      0.601       0.63      0.265

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      7/100     0.783G      1.527      4.067     0.5281      1.266      1.379         10        320: 100% ━━━━━━━━━━━━ 113/113 14.3it/s 7.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390      0.723      0.767      0.808      0.432      0.653      0.693      0.653      0.257

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      8/100     0.783G      1.529      3.938     0.5209      1.249      1.379         14        320: 100% ━━━━━━━━━━━━ 113/113 13.6it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.1it/s 1.0s
                   all        224        390      0.726      0.795       0.84      0.468      0.689      0.685       0.69      0.304

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      9/100     0.783G       1.51       3.98     0.5216      1.231      1.359          8        320: 100% ━━━━━━━━━━━━ 113/113 14.1it/s 8.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.5it/s 1.1s
                   all        224        390      0.772      0.757      0.821      0.471      0.753      0.717      0.722        0.3

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     10/100     0.783G      1.511       3.83     0.5104      1.239      1.372          9        320: 100% ━━━━━━━━━━━━ 113/113 14.2it/s 8.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.5it/s 0.9s
                   all        224        390      0.817      0.729      0.843      0.457      0.789      0.647      0.721      0.326

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     11/100     0.783G      1.506       3.89     0.5131      1.234      1.362         12        320: 100% ━━━━━━━━━━━━ 113/113 13.7it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.2it/s 1.0s
                   all        224        390      0.825      0.761       0.85      0.477      0.785      0.717      0.744       0.32

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     12/100     0.783G      1.501      3.776       0.51      1.223      1.358         12        320: 100% ━━━━━━━━━━━━ 113/113 14.0it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.9it/s 1.0s
                   all        224        390      0.784      0.794      0.858      0.442      0.757      0.701      0.719      0.296

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     13/100     0.783G      1.489      3.684     0.5057      1.184      1.354          9        320: 100% ━━━━━━━━━━━━ 113/113 14.8it/s 7.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390       0.76      0.837      0.864      0.479      0.735       0.71      0.748      0.322

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     14/100     0.783G      1.486      3.744     0.5014      1.179      1.353          9        320: 100% ━━━━━━━━━━━━ 113/113 13.5it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390      0.771      0.774      0.851      0.477      0.795      0.654      0.724       0.33

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     15/100     0.783G      1.466      3.636     0.4992      1.159       1.34         13        320: 100% ━━━━━━━━━━━━ 113/113 13.7it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 0.9s
                   all        224        390      0.779      0.812      0.867      0.484      0.741      0.732      0.754      0.336

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     16/100     0.783G      1.458       3.67     0.4994       1.19      1.349         14        320: 100% ━━━━━━━━━━━━ 113/113 14.2it/s 8.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.0it/s 1.0s
                   all        224        390       0.72      0.814      0.834      0.425      0.767      0.675      0.735      0.318

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     17/100     0.783G      1.472      3.643     0.4922       1.17      1.343         20        320: 100% ━━━━━━━━━━━━ 113/113 14.4it/s 7.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 1.0s
                   all        224        390      0.786      0.798      0.859      0.475      0.749      0.716      0.745      0.339

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     18/100     0.783G      1.471      3.612     0.4946      1.179      1.346         12        320: 100% ━━━━━━━━━━━━ 113/113 13.9it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390      0.784      0.782      0.869      0.482      0.756      0.704      0.739      0.331

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     19/100     0.783G       1.46      3.618     0.4932       1.15      1.331         15        320: 100% ━━━━━━━━━━━━ 113/113 13.7it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.8it/s 1.0s
                   all        224        390      0.798      0.774      0.867      0.504      0.843      0.653      0.755      0.343

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     20/100     0.783G      1.434       3.47     0.4868      1.139       1.33          7        320: 100% ━━━━━━━━━━━━ 113/113 14.1it/s 8.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.9it/s 1.0s
                   all        224        390      0.841      0.757       0.88      0.476       0.82      0.715      0.775      0.381

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     21/100     0.783G      1.439      3.498     0.4878      1.113      1.329         14        320: 100% ━━━━━━━━━━━━ 113/113 13.8it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.0it/s 1.0s
                   all        224        390      0.832      0.789       0.89      0.478      0.831      0.741      0.784       0.36

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     22/100     0.783G      1.428      3.421     0.4859      1.101      1.317         12        320: 100% ━━━━━━━━━━━━ 113/113 13.9it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.2it/s 1.0s
                   all        224        390      0.829      0.764      0.877      0.483      0.783      0.698      0.777      0.352

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     23/100     0.783G      1.437      3.381     0.4793      1.099      1.326         21        320: 100% ━━━━━━━━━━━━ 113/113 14.2it/s 8.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.0it/s 1.0s
                   all        224        390      0.823      0.764      0.872      0.497       0.83      0.721      0.785      0.377

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     24/100     0.783G      1.435      3.451     0.4825       1.12      1.325         13        320: 100% ━━━━━━━━━━━━ 113/113 13.9it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 0.9s
                   all        224        390      0.729      0.807      0.869      0.496      0.832      0.663      0.756      0.366

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     25/100     0.783G      1.426      3.347     0.4759       1.08      1.312         13        320: 100% ━━━━━━━━━━━━ 113/113 13.9it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390      0.823      0.833      0.898      0.509      0.811      0.735      0.797      0.373

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     26/100     0.783G       1.42      3.362     0.4768      1.083      1.314         11        320: 100% ━━━━━━━━━━━━ 113/113 14.1it/s 8.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.5it/s 1.1s
                   all        224        390      0.828       0.81      0.901      0.528      0.832       0.74        0.8      0.389

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     27/100     0.783G      1.439      3.341     0.4761      1.069      1.311         17        320: 100% ━━━━━━━━━━━━ 113/113 14.1it/s 8.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.5it/s 0.9s
                   all        224        390      0.879       0.75      0.898      0.521      0.848      0.697      0.777      0.385

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     28/100     0.783G      1.407      3.289     0.4776      1.081      1.303         16        320: 100% ━━━━━━━━━━━━ 113/113 13.5it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.5it/s 0.9s
                   all        224        390      0.815       0.83      0.901       0.51      0.842      0.713        0.8      0.395

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     29/100     0.783G      1.408      3.248     0.4747      1.058      1.303          8        320: 100% ━━━━━━━━━━━━ 113/113 13.6it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.1it/s 1.0s
                   all        224        390      0.831      0.802      0.895      0.528      0.838      0.761      0.806      0.405

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     30/100     0.783G      1.411      3.321     0.4697      1.083      1.314         11        320: 100% ━━━━━━━━━━━━ 113/113 14.1it/s 8.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.7it/s 1.0s
                   all        224        390      0.806       0.79      0.899      0.518      0.838      0.703      0.811      0.412

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     31/100     0.783G      1.409      3.317     0.4762      1.068      1.311          8        320: 100% ━━━━━━━━━━━━ 113/113 14.0it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.5it/s 0.9s
                   all        224        390      0.804      0.821      0.882      0.521      0.773      0.753      0.783      0.396

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     32/100     0.783G      1.404      3.292     0.4726      1.058      1.302          9        320: 100% ━━━━━━━━━━━━ 113/113 13.8it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 0.9s
                   all        224        390      0.787       0.81      0.885      0.527      0.814      0.692      0.788      0.402

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     33/100     0.783G      1.403      3.252     0.4697      1.055      1.298         18        320: 100% ━━━━━━━━━━━━ 113/113 13.5it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.5it/s 1.1s
                   all        224        390      0.802       0.81      0.893      0.515      0.774      0.725      0.779      0.414

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     34/100     0.783G      1.388      3.149     0.4666      1.045      1.292         18        320: 100% ━━━━━━━━━━━━ 113/113 14.4it/s 7.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390      0.844      0.792      0.899      0.517      0.852      0.705      0.796      0.377

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     35/100     0.783G      1.404      3.155     0.4669      1.044      1.304         12        320: 100% ━━━━━━━━━━━━ 113/113 13.7it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 0.9s
                   all        224        390      0.798      0.825      0.894      0.505      0.819      0.725       0.79      0.406

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     36/100     0.783G      1.395      3.115     0.4712      1.051      1.298         12        320: 100% ━━━━━━━━━━━━ 113/113 13.6it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.0it/s 1.0s
                   all        224        390      0.875      0.779      0.897      0.512      0.852      0.723      0.809      0.404

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     37/100     0.783G      1.377      3.142      0.467      1.029      1.288         11        320: 100% ━━━━━━━━━━━━ 113/113 14.0it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.8it/s 1.0s
                   all        224        390      0.842      0.795      0.898      0.506      0.831      0.736      0.815      0.421

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     38/100     0.783G       1.38      3.014     0.4635      1.029       1.29         12        320: 100% ━━━━━━━━━━━━ 113/113 13.8it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390      0.843      0.787      0.892      0.528      0.848      0.715      0.819      0.435

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     39/100     0.783G      1.375      3.036     0.4629       1.03      1.291         17        320: 100% ━━━━━━━━━━━━ 113/113 13.6it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.6it/s 0.9s
                   all        224        390      0.843      0.833        0.9      0.532      0.827      0.751      0.818      0.426

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     40/100     0.783G      1.377       3.06     0.4658      1.011      1.285         16        320: 100% ━━━━━━━━━━━━ 113/113 13.7it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.1it/s 1.0s
                   all        224        390      0.777      0.835      0.894      0.502      0.836      0.723        0.8      0.419

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     41/100     0.783G       1.38      3.055     0.4588      1.033      1.288         21        320: 100% ━━━━━━━━━━━━ 113/113 14.0it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390      0.874      0.772      0.909      0.537      0.854      0.733      0.813      0.415

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     42/100     0.783G      1.366      3.003     0.4586      1.003      1.277         16        320: 100% ━━━━━━━━━━━━ 113/113 13.4it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390      0.823      0.833      0.904      0.501      0.809      0.747      0.795      0.417

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     43/100     0.783G      1.363      3.064     0.4631      1.015      1.275         16        320: 100% ━━━━━━━━━━━━ 113/113 13.4it/s 8.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 0.9s
                   all        224        390      0.854      0.792       0.91      0.516      0.814      0.753      0.826      0.428

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     44/100     0.783G      1.357      3.049     0.4649      1.002      1.272         13        320: 100% ━━━━━━━━━━━━ 113/113 13.8it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.0it/s 1.0s
                   all        224        390      0.802      0.831      0.902      0.535      0.842      0.738      0.809      0.421

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     45/100     0.783G       1.36       2.98     0.4598     0.9821      1.275         14        320: 100% ━━━━━━━━━━━━ 113/113 14.6it/s 7.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390      0.822      0.828      0.905      0.533      0.862      0.718      0.821      0.424

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     46/100     0.783G      1.348          3     0.4583      1.002      1.276         15        320: 100% ━━━━━━━━━━━━ 113/113 13.6it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.1it/s 1.0s
                   all        224        390      0.903      0.777      0.915       0.55      0.891      0.733      0.839      0.427

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     47/100     0.783G      1.361      2.971      0.456     0.9873      1.272         19        320: 100% ━━━━━━━━━━━━ 113/113 13.5it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.0it/s 1.0s
                   all        224        390      0.851      0.795      0.899      0.533       0.84      0.746       0.83      0.444

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     48/100     0.783G      1.348      2.882     0.4513     0.9764      1.274         11        320: 100% ━━━━━━━━━━━━ 113/113 14.4it/s 7.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.5it/s 0.9s
                   all        224        390      0.893      0.773       0.92      0.528      0.878      0.749      0.838      0.457

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     49/100     0.783G      1.357      2.899     0.4524     0.9727       1.27         11        320: 100% ━━━━━━━━━━━━ 113/113 14.1it/s 8.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.2it/s 1.0s
                   all        224        390      0.848      0.798       0.91      0.547      0.834      0.777      0.827      0.443

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     50/100     0.783G      1.349      2.912     0.4552     0.9722      1.274         13        320: 100% ━━━━━━━━━━━━ 113/113 13.7it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390      0.854      0.815      0.917       0.54      0.857      0.761      0.829      0.459

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     51/100     0.783G      1.334       2.81     0.4518     0.9547      1.248         13        320: 100% ━━━━━━━━━━━━ 113/113 14.2it/s 7.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.9it/s 1.0s
                   all        224        390      0.855      0.797      0.897      0.534      0.841      0.764       0.81      0.452

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     52/100     0.783G      1.334      2.865     0.4532     0.9622      1.271         19        320: 100% ━━━━━━━━━━━━ 113/113 13.6it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390       0.83      0.843      0.914       0.54       0.88      0.744      0.836       0.47

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     53/100     0.783G      1.329      2.841     0.4563     0.9561      1.254         21        320: 100% ━━━━━━━━━━━━ 113/113 13.6it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.2it/s 1.0s
                   all        224        390       0.85      0.805      0.906      0.544      0.848      0.759      0.832      0.442

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     54/100     0.783G      1.336      2.764     0.4473      0.958      1.266          8        320: 100% ━━━━━━━━━━━━ 113/113 13.5it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.1it/s 1.0s
                   all        224        390      0.844      0.807      0.914      0.546      0.882      0.753      0.847      0.468

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     55/100     0.783G      1.312      2.833     0.4471     0.9485      1.248         16        320: 100% ━━━━━━━━━━━━ 113/113 14.5it/s 7.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 0.9s
                   all        224        390      0.841      0.831      0.911      0.543      0.822      0.784      0.834      0.461

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     56/100     0.783G       1.34      2.819     0.4542     0.9672      1.268         10        320: 100% ━━━━━━━━━━━━ 113/113 14.0it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 0.9s
                   all        224        390      0.831      0.812      0.906      0.532      0.831      0.772       0.84      0.476

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     57/100     0.783G      1.309       2.73     0.4516     0.9393      1.248         13        320: 100% ━━━━━━━━━━━━ 113/113 13.4it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 1.0s
                   all        224        390      0.861      0.776      0.908      0.549      0.854      0.759      0.825      0.474

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     58/100     0.783G       1.32      2.786     0.4493     0.9499      1.256          9        320: 100% ━━━━━━━━━━━━ 113/113 14.2it/s 8.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.1it/s 1.0s
                   all        224        390      0.879      0.796      0.913      0.552      0.872      0.753      0.832      0.469

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     59/100     0.783G      1.296      2.788     0.4492     0.9405      1.248          8        320: 100% ━━━━━━━━━━━━ 113/113 13.9it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390      0.871        0.8      0.915      0.551      0.832      0.764      0.839      0.488

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     60/100     0.783G      1.314      2.782     0.4473     0.9311      1.253         15        320: 100% ━━━━━━━━━━━━ 113/113 13.2it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390      0.852      0.823      0.915      0.541      0.827      0.769       0.83      0.475

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     61/100     0.783G      1.289       2.67     0.4437     0.8988      1.235         13        320: 100% ━━━━━━━━━━━━ 113/113 13.5it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390      0.822      0.828      0.917      0.544      0.834      0.754      0.828      0.478

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     62/100     0.783G      1.306      2.788     0.4479     0.9341      1.237         14        320: 100% ━━━━━━━━━━━━ 113/113 14.4it/s 7.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 1.0s
                   all        224        390      0.881      0.787      0.915      0.554      0.861       0.76      0.831      0.481

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     63/100     0.783G      1.287      2.691     0.4485     0.9138       1.24         15        320: 100% ━━━━━━━━━━━━ 113/113 13.3it/s 8.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.1it/s 1.0s
                   all        224        390      0.865      0.815      0.921      0.554      0.845      0.778       0.84      0.489

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     64/100     0.783G      1.299      2.718      0.445     0.9194      1.235          9        320: 100% ━━━━━━━━━━━━ 113/113 13.7it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.1it/s 1.0s
                   all        224        390      0.902       0.79      0.923      0.551      0.887      0.774      0.849      0.493

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     65/100     0.783G      1.283      2.677     0.4473     0.9109      1.239         11        320: 100% ━━━━━━━━━━━━ 113/113 14.2it/s 8.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.1it/s 1.0s
                   all        224        390      0.856      0.815      0.921      0.549      0.857      0.761      0.835      0.486

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     66/100     0.783G      1.295      2.679      0.444     0.9193      1.247         14        320: 100% ━━━━━━━━━━━━ 113/113 14.2it/s 8.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.0it/s 1.0s
                   all        224        390      0.888       0.79      0.919      0.558       0.88      0.767      0.846      0.492

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     67/100     0.783G      1.278      2.623     0.4426     0.8998      1.238         13        320: 100% ━━━━━━━━━━━━ 113/113 13.7it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.5it/s 0.9s
                   all        224        390      0.871      0.807      0.921      0.558      0.865      0.777      0.852      0.499

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     68/100     0.783G      1.285      2.613     0.4434     0.8942      1.233         15        320: 100% ━━━━━━━━━━━━ 113/113 13.4it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.0it/s 1.0s
                   all        224        390      0.872      0.808      0.922      0.557      0.872      0.777      0.851      0.492

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     69/100     0.783G      1.273      2.584     0.4415     0.8876      1.231         13        320: 100% ━━━━━━━━━━━━ 113/113 14.1it/s 8.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.8it/s 1.0s
                   all        224        390      0.879      0.812      0.924      0.564      0.858      0.797      0.863      0.495

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     70/100     0.783G      1.274      2.656     0.4503     0.9045      1.228         14        320: 100% ━━━━━━━━━━━━ 113/113 13.8it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 0.9s
                   all        224        390       0.87      0.815      0.923      0.563      0.844      0.786      0.845      0.501

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     71/100     0.783G      1.273      2.543     0.4399      0.877      1.229          8        320: 100% ━━━━━━━━━━━━ 113/113 14.0it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.6it/s 0.9s
                   all        224        390      0.877      0.812      0.919      0.561      0.858      0.786      0.856       0.51

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     72/100     0.783G      1.255      2.571     0.4397     0.8839      1.218         17        320: 100% ━━━━━━━━━━━━ 113/113 13.9it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.6it/s 1.1s
                   all        224        390       0.88      0.815      0.926      0.564      0.857       0.79      0.844      0.504

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     73/100     0.783G      1.275      2.555     0.4422     0.8833      1.227         18        320: 100% ━━━━━━━━━━━━ 113/113 14.2it/s 8.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390      0.874       0.81      0.925      0.558      0.857      0.791      0.856       0.51

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     74/100     0.783G      1.285      2.543     0.4423     0.8932      1.238         15        320: 100% ━━━━━━━━━━━━ 113/113 14.0it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 0.9s
                   all        224        390      0.882      0.794      0.924      0.556      0.853       0.78      0.849      0.506

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     75/100     0.783G      1.244      2.584     0.4446     0.8789      1.214         15        320: 100% ━━━━━━━━━━━━ 113/113 13.6it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.6it/s 0.9s
                   all        224        390      0.859      0.832      0.928      0.567      0.848      0.784      0.856      0.511

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     76/100     0.783G      1.254      2.543     0.4452     0.8776      1.217         16        320: 100% ━━━━━━━━━━━━ 113/113 14.0it/s 8.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.1it/s 1.0s
                   all        224        390       0.81      0.873      0.928      0.565      0.863      0.767      0.849      0.507

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     77/100     0.783G      1.257       2.59     0.4388     0.8731      1.218         13        320: 100% ━━━━━━━━━━━━ 113/113 13.9it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.2it/s 1.0s
                   all        224        390      0.875      0.818      0.928      0.568      0.857      0.797      0.859      0.516

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     78/100     0.783G      1.255       2.52       0.44     0.8702      1.218         10        320: 100% ━━━━━━━━━━━━ 113/113 13.6it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 1.0s
                   all        224        390       0.89      0.802      0.926      0.576      0.881      0.779      0.859      0.514

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     79/100     0.783G      1.258      2.515     0.4431     0.8716      1.221          7        320: 100% ━━━━━━━━━━━━ 113/113 14.1it/s 8.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.8it/s 1.0s
                   all        224        390      0.887      0.797      0.919      0.566      0.866      0.774      0.847      0.502

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     80/100     0.783G      1.246      2.501     0.4369     0.8694      1.214         13        320: 100% ━━━━━━━━━━━━ 113/113 14.2it/s 7.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 0.9s
                   all        224        390      0.845      0.828      0.921      0.561      0.874      0.763       0.85      0.507

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     81/100     0.783G       1.25      2.468     0.4411     0.8681      1.218         16        320: 100% ━━━━━━━━━━━━ 113/113 13.7it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390      0.879        0.8      0.921       0.56      0.878      0.766      0.857      0.514

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     82/100     0.783G      1.253      2.496     0.4399      0.863      1.221         13        320: 100% ━━━━━━━━━━━━ 113/113 13.9it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.6it/s 1.1s
                   all        224        390      0.879      0.805      0.925      0.561       0.88      0.758      0.854      0.517

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     83/100     0.783G      1.241      2.495     0.4385     0.8679      1.212         12        320: 100% ━━━━━━━━━━━━ 113/113 14.4it/s 7.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 0.9s
                   all        224        390      0.866      0.805      0.924      0.569      0.886      0.753      0.855      0.517

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     84/100     0.783G      1.228      2.421     0.4377     0.8522      1.202          8        320: 100% ━━━━━━━━━━━━ 113/113 13.5it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 0.9s
                   all        224        390      0.897      0.787      0.922      0.572      0.881      0.769      0.861      0.519

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     85/100     0.783G      1.258      2.556     0.4392     0.8725      1.215         17        320: 100% ━━━━━━━━━━━━ 113/113 13.4it/s 8.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.9it/s 1.0s
                   all        224        390      0.898      0.785      0.924       0.57      0.891      0.764      0.858      0.516

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     86/100     0.783G      1.251      2.532     0.4393     0.8734      1.219         15        320: 100% ━━━━━━━━━━━━ 113/113 13.8it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.0it/s 1.0s
                   all        224        390      0.916      0.786      0.925      0.571      0.895      0.765      0.857      0.517

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     87/100     0.783G      1.241      2.408     0.4373     0.8494      1.211         14        320: 100% ━━━━━━━━━━━━ 113/113 13.8it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.6it/s 0.9s
                   all        224        390        0.9      0.789      0.925      0.572      0.888      0.769      0.856      0.519

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     88/100     0.783G      1.234      2.458     0.4358     0.8486      1.206         16        320: 100% ━━━━━━━━━━━━ 113/113 13.4it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 0.9s
                   all        224        390      0.897       0.79      0.926      0.571      0.895      0.769      0.863      0.522

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     89/100     0.783G      1.254      2.521     0.4427     0.8721      1.214         16        320: 100% ━━━━━━━━━━━━ 113/113 13.8it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.9it/s 1.0s
                   all        224        390      0.902      0.791      0.927      0.576      0.897      0.779      0.856      0.523

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     90/100     0.783G      1.232      2.413     0.4378     0.8418      1.206          8        320: 100% ━━━━━━━━━━━━ 113/113 13.7it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.0it/s 1.0s
                   all        224        390        0.9      0.792      0.926      0.572      0.896      0.766      0.859      0.519
Closing dataloader mosaic
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     91/100     0.783G      1.149      1.775     0.4272     0.6917       1.16          7        320: 100% ━━━━━━━━━━━━ 113/113 11.0it/s 10.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.1it/s 1.0s
                   all        224        390      0.859      0.824      0.928       0.57      0.883      0.767      0.855      0.512

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     92/100     0.783G      1.133      1.753     0.4283     0.6777      1.156         10        320: 100% ━━━━━━━━━━━━ 113/113 13.1it/s 8.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.4it/s 0.9s
                   all        224        390       0.86      0.841      0.929      0.568      0.889      0.772      0.858      0.518

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     93/100     0.783G      1.144      1.755     0.4282     0.6795       1.16         10        320: 100% ━━━━━━━━━━━━ 113/113 13.7it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.8it/s 1.0s
                   all        224        390      0.848      0.841      0.929      0.573      0.895      0.763      0.859      0.519

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     94/100     0.783G      1.147       1.74     0.4288     0.6734      1.153         10        320: 100% ━━━━━━━━━━━━ 113/113 13.9it/s 8.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.3it/s 1.0s
                   all        224        390       0.87      0.821      0.928       0.57       0.89      0.771      0.852      0.514

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     95/100     0.783G      1.155      1.766     0.4297     0.6852      1.167         10        320: 100% ━━━━━━━━━━━━ 113/113 13.6it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.2it/s 1.0s
                   all        224        390      0.877      0.821      0.929      0.573      0.888      0.769      0.858      0.519

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     96/100     0.783G      1.136       1.73     0.4271     0.6723      1.155          9        320: 100% ━━━━━━━━━━━━ 113/113 13.6it/s 8.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.2it/s 1.0s
                   all        224        390       0.88      0.813      0.929      0.575      0.889       0.77      0.851       0.52

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     97/100     0.783G      1.136       1.71     0.4244     0.6719      1.161          8        320: 100% ━━━━━━━━━━━━ 113/113 14.5it/s 7.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.0it/s 1.0s
                   all        224        390      0.866      0.815      0.928       0.57      0.887       0.77      0.851      0.519

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     98/100     0.783G      1.141      1.717     0.4249     0.6716       1.16          8        320: 100% ━━━━━━━━━━━━ 113/113 13.5it/s 8.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.1it/s 1.0s
                   all        224        390      0.862       0.82      0.927      0.568      0.891      0.772      0.853      0.519

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     99/100     0.783G      1.126      1.706      0.424     0.6723      1.152          8        320: 100% ━━━━━━━━━━━━ 113/113 13.7it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 7.2it/s 1.0s
                   all        224        390      0.864      0.811      0.928      0.573      0.893      0.773      0.855      0.522

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    100/100     0.783G      1.147      1.707     0.4264     0.6674       1.16          8        320: 100% ━━━━━━━━━━━━ 113/113 13.9it/s 8.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 6.8it/s 1.0s
                   all        224        390      0.869      0.815      0.927      0.572      0.892      0.766      0.856      0.523

100 epochs completed in 0.265 hours.
Optimizer stripped from /content/Shoes_VTO/outputs/stage_a/run_v5_merged_13kp/weights/last.pt, 6.6MB
Optimizer stripped from /content/Shoes_VTO/outputs/stage_a/run_v5_merged_13kp/weights/best.pt, 6.6MB

Validating /content/Shoes_VTO/outputs/stage_a/run_v5_merged_13kp/weights/best.pt...
Ultralytics 8.4.176 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
YOLOv8n-pose summary (fused): 81 layers, 3,209,267 parameters, 0 gradients, 8.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 2.6it/s 2.7s
                   all        224        390      0.903      0.792      0.927      0.575      0.902      0.766      0.855      0.522
             left_foot        191        191      0.908      0.779      0.924      0.559      0.893      0.738      0.839      0.499
            right_foot        199        199      0.897      0.804      0.929      0.592       0.91      0.794      0.872      0.545
Speed: 0.1ms preprocess, 4.3ms inference, 0.0ms loss, 1.5ms postprocess per image
Results saved to /content/Shoes_VTO/outputs/stage_a/run_v5_merged_13kp

===========================================================================
  Training Completed Successfully!
  Time Elapsed:    16.2 minutes
  Best Checkpoint: /content/Shoes_VTO/outputs/stage_a/run_v5_merged_13kp/weights/best.pt
===========================================================================

[*] Running final validation on best checkpoint...
Ultralytics 8.4.176 🚀 Python-3.13.15 torch-2.11.0+cu130 CUDA:0 (Tesla T4, 14913MiB)
YOLOv8n-pose summary (fused): 81 layers, 3,209,267 parameters, 0 gradients, 8.8 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3749.0±1018.7 MB/s, size: 188.0 KB)
val: Scanning /content/Shoes_VTO/data/merged_v3_v4_13kp/valid/labels.cache... 224 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 224/224 94.0Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.4it/s 3.2s
                   all        224        390      0.901       0.79      0.927      0.574      0.903      0.771      0.856      0.524
             left_foot        191        191      0.908      0.777      0.925      0.557      0.899      0.748      0.841      0.502
            right_foot        199        199      0.893      0.804      0.929      0.592      0.907      0.794      0.872      0.546
Speed: 0.3ms preprocess, 4.0ms inference, 0.0ms loss, 2.5ms postprocess per image
Results saved to /content/Shoes_VTO/runs/pose/val-2
  [+] Validation Results:
      Box mAP@50:    92.69%
      Box mAP@50-95: 57.40%
      Pose mAP@50:   85.63%
      Pose mAP@50-95:52.40%

[OK] Pipeline complete.
