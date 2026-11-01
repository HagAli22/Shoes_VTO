// demo.js - Shoes VTO Qualcomm Two-Stage vs Single-Stage YOLO Mobile Benchmark (WASM Only)

(function () {
  'use strict';

  // ── ONNX Runtime Web WASM Configuration ─────────────────────────────────────
  // Explicitly WASM only, single-threaded for universal mobile browser compatibility.
  // CRITICAL: wasmPaths must be a full URL so dynamic import() in browser does not fail with 'bare specifier' error.
  if (typeof ort !== 'undefined') {
    const wasmDir = new URL('vendor/', window.location.href).href;
    ort.env.wasm.wasmPaths = wasmDir;
    ort.env.wasm.numThreads = 1;
    console.log('[ORT Web] Configured wasmPaths to:', wasmDir);
  }

  // ── Keypoint Schema & Connections (13 Keypoints) ────────────────────────────
  const KEYPOINT_NAMES = [
    "toe_ground",        // 0
    "heel_back",         // 1
    "heel_ground",       // 2
    "ball_medial",       // 3
    "ball_lateral",      // 4
    "instep_top",        // 5
    "arch_medial",       // 6
    "midfoot_lateral",   // 7
    "malleolus_medial",  // 8
    "malleolus_lateral", // 9
    "toe_tip",           // 10
    "throat",            // 11
    "achilles"           // 12
  ];

  const SKELETON_PAIRS = [
    [2, 1],   // heel_ground -> heel_back
    [2, 6],   // heel_ground -> arch_medial
    [6, 3],   // arch_medial -> ball_medial
    [3, 10],  // ball_medial -> toe_tip
    [10, 0],  // toe_tip -> toe_ground
    [0, 4],   // toe_ground -> ball_lateral
    [4, 7],   // ball_lateral -> midfoot_lateral
    [7, 2],   // midfoot_lateral -> heel_ground
    [10, 5],  // toe_tip -> instep_top
    [5, 11],  // instep_top -> throat
    [8, 11],  // malleolus_medial -> throat
    [9, 11],  // malleolus_lateral -> throat
    [8, 12],  // malleolus_medial -> achilles
    [9, 12],  // malleolus_lateral -> achilles
    [1, 12]   // heel_back -> achilles
  ];

  // ── Colors ──────────────────────────────────────────────────────────────────
  const COLOR_LEFT = {
    bbox: '#00bfff',      // Sky Blue / Cyan (matches eval_video.py BGR (255, 191, 0))
    skeleton: '#0099e6',
    kptVis: '#ffff00',    // Yellow
    kptOcc: '#b3b300',
    label: 'Left Foot'
  };

  const COLOR_RIGHT = {
    bbox: '#ff8c00',      // Orange (matches eval_video.py BGR (0, 140, 255))
    skeleton: '#32cd32',  // Lime Green
    kptVis: '#00ff00',    // Bright Green
    kptOcc: '#009900',
    label: 'Right Foot'
  };

  // ── Model Configurations ────────────────────────────────────────────────────
  const MODEL_CONFIGS = {
    // --- Two-Stage Qualcomm Experiments (Apache-2.0, improved5) ---
    'twostage_fp32_fp32': {
      type: 'twostage',
      name: 'NanoDet(FP32) + RTMPose(FP32)',
      pill: 'ND-FP32 + RTM-FP32',
      sizeMb: '19.09 MB',
      detPath: 'models/nanodet-improved5-fp32.onnx',
      detPrecision: 'float32',
      posePath: 'models/rtmpose-13kp-hand-improved5-fp32.onnx',
      posePrecision: 'float32'
    },
    'twostage_fp32_int8': {
      type: 'twostage',
      name: 'NanoDet(FP32) + RTMPose(INT8)',
      pill: 'ND-FP32 + RTM-INT8',
      sizeMb: '8.04 MB',
      detPath: 'models/nanodet-improved5-fp32.onnx',
      detPrecision: 'float32',
      posePath: 'models/rtmpose-13kp-hand-improved5-int8.onnx',
      posePrecision: 'float32'
    },
    'twostage_int8_int8': {
      type: 'twostage',
      name: 'NanoDet(INT8) + RTMPose(INT8)',
      pill: 'ND-INT8 + RTM-INT8',
      sizeMb: '5.32 MB',
      detPath: 'models/nanodet-improved5-int8.onnx',
      detPrecision: 'float32',
      posePath: 'models/rtmpose-13kp-hand-improved5-int8.onnx',
      posePrecision: 'float32'
    },
    'twostage_int8_fp16': {
      type: 'twostage',
      name: 'NanoDet(INT8) + RTMPose(FP16)',
      pill: 'ND-INT8 + RTM-FP16',
      sizeMb: '8.92 MB',
      detPath: 'models/nanodet-improved5-int8.onnx',
      detPrecision: 'float32',
      posePath: 'models/rtmpose-13kp-hand-improved5-fp16.onnx',
      posePrecision: 'float16'
    },
    'twostage_fp32_fp16': {
      type: 'twostage',
      name: 'NanoDet(FP32) + RTMPose(FP16)',
      pill: 'ND-FP32 + RTM-FP16',
      sizeMb: '11.64 MB',
      detPath: 'models/nanodet-improved5-fp32.onnx',
      detPrecision: 'float32',
      posePath: 'models/rtmpose-13kp-hand-improved5-fp16.onnx',
      posePrecision: 'float16'
    },
    'twostage_fp16_fp16': {
      type: 'twostage',
      name: 'NanoDet(FP16) + RTMPose(FP16)',
      pill: 'ND-FP16 + RTM-FP16',
      sizeMb: '9.66 MB',
      detPath: 'models/nanodet-improved5-fp16.onnx',
      detPrecision: 'float16',
      posePath: 'models/rtmpose-13kp-hand-improved5-fp16.onnx',
      posePrecision: 'float16'
    },
    // --- Single-Stage YOLO Baselines (AGPLv3) ---
    'yolo_int8': {
      type: 'yolo',
      name: 'YOLOv8n (INT8)',
      pill: 'YOLO-INT8',
      sizeMb: '3.46 MB',
      path: 'models/stage-a-256-13kp-int8.onnx',
      precision: 'float32'
    },
    'yolo_fp16': {
      type: 'yolo',
      name: 'YOLOv8n (FP16)',
      pill: 'YOLO-FP16',
      sizeMb: '6.53 MB',
      path: 'models/stage-a-256-13kp-fp16.onnx',
      precision: 'float16'
    }
  };

  // ── One-Euro Filter Implementation ──────────────────────────────────────────
  class OneEuroFilter {
    constructor(minCutoff = 1.2, beta = 0.008, dCutoff = 1.0) {
      this.minCutoff = minCutoff;
      this.beta = beta;
      this.dCutoff = dCutoff;
      this.xPrev = null;
      this.dxPrev = 0.0;
      this.tPrev = null;
    }

    alpha(rate, cutoff) {
      const tau = 1.0 / (2.0 * Math.PI * cutoff);
      const te = rate > 0 ? 1.0 / rate : 0.033;
      return 1.0 / (1.0 + tau / te);
    }

    filter(x, t) {
      if (this.xPrev === null) {
        this.xPrev = x;
        this.tPrev = t;
        return x;
      }
      const dt = Math.max(t - this.tPrev, 1e-4);
      this.tPrev = t;
      const rate = 1.0 / dt;

      const dx = (x - this.xPrev) * rate;
      const edx = this.dxPrev + this.alpha(rate, this.dCutoff) * (dx - this.dxPrev);
      this.dxPrev = edx;

      const cutoff = this.minCutoff + this.beta * Math.abs(edx);
      const a = this.alpha(rate, cutoff);
      const xFiltered = this.xPrev + a * (x - this.xPrev);
      this.xPrev = xFiltered;
      return xFiltered;
    }

    reset() {
      this.xPrev = null;
      this.dxPrev = 0.0;
      this.tPrev = null;
    }
  }

  class KeypointSmoother {
    constructor(numKeypoints = 13) {
      this.filtersX = Array.from({ length: numKeypoints }, () => new OneEuroFilter());
      this.filtersY = Array.from({ length: numKeypoints }, () => new OneEuroFilter());
    }

    smooth(keypoints, t) {
      return keypoints.map((kp, i) => {
        const fx = this.filtersX[i].filter(kp.x, t);
        const fy = this.filtersY[i].filter(kp.y, t);
        return {
          index: kp.index,
          name: kp.name,
          x: fx,
          y: fy,
          visibility: kp.visibility
        };
      });
    }

    reset() {
      this.filtersX.forEach(f => f.reset());
      this.filtersY.forEach(f => f.reset());
    }
  }

  // ── Kalman Box Tracker (8-State Constant Velocity [cx, cy, w, h, vx, vy, vw, vh]^T) ──
  class KalmanBoxTracker {
    constructor(bbox) {
      const cx = (bbox[0] + bbox[2]) / 2.0;
      const cy = (bbox[1] + bbox[3]) / 2.0;
      const w = Math.max(10.0, bbox[2] - bbox[0]);
      const h = Math.max(10.0, bbox[3] - bbox[1]);

      // State: [cx, cy, w, h, vx, vy, vw, vh]
      this.x = [cx, cy, w, h, 0.0, 0.0, 0.0, 0.0];
      this.pPos = 1.0;
      this.pVel = 10.0;
      this.qPos = 0.05;
      this.qVel = 0.10;
      this.rPos = 1.5;
      this.isFirstFrame = true;
    }

    stepPredict() {
      if (this.isFirstFrame) return this.getBbox();
      this.x[0] += this.x[4];
      this.x[1] += this.x[5];
      this.x[2] += this.x[6];
      this.x[3] += this.x[7];
      this.pPos += this.pVel + this.qPos;
      this.pVel += this.qVel;
      return this.getBbox();
    }

    updateMeasurement(bbox) {
      const zx = (bbox[0] + bbox[2]) / 2.0;
      const zy = (bbox[1] + bbox[3]) / 2.0;
      const zw = Math.max(10.0, bbox[2] - bbox[0]);
      const zh = Math.max(10.0, bbox[3] - bbox[1]);

      if (this.isFirstFrame) {
        this.x[0] = zx;
        this.x[1] = zy;
        this.x[2] = zw;
        this.x[3] = zh;
        this.isFirstFrame = false;
        return this.getBbox();
      }

      const k = this.pPos / (this.pPos + this.rPos);
      this.x[4] = 0.6 * this.x[4] + 0.4 * (zx - this.x[0]);
      this.x[5] = 0.6 * this.x[5] + 0.4 * (zy - this.x[1]);
      this.x[6] = 0.6 * this.x[6] + 0.4 * (zw - this.x[2]);
      this.x[7] = 0.6 * this.x[7] + 0.4 * (zh - this.x[3]);

      this.x[0] += k * (zx - this.x[0]);
      this.x[1] += k * (zy - this.x[1]);
      this.x[2] += k * (zw - this.x[2]);
      this.x[3] += k * (zh - this.x[3]);
      this.pPos = (1.0 - k) * this.pPos;

      return this.getBbox();
    }

    getBbox() {
      const cx = this.x[0];
      const cy = this.x[1];
      const w = Math.max(10.0, this.x[2]);
      const h = Math.max(10.0, this.x[3]);
      return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0];
    }

    getVelocity() {
      return [this.x[4], this.x[5]];
    }

    getSpeed() {
      return Math.sqrt(this.x[4] * this.x[4] + this.x[5] * this.x[5]);
    }
  }

  // ── Stateful Foot Track ───────────────────────────────────────────────────────
  class FootTrack {
    constructor(side, rawBbox, handProb = 0.5) {
      this.side = side; // 0 = Left, 1 = Right
      this.kalman = new KalmanBoxTracker(rawBbox);
      this.handProb = handProb;
      this.lostCount = 0;
      this.age = 1;
      this.meanConf = 0.50;
      this.validKpts = 10;
      this.lastKps = null;
      this.smoothedBbox = [...rawBbox];
      this.predictedBbox = [...rawBbox];
    }

    beginFramePredict() {
      const pBox = this.kalman.stepPredict();
      this.predictedBbox = [...pBox];
      return pBox;
    }

    applyObservation(rawBbox, kps, handProb) {
      const postBox = this.kalman.updateMeasurement(rawBbox);
      this.smoothedBbox = [...postBox];
      this.handProb = 0.8 * this.handProb + 0.2 * handProb;
      this.lastKps = kps;
      this.lostCount = 0;
      this.age += 1;

      let sumConf = 0;
      let valid = 0;
      for (const kp of kps) {
        sumConf += kp.visibility;
        if (kp.visibility >= 0.20) valid++;
      }
      this.meanConf = sumConf / 13.0;
      this.validKpts = valid;
      return postBox;
    }

    extrapolateUnobserved() {
      this.smoothedBbox = [...this.predictedBbox];
      this.lostCount += 1;
      this.age += 1;
      if (this.lastKps) {
        const [vx, vy] = this.kalman.getVelocity();
        for (const kp of this.lastKps) {
          kp.x += vx;
          kp.y += vy;
          kp.visibility *= 0.95;
        }
      }
      return this.smoothedBbox;
    }
  }

  // ── Motion-Adaptive Detector Policy (Fixed Heartbeat 5 + Recovery) ───────────
  class MotionAdaptiveDetector {
    constructor(interval = 5) {
      this.interval = interval;
      this.framesSinceDet = 0;
      this.totalFrames = 0;
      this.skippedFrames = 0;
      this.lastReason = 'initial';
    }

    shouldDetect(tracks, heartbeatEnabled) {
      this.totalFrames++;
      this.framesSinceDet++;

      if (!heartbeatEnabled) {
        this.framesSinceDet = 0;
        this.lastReason = 'every_frame';
        return { detect: true, reason: 'every_frame' };
      }

      // 1. Mandatory Heartbeat Interval (default 5 frames)
      if (this.framesSinceDet >= this.interval) {
        this.framesSinceDet = 0;
        this.lastReason = 'heartbeat_interval';
        return { detect: true, reason: 'heartbeat_interval' };
      }

      // 2. Active Track Availability
      const activeSides = [0, 1].filter(s => tracks[s] && tracks[s].lostCount === 0);
      if (activeSides.length < 2) {
        this.framesSinceDet = 0;
        this.lastReason = 'missing_tracks';
        return { detect: true, reason: 'missing_tracks' };
      }

      // 3. Pose Quality & Confidence Degradation
      for (const side of activeSides) {
        const trk = tracks[side];
        if (trk.meanConf < 0.25 || trk.validKpts < 3) {
          this.framesSinceDet = 0;
          this.lastReason = 'low_confidence';
          return { detect: true, reason: 'low_confidence' };
        }
        if (trk.kalman.getSpeed() > 25.0) {
          this.framesSinceDet = 0;
          this.lastReason = 'fast_motion';
          return { detect: true, reason: 'fast_motion' };
        }
      }

      // All tracks stable: SKIP Stage A!
      this.skippedFrames++;
      this.lastReason = 'pose_guided_skip';
      return { detect: false, reason: 'pose_guided_skip' };
    }

    getSkipRatePct() {
      return this.totalFrames > 0 ? (this.skippedFrames / this.totalFrames * 100.0) : 0.0;
    }

    reset() {
      this.framesSinceDet = 0;
      this.totalFrames = 0;
      this.skippedFrames = 0;
      this.lastReason = 'reset';
    }
  }

  // ── State Management ────────────────────────────────────────────────────────
  const state = {
    activeModelKey: 'twostage_fp32_fp32',
    activeModelConfig: MODEL_CONFIGS['twostage_fp32_fp32'],
    currentSource: 'video',
    filterEnabled: true,
    heartbeatEnabled: true,
    facingMode: 'environment',
    isPlaying: false,
    isLoadingModel: false,

    // Sessions for Two-Stage
    detSession: null,
    detInputName: null,
    loadedDetPath: null,
    poseSession: null,
    poseInputName: null,
    loadedPosePath: null,

    // Session for Single-Stage YOLO
    yoloSession: null,
    yoloInputName: null,

    modelError: null,
    loadDuration: null,

    tracks: {
      0: null, // Left
      1: null  // Right
    },
    adaptiveDetector: new MotionAdaptiveDetector(5),

    smoothers: {
      0: new KeypointSmoother(13), // Left
      1: new KeypointSmoother(13)  // Right
    },

    // Telemetry rolling window
    stageATimes: [],
    stageBTimes: [],
    inferTimes: [],
    prePostTimes: [],
    e2eTimes: [],
    lastFrameTime: performance.now(),
    fps: 0,
    activeDetections: 0
  };

  // ── DOM Elements ────────────────────────────────────────────────────────────
  const banner = document.getElementById('diagnosticBanner');
  const video = document.getElementById('inputVideo');
  const canvas = document.getElementById('renderCanvas');
  const ctx = canvas.getContext('2d');

  const allModelButtons = document.querySelectorAll('.segmented-control button[data-model]');
  const sourceButtons = document.querySelectorAll('#sourceControl button');
  const heartbeatToggle = document.getElementById('heartbeatToggle');
  const filterToggle = document.getElementById('filterToggle');
  const playPauseBtn = document.getElementById('playPauseBtn');
  const playPauseIcon = document.getElementById('playPauseIcon');
  const playPauseText = document.getElementById('playPauseText');
  const flipCameraBtn = document.getElementById('flipCameraBtn');
  const restartBtn = document.getElementById('restartBtn');

  const hudFps = document.getElementById('hudFps');
  const hudInferMed = document.getElementById('hudInferMed');
  const hudInferP95 = document.getElementById('hudInferP95');
  const hudInferCur = document.getElementById('hudInferCur');
  const hudProc = document.getElementById('hudProc');
  const hudE2E = document.getElementById('hudE2E');
  const hudLoadTime = document.getElementById('hudLoadTime');
  const hudStageSplit = document.getElementById('hudStageSplit');
  const hudTrackingMode = document.getElementById('hudTrackingMode');
  const hudSkipRate = document.getElementById('hudSkipRate');
  const hudDetections = document.getElementById('hudDetections');
  const overlayModelPill = document.getElementById('overlayModelPill');
  const overlayFpsPill = document.getElementById('overlayFpsPill');

  // ── Offscreen Preprocessing & Cropping Canvases ──────────────────────────────
  const offscreenCanvas = document.createElement('canvas');
  offscreenCanvas.width = 256;
  offscreenCanvas.height = 256;
  const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });

  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = 256;
  cropCanvas.height = 256;
  const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });

  // ── Float16 Conversion Utilities ────────────────────────────────────────────
  function float32ToFloat16(val) {
    const floatView = new Float32Array(1);
    const int32View = new Int32Array(floatView.buffer);
    floatView[0] = val;
    const x = int32View[0];
    let bits = (x >> 16) & 0x8000;
    let m = (x >> 12) & 0x07ff;
    const e = (x >> 23) & 0xff;
    if (e < 103) return bits;
    if (e > 142) {
      bits |= 0x7c00;
      bits |= ((e === 255) ? 0 : 1) && (x & 0x007fffff);
      return bits;
    }
    if (e < 113) {
      m |= 0x0800;
      bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
      return bits;
    }
    bits |= ((e - 112) << 10) | (m >> 1);
    bits += m & 1;
    return bits;
  }

  function float16ToFloat32(h) {
    const s = (h >> 15) & 0x0001;
    const e = (h >> 10) & 0x001f;
    const f = h & 0x03ff;
    if (e === 0) {
      return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    } else if (e === 31) {
      return f ? NaN : ((s ? -1 : 1) * Infinity);
    }
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  function getFloatValue(array, index) {
    if (array instanceof Float32Array || array instanceof Float64Array) {
      return array[index];
    }
    if (typeof Float16Array !== 'undefined' && array instanceof Float16Array) {
      return array[index];
    }
    if (array instanceof Uint16Array) {
      return float16ToFloat32(array[index]);
    }
    return Number(array[index]);
  }

  // ── Diagnostic Banner Helper ────────────────────────────────────────────────
  function showBanner(type, message) {
    banner.className = `banner ${type}`;
    banner.textContent = message;
    banner.style.display = 'block';
  }

  function hideBanner() {
    banner.style.display = 'none';
  }

  // ── Session Creation Helper (Strict WASM) ───────────────────────────────────
  async function createWasmSession(modelPath) {
    const sessionOptions = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    };
    return await ort.InferenceSession.create(modelPath, sessionOptions);
  }

  // ── Model Loader (Atomic Commit to Avoid Precision Race Conditions) ───────
  async function loadModel(modelKey) {
    const config = MODEL_CONFIGS[modelKey];
    if (!config) {
      console.error('Unknown model key:', modelKey);
      return;
    }

    state.isLoadingModel = true;
    state.modelError = null;
    overlayModelPill.textContent = `${config.pill} WASM`;

    showBanner('info', `Loading ${config.name} (${config.sizeMb}) via WASM backend...`);

    const t0 = performance.now();
    try {
      if (typeof ort === 'undefined') {
        throw new Error('ONNX Runtime Web library failed to load.');
      }

      if (config.type === 'twostage') {
        // 1. Create or reuse Stage A Detector session
        let newDetSession = state.detSession;
        let newDetInputName = state.detInputName;
        if (!newDetSession || state.loadedDetPath !== config.detPath) {
          showBanner('info', `[1/2] Loading Stage A Detector: ${config.detPath}...`);
          newDetSession = await createWasmSession(config.detPath);
          newDetInputName = newDetSession.inputNames[0];

          const isDetFp16 = config.detPrecision === 'float16';
          const detWarmupTensor = createDummyTensor(isDetFp16 ? 'float16' : 'float32', [1, 3, 256, 256]);
          await newDetSession.run({ [newDetInputName]: detWarmupTensor });
          state.loadedDetPath = config.detPath;
        }

        // 2. Create or reuse Stage B RTMPose session with matching precision
        let newPoseSession = state.poseSession;
        let newPoseInputName = state.poseInputName;
        if (!newPoseSession || state.loadedPosePath !== config.posePath) {
          showBanner('info', `[2/2] Loading Stage B RTMPose: ${config.posePath}...`);
          newPoseSession = await createWasmSession(config.posePath);
          newPoseInputName = newPoseSession.inputNames[0];

          const isPoseFp16 = config.posePrecision === 'float16';
          const poseWarmupTensor = createDummyTensor(isPoseFp16 ? 'float16' : 'float32', [1, 3, 256, 256]);
          await newPoseSession.run({ [newPoseInputName]: poseWarmupTensor });
          state.loadedPosePath = config.posePath;
        }

        // 3. Reset foot tracks and detector state on model change
        state.tracks[0] = null;
        state.tracks[1] = null;
        state.adaptiveDetector.reset();

        // 4. ATOMIC COMMIT: Swap sessions and configuration simultaneously
        state.yoloSession = null;
        state.yoloInputName = null;
        state.detSession = newDetSession;
        state.detInputName = newDetInputName;
        state.poseSession = newPoseSession;
        state.poseInputName = newPoseInputName;
        state.activeModelKey = modelKey;
        state.activeModelConfig = config;

      } else {
        // Single-Stage YOLO
        showBanner('info', `Loading Single-Stage YOLO: ${config.path}...`);
        const newYoloSession = await createWasmSession(config.path);
        const newYoloInputName = newYoloSession.inputNames[0];

        const isFp16 = config.precision === 'float16';
        const yoloWarmupTensor = createDummyTensor(isFp16 ? 'float16' : 'float32', [1, 3, 256, 256]);
        await newYoloSession.run({ [newYoloInputName]: yoloWarmupTensor });

        // ATOMIC COMMIT
        state.detSession = null;
        state.detInputName = null;
        state.poseSession = null;
        state.poseInputName = null;
        state.yoloSession = newYoloSession;
        state.yoloInputName = newYoloInputName;
        state.activeModelKey = modelKey;
        state.activeModelConfig = config;
      }

      state.isLoadingModel = false;
      const dt = (performance.now() - t0).toFixed(0);
      state.loadDuration = dt;
      hudLoadTime.textContent = `${dt} ms`;

      showBanner('info', `✓ ${config.name} loaded in ${dt} ms. Execution Provider: WASM (CPU Only).`);
      setTimeout(() => {
        if (!state.modelError) hideBanner();
      }, 4000);

    } catch (err) {
      console.error('Model load failure:', err);
      state.isLoadingModel = false;
      state.modelError = err.message || String(err);
      showBanner('error', `❌ Model Load Error: ${state.modelError}`);
    }
  }

  // ── Warmup Helpers ──────────────────────────────────────────────────────────
  async function warmUpTwoStage(config) {
    const isDetFp16 = config.detPrecision === 'float16';
    const detTensor = createDummyTensor(isDetFp16 ? 'float16' : 'float32', [1, 3, 256, 256]);
    await state.detSession.run({ [state.detInputName]: detTensor });

    const poseTensor = createDummyTensor('float16', [1, 3, 256, 256]);
    await state.poseSession.run({ [state.poseInputName]: poseTensor });
    console.log('[TwoStage] Warmup inference completed.');
  }

  async function warmUpYolo(config) {
    const isFp16 = config.precision === 'float16';
    const tensor = createDummyTensor(isFp16 ? 'float16' : 'float32', [1, 3, 256, 256]);
    await state.yoloSession.run({ [state.yoloInputName]: tensor });
    console.log('[YOLO] Warmup inference completed.');
  }

  function createDummyTensor(type, dims) {
    const totalElements = dims.reduce((a, b) => a * b, 1);
    if (type === 'float16') {
      const f16Data = (typeof Float16Array !== 'undefined')
        ? new Float16Array(totalElements)
        : new Uint16Array(totalElements);
      return new ort.Tensor('float16', f16Data, dims);
    } else {
      const f32 = new Float32Array(totalElements);
      return new ort.Tensor('float32', f32, dims);
    }
  }

  // ── Frame Preprocessing ─────────────────────────────────────────────────────
  // NanoDet Stage A Direct Stretch Resize (Exact Parity with eval_video.py & generate_quantized_videos.py)
  function preprocessNanoDetDirect(sourceElement, precision = 'float32') {
    const srcW = sourceElement.videoWidth || sourceElement.width;
    const srcH = sourceElement.videoHeight || sourceElement.height;
    if (!srcW || !srcH) return null;

    const targetSize = 256;
    // Direct stretch resize without padding (matching cv2.resize(frame, (256, 256)))
    offscreenCtx.drawImage(sourceElement, 0, 0, srcW, srcH, 0, 0, targetSize, targetSize);

    const tensor = makeTensorFromCanvas(offscreenCtx, targetSize, targetSize, precision);
    const scaleX = srcW / targetSize;
    const scaleY = srcH / targetSize;

    return {
      tensor,
      scaleX,
      scaleY,
      srcW,
      srcH
    };
  }

  // Single-Stage YOLO Letterbox with 114 Gray Padding
  function preprocessYoloLetterbox(sourceElement, precision = 'float32') {
    const srcW = sourceElement.videoWidth || sourceElement.width;
    const srcH = sourceElement.videoHeight || sourceElement.height;
    if (!srcW || !srcH) return null;

    const targetSize = 256;
    const ratio = Math.min(targetSize / srcW, targetSize / srcH);
    const nw = Math.round(srcW * ratio);
    const nh = Math.round(srcH * ratio);
    const padX = (targetSize - nw) / 2;
    const padY = (targetSize - nh) / 2;

    offscreenCtx.fillStyle = 'rgb(114, 114, 114)';
    offscreenCtx.fillRect(0, 0, targetSize, targetSize);
    offscreenCtx.drawImage(sourceElement, 0, 0, srcW, srcH, padX, padY, nw, nh);

    const tensor = makeTensorFromCanvas(offscreenCtx, targetSize, targetSize, precision);
    return {
      tensor,
      ratio,
      padX,
      padY,
      srcW,
      srcH
    };
  }

  function makeTensorFromCanvas(context, w, h, precision = 'float32') {
    const imgData = context.getImageData(0, 0, w, h).data;
    const numPixels = w * h;
    const rOffset = 0;
    const gOffset = numPixels;
    const bOffset = numPixels * 2;

    if (precision === 'float16') {
      let tensorData;
      if (typeof Float16Array !== 'undefined') {
        tensorData = new Float16Array(3 * numPixels);
        for (let i = 0; i < numPixels; i++) {
          const srcIdx = i * 4;
          tensorData[rOffset + i] = imgData[srcIdx] / 255.0;
          tensorData[gOffset + i] = imgData[srcIdx + 1] / 255.0;
          tensorData[bOffset + i] = imgData[srcIdx + 2] / 255.0;
        }
      } else {
        tensorData = new Uint16Array(3 * numPixels);
        for (let i = 0; i < numPixels; i++) {
          const srcIdx = i * 4;
          tensorData[rOffset + i] = float32ToFloat16(imgData[srcIdx] / 255.0);
          tensorData[gOffset + i] = float32ToFloat16(imgData[srcIdx + 1] / 255.0);
          tensorData[bOffset + i] = float32ToFloat16(imgData[srcIdx + 2] / 255.0);
        }
      }
      return new ort.Tensor('float16', tensorData, [1, 3, h, w]);
    } else {
      const tensorData = new Float32Array(3 * numPixels);
      for (let i = 0; i < numPixels; i++) {
        const srcIdx = i * 4;
        tensorData[rOffset + i] = imgData[srcIdx] / 255.0;
        tensorData[gOffset + i] = imgData[srcIdx + 1] / 255.0;
        tensorData[bOffset + i] = imgData[srcIdx + 2] / 255.0;
      }
      return new ort.Tensor('float32', tensorData, [1, 3, h, w]);
    }
  }

  // ── Geometry & NMS Utilities ────────────────────────────────────────────────
  function computeIoU(b1, b2) {
    const x1 = Math.max(b1[0], b2[0]);
    const y1 = Math.max(b1[1], b2[1]);
    const x2 = Math.min(b1[2], b2[2]);
    const y2 = Math.min(b1[3], b2[3]);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const a1 = (b1[2] - b1[0]) * (b1[3] - b1[1]);
    const a2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
    const union = a1 + a2 - inter;
    return union > 0 ? inter / union : 0;
  }

  function nms(boxes, scores, iouThresh = 0.45) {
    const indices = Array.from({ length: scores.length }, (_, i) => i);
    indices.sort((a, b) => scores[b] - scores[a]);
    const keep = [];

    for (const idx of indices) {
      let overlap = false;
      for (const k of keep) {
        if (computeIoU(boxes[idx], boxes[k]) > iouThresh) {
          overlap = true;
          break;
        }
      }
      if (!overlap) keep.push(idx);
    }
    return keep;
  }

  // ── Candidate Filter & Keypoint Bbox Utilities ──────────────────────────────
  function computeKptBbox(keypoints, padding = 10.0) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const kp of keypoints) {
      if (kp.visibility >= 0.15) {
        if (kp.x < minX) minX = kp.x;
        if (kp.y < minY) minY = kp.y;
        if (kp.x > maxX) maxX = kp.x;
        if (kp.y > maxY) maxY = kp.y;
      }
    }
    if (minX === Infinity) return null;
    return [
      Math.max(0, minX - padding),
      Math.max(0, minY - padding),
      maxX + padding,
      maxY + padding
    ];
  }

  function filterCandidateRois(rawDetections, tracks, iouThresh = 0.40) {
    if (rawDetections.length <= 1) return rawDetections;

    const matched = [];
    const unmatched = [];

    for (const det of rawDetections) {
      let isMatched = false;
      for (const side of [0, 1]) {
        const trk = tracks[side];
        if (trk && trk.lostCount === 0 && computeIoU(det.origBox, trk.smoothedBbox) > 0.25) {
          matched.push(det);
          isMatched = true;
          break;
        }
      }
      if (!isMatched) {
        unmatched.push(det);
      }
    }

    unmatched.sort((a, b) => b.score - a.score);
    const selected = matched.slice(0, 2);

    for (const cand of unmatched) {
      if (selected.length >= 2) break;
      let overlap = false;
      for (const s of selected) {
        if (computeIoU(cand.origBox, s.origBox) >= iouThresh) {
          overlap = true;
          break;
        }
      }
      if (!overlap) {
        selected.push(cand);
      }
    }

    return selected.slice(0, 2);
  }

  // ── NanoDet Stage A Decoder ─────────────────────────────────────────────────
  // boxes [1, 1344, 4] (float32), scores [1, 1344, 1] (float32 or float16)
  function decodeNanoDet(boxesTensor, scoresTensor, prepMeta, confThresh = 0.35, nmsThresh = 0.45) {
    const numAnchors = 1344;
    const boxesData = boxesTensor.data;
    const scoresData = scoresTensor.data;

    const candidates = [];

    for (let a = 0; a < numAnchors; a++) {
      const score = getFloatValue(scoresData, a);
      if (score >= confThresh) {
        const x1 = getFloatValue(boxesData, a * 4 + 0);
        const y1 = getFloatValue(boxesData, a * 4 + 1);
        const x2 = getFloatValue(boxesData, a * 4 + 2);
        const y2 = getFloatValue(boxesData, a * 4 + 3);

        // Map from direct stretch 256x256 to original source coords (exact parity with eval_video.py)
        const origX1 = Math.max(0, Math.min(prepMeta.srcW, x1 * prepMeta.scaleX));
        const origY1 = Math.max(0, Math.min(prepMeta.srcH, y1 * prepMeta.scaleY));
        const origX2 = Math.max(0, Math.min(prepMeta.srcW, x2 * prepMeta.scaleX));
        const origY2 = Math.max(0, Math.min(prepMeta.srcH, y2 * prepMeta.scaleY));

        candidates.push({
          boxPx: [x1, y1, x2, y2],
          origBox: [origX1, origY1, origX2, origY2],
          score: score
        });
      }
    }

    if (candidates.length === 0) return [];

    const keepIndices = nms(candidates.map(c => c.boxPx), candidates.map(c => c.score), nmsThresh);
    let detections = keepIndices.map(i => candidates[i]);

    // Limit to top 2 detected feet
    if (detections.length > 2) {
      detections.sort((a, b) => b.score - a.score);
      detections = detections.slice(0, 2);
    }

    return detections;
  }

  // ── Stage B Crop Geometry & Extraction (1.25x Square Crop) ───────────────────
  function computeCropBox(cx, cy, bw, bh, cropPadding = 0.25) {
    const bwSafe = Math.max(1.0, bw);
    const bhSafe = Math.max(1.0, bh);
    const ar = Math.max(bwSafe / bhSafe, bhSafe / bwSafe);
    const extraMargin = ar > 1.3 ? 0.15 : 0.05;
    const size = Math.max(bwSafe, bhSafe) * (1.0 + cropPadding + extraMargin);
    return { cx, cy, size };
  }

  function extractCropTensor(sourceElement, cx, cy, size, precision = 'float16') {
    const srcW = sourceElement.videoWidth || sourceElement.width;
    const srcH = sourceElement.videoHeight || sourceElement.height;

    const cropX1 = cx - size / 2.0;
    const cropY1 = cy - size / 2.0;
    const cropX2 = cx + size / 2.0;
    const cropY2 = cy + size / 2.0;

    const sx = Math.max(0, cropX1);
    const sy = Math.max(0, cropY1);
    const sx2 = Math.min(srcW, cropX2);
    const sy2 = Math.min(srcH, cropY2);
    const sw = Math.max(0, sx2 - sx);
    const sh = Math.max(0, sy2 - sy);

    const scale = 256.0 / size;
    const dx = (sx - cropX1) * scale;
    const dy = (sy - cropY1) * scale;
    const dw = sw * scale;
    const dh = sh * scale;

    cropCtx.fillStyle = 'rgb(114, 114, 114)';
    cropCtx.fillRect(0, 0, 256, 256);

    if (sw > 0 && sh > 0) {
      cropCtx.drawImage(sourceElement, sx, sy, sw, sh, dx, dy, dw, dh);
    }

    return makeTensorFromCanvas(cropCtx, 256, 256, precision);
  }

  // ── SimCC Sub-pixel Continuous Expectation Decoder ───────────────────────────
  function decodeSimccAxis(logitsData, offset, length = 512, ratio = 2.0, win = 12) {
    let maxVal = -Infinity;
    for (let i = 0; i < length; i++) {
      const val = getFloatValue(logitsData, offset + i);
      if (val > maxVal) maxVal = val;
    }

    let sumExp = 0;
    const probs = new Float32Array(length);
    let peakIdx = 0;
    let peakProb = -1;

    for (let i = 0; i < length; i++) {
      const p = Math.exp(getFloatValue(logitsData, offset + i) - maxVal);
      probs[i] = p;
      sumExp += p;
    }

    for (let i = 0; i < length; i++) {
      probs[i] /= sumExp;
      if (probs[i] > peakProb) {
        peakProb = probs[i];
        peakIdx = i;
      }
    }

    const startIdx = Math.max(0, peakIdx - win);
    const endIdx = Math.min(length - 1, peakIdx + win);
    let mass = 0;
    let weightedSum = 0;

    for (let i = startIdx; i <= endIdx; i++) {
      const p = probs[i];
      mass += p;
      weightedSum += p * i;
    }

    const coord = mass > 1e-8 ? (weightedSum / mass) / ratio : (peakIdx / ratio);
    return { coord, mass };
  }

  function decodeRtmPoseOutputs(outputs, cx, cy, size) {
    const simccXData = outputs['simcc_x'].data;
    const simccYData = outputs['simcc_y'].data;
    const visData = outputs['vis'] ? outputs['vis'].data : null;
    const handData = outputs['hand'] ? outputs['hand'].data : null;

    const keypoints = [];
    let sumConf = 0;
    let validKptsCount = 0;

    for (let k = 0; k < 13; k++) {
      const resX = decodeSimccAxis(simccXData, k * 512, 512, 2.0, 12);
      const resY = decodeSimccAxis(simccYData, k * 512, 512, 2.0, 12);

      const kptScore = Math.sqrt(Math.max(0, resX.mass * resY.mass));
      const vis = visData ? getFloatValue(visData, k) : 1.0;
      const conf = kptScore * vis;

      // Project back from crop space [0..256] to original source frame
      const kptOrigX = cx + (resX.coord - 128.0) * (size / 256.0);
      const kptOrigY = cy + (resY.coord - 128.0) * (size / 256.0);

      keypoints.push({
        index: k,
        name: KEYPOINT_NAMES[k],
        x: kptOrigX,
        y: kptOrigY,
        visibility: conf
      });

      sumConf += conf;
      if (conf >= 0.20) validKptsCount++;
    }

    const meanConf = sumConf / 13.0;
    const handProb = handData ? getFloatValue(handData, 0) : 0.5;

    return {
      keypoints,
      meanConf,
      validKptsCount,
      handProb
    };
  }

  // ── Deterministic Geometric Chirality ───────────────────────────────────────
  function determineGeometricChirality13kp(keypoints) {
    const toe = keypoints[10];       // toe_tip
    const ballLat = keypoints[4];    // ball_lateral
    const ballMed = keypoints[3];    // ball_medial

    let heel;
    if (keypoints[2].visibility > 0.20) {
      heel = keypoints[2];
    } else if (keypoints[1].visibility > 0.20) {
      heel = keypoints[1];
    } else {
      heel = {
        x: 0.5 * (keypoints[8].x + keypoints[9].x),
        y: 0.5 * (keypoints[8].y + keypoints[9].y)
      };
    }

    const vAxisX = toe.x - heel.x;
    const vAxisY = toe.y - heel.y;

    const crossLat = vAxisX * (ballLat.y - heel.y) - vAxisY * (ballLat.x - heel.x);
    const crossMed = vAxisX * (ballMed.y - heel.y) - vAxisY * (ballMed.x - heel.x);

    const diff = crossLat - crossMed;
    return diff < 0 ? 0 : 1; // 0 = Left Foot, 1 = Right Foot
  }

  // ── Single-Stage YOLO Decoder ───────────────────────────────────────────────
  function sigmoid(v) {
    return 1.0 / (1.0 + Math.exp(-v));
  }

  function decodeYoloPose(rawOutput, prepMeta, confThresh = 0.25, nmsThresh = 0.45) {
    const numAnchors = 1344;
    const data = rawOutput.data;
    const candidates = [];

    for (let a = 0; a < numAnchors; a++) {
      const cls0 = data[4 * numAnchors + a];
      const cls1 = data[5 * numAnchors + a];
      const maxCls = Math.max(cls0, cls1);

      if (maxCls >= confThresh) {
        const cx = data[0 * numAnchors + a];
        const cy = data[1 * numAnchors + a];
        const w = data[2 * numAnchors + a];
        const h = data[3 * numAnchors + a];

        const x1 = cx - w / 2;
        const y1 = cy - h / 2;
        const x2 = cx + w / 2;
        const y2 = cy + h / 2;

        const kpts = [];
        for (let k = 0; k < 13; k++) {
          const kx = data[(6 + k * 3) * numAnchors + a];
          const ky = data[(7 + k * 3) * numAnchors + a];
          const kv = data[(8 + k * 3) * numAnchors + a];
          const vis = (kv >= 0 && kv <= 1) ? kv : sigmoid(kv);

          kpts.push({
            index: k,
            name: KEYPOINT_NAMES[k],
            x: (kx - prepMeta.padX) / prepMeta.ratio,
            y: (ky - prepMeta.padY) / prepMeta.ratio,
            visibility: vis
          });
        }

        candidates.push({
          boxPx: [x1, y1, x2, y2],
          origBox: [
            Math.max(0, Math.min(prepMeta.srcW, (x1 - prepMeta.padX) / prepMeta.ratio)),
            Math.max(0, Math.min(prepMeta.srcH, (y1 - prepMeta.padY) / prepMeta.ratio)),
            Math.max(0, Math.min(prepMeta.srcW, (x2 - prepMeta.padX) / prepMeta.ratio)),
            Math.max(0, Math.min(prepMeta.srcH, (y2 - prepMeta.padY) / prepMeta.ratio))
          ],
          score: maxCls,
          rawClassId: cls0 >= cls1 ? 0 : 1,
          keypoints: kpts
        });
      }
    }

    if (candidates.length === 0) return [];

    const keepIndices = nms(candidates.map(c => c.boxPx), candidates.map(c => c.score), nmsThresh);
    let detections = keepIndices.map(idx => candidates[idx]);

    // Physical Foot Deduplication (IoU > 0.60)
    if (detections.length > 1) {
      detections.sort((a, b) => b.score - a.score);
      const deduped = [];
      for (const d of detections) {
        let overlap = false;
        for (const k of deduped) {
          if (computeIoU(d.origBox, k.origBox) > 0.60) {
            overlap = true;
            break;
          }
        }
        if (!overlap) deduped.push(d);
      }
      detections = deduped;
    }

    return detections;
  }

  // ── Render Visualization ────────────────────────────────────────────────────
  function renderDetections(sourceElement, detections) {
    const srcW = sourceElement.videoWidth || sourceElement.width;
    const srcH = sourceElement.videoHeight || sourceElement.height;
    if (!srcW || !srcH) return;

    if (canvas.width !== srcW || canvas.height !== srcH) {
      canvas.width = srcW;
      canvas.height = srcH;
    }

    ctx.clearRect(0, 0, srcW, srcH);
    ctx.drawImage(sourceElement, 0, 0, srcW, srcH);

    // Scale-proportional rendering for crisp display on both mobile and 1080p/4K video
    const scaleFactor = Math.max(1, srcW / 640);

    for (const det of detections) {
      const clsId = det.classId;
      const cfg = clsId === 0 ? COLOR_LEFT : COLOR_RIGHT;
      const b = det.origBox;
      const kpts = det.keypoints;

      // 1. Bounding box
      ctx.strokeStyle = cfg.bbox;
      ctx.lineWidth = Math.round(3 * scaleFactor);
      ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);

      // Badge
      ctx.fillStyle = cfg.bbox;
      const badgeText = `${cfg.label} ${(det.score * 100).toFixed(0)}%`;
      const fontSize = Math.round(14 * scaleFactor);
      const badgeHeight = Math.round(24 * scaleFactor);
      ctx.font = `bold ${fontSize}px sans-serif`;
      const textWidth = ctx.measureText(badgeText).width;
      ctx.fillRect(b[0], Math.max(0, b[1] - badgeHeight), textWidth + 12 * scaleFactor, badgeHeight);
      ctx.fillStyle = '#0f172a';
      ctx.fillText(badgeText, b[0] + 6 * scaleFactor, Math.max(fontSize + 4, b[1] - 6 * scaleFactor));

      // 2. Skeleton
      ctx.strokeStyle = cfg.skeleton;
      ctx.lineWidth = Math.round(2.5 * scaleFactor);
      for (const [i1, i2] of SKELETON_PAIRS) {
        const p1 = kpts[i1];
        const p2 = kpts[i2];
        if (p1.visibility >= 0.15 && p2.visibility >= 0.15) {
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }

      // 3. Keypoints
      const ptRadius = Math.round(4.5 * scaleFactor);
      for (const kp of kpts) {
        if (kp.visibility < 0.15) continue;
        ctx.beginPath();
        if (kp.visibility >= 0.35) {
          ctx.fillStyle = cfg.kptVis;
          ctx.arc(kp.x, kp.y, ptRadius, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = '#000';
          ctx.lineWidth = Math.max(1, Math.round(1 * scaleFactor));
          ctx.stroke();
        } else {
          ctx.strokeStyle = cfg.kptOcc;
          ctx.lineWidth = Math.max(1.5, Math.round(2 * scaleFactor));
          ctx.arc(kp.x, kp.y, Math.round(3.5 * scaleFactor), 0, Math.PI * 2);
          ctx.stroke();
        }
      }
    }
  }

  // ── Main Frame Processing Loop ──────────────────────────────────────────────
  let animFrameId = null;

  async function processFrame() {
    if (!state.isPlaying) return;

    const frameStart = performance.now();
    const sourceElement = video;

    if (sourceElement.readyState < 2 || sourceElement.paused) {
      animFrameId = requestAnimationFrame(processFrame);
      return;
    }

    // Guard: Do not execute inference while a new model session is loading
    if (state.isLoadingModel) {
      renderDetections(sourceElement, []);
      if (state.isPlaying) {
        animFrameId = requestAnimationFrame(processFrame);
      }
      return;
    }

    const cfg = state.activeModelConfig;

    if (!state.modelError && cfg) {
      try {
        if (cfg.type === 'twostage' && state.detSession && state.poseSession) {
          // ── Two-Stage Qualcomm Pipeline (Pose-Guided Heartbeat 5x) ──
          const dec = state.adaptiveDetector.shouldDetect(state.tracks, state.heartbeatEnabled);
          let tDet = 0;
          let totalPoseInfer = 0;
          let finalDetections = [];

          if (dec.detect) {
            // ── Path 1: Stage A Detector Frame (Heartbeat / Recovery / Fallback) ──
            const tPre0 = performance.now();
            const prep = preprocessNanoDetDirect(sourceElement, cfg.detPrecision);
            const tPre = performance.now() - tPre0;

            if (prep) {
              const tDet0 = performance.now();
              const detOutputs = await state.detSession.run({ [state.detInputName]: prep.tensor });
              tDet = performance.now() - tDet0;

              const boxesOut = detOutputs['boxes'];
              const scoresOut = detOutputs['scores'];
              const rawDetections = decodeNanoDet(boxesOut, scoresOut, prep, 0.40, 0.45);

              // Candidate Filter: Limit to strict maximum of 2 candidate foot ROIs
              const candidateDets = filterCandidateRois(rawDetections, state.tracks, 0.40);
              const validCandidates = [];

              for (const det of candidateDets) {
                const b = det.origBox;
                const bw = b[2] - b[0];
                const bh = b[3] - b[1];
                const cx = (b[0] + b[2]) / 2.0;
                const cy = (b[1] + b[3]) / 2.0;

                const cropBox = computeCropBox(cx, cy, bw, bh, 0.25);
                const cropTensor = extractCropTensor(sourceElement, cropBox.cx, cropBox.cy, cropBox.size, cfg.posePrecision);

                const tPose0 = performance.now();
                const poseOutputs = await state.poseSession.run({ [state.poseInputName]: cropTensor });
                totalPoseInfer += (performance.now() - tPose0);

                const poseResult = decodeRtmPoseOutputs(poseOutputs, cropBox.cx, cropBox.cy, cropBox.size);

                // Anatomical Clutter Gate: Reject carpet leaves, floor patterns, shadows
                if (poseResult.meanConf < 0.20 || poseResult.validKptsCount < 3) {
                  continue;
                }

                validCandidates.push({
                  origBox: det.origBox,
                  score: det.score,
                  keypoints: poseResult.keypoints,
                  handProb: poseResult.handProb, // <0.50: Left, >=0.50: Right
                  meanConf: poseResult.meanConf,
                  validKptsCount: poseResult.validKptsCount,
                  jointScore: det.score * (0.5 + 0.5 * poseResult.meanConf),
                  cx: cx,
                  cy: cy
                });
              }

              // Dual-Foot Chirality Partitioning
              let leftCands = validCandidates.filter(c => c.handProb < 0.50);
              let rightCands = validCandidates.filter(c => c.handProb >= 0.50);

              if (leftCands.length >= 2 && rightCands.length === 0) {
                leftCands.sort((a, b) => a.cx - b.cx);
                if (Math.abs(leftCands[1].cx - leftCands[0].cx) > 60) {
                  rightCands.push(leftCands.pop());
                }
              } else if (rightCands.length >= 2 && leftCands.length === 0) {
                rightCands.sort((a, b) => a.cx - b.cx);
                if (Math.abs(rightCands[1].cx - rightCands[0].cx) > 60) {
                  leftCands.push(rightCands.shift());
                }
              }

              const bestLeft = leftCands.length > 0
                ? leftCands.reduce((prev, curr) => (curr.jointScore > prev.jointScore ? curr : prev))
                : null;
              const bestRight = rightCands.length > 0
                ? rightCands.reduce((prev, curr) => (curr.jointScore > prev.jointScore ? curr : prev))
                : null;

              if (bestLeft) {
                if (!state.tracks[0]) {
                  state.tracks[0] = new FootTrack(0, bestLeft.origBox, bestLeft.handProb);
                }
                const kptBox = computeKptBbox(bestLeft.keypoints) || bestLeft.origBox;
                state.tracks[0].applyObservation(kptBox, bestLeft.keypoints, bestLeft.handProb);
                finalDetections.push({
                  origBox: state.tracks[0].smoothedBbox,
                  score: bestLeft.score,
                  classId: 0,
                  keypoints: bestLeft.keypoints
                });
              } else if (state.tracks[0]) {
                state.tracks[0].extrapolateUnobserved();
              }

              if (bestRight) {
                if (!state.tracks[1]) {
                  state.tracks[1] = new FootTrack(1, bestRight.origBox, bestRight.handProb);
                }
                const kptBox = computeKptBbox(bestRight.keypoints) || bestRight.origBox;
                state.tracks[1].applyObservation(kptBox, bestRight.keypoints, bestRight.handProb);
                finalDetections.push({
                  origBox: state.tracks[1].smoothedBbox,
                  score: bestRight.score,
                  classId: 1,
                  keypoints: bestRight.keypoints
                });
              } else if (state.tracks[1]) {
                state.tracks[1].extrapolateUnobserved();
              }
            }

          } else {
            // ── Path 2: Pose-Guided Tracking Frame (NanoDet Skipped! ~80% Speedup) ──
            for (const side of [0, 1]) {
              const trk = state.tracks[side];
              if (!trk || trk.lostCount > 3) continue;

              const pBox = trk.beginFramePredict();
              const cx = (pBox[0] + pBox[2]) / 2.0;
              const cy = (pBox[1] + pBox[3]) / 2.0;
              const bw = pBox[2] - pBox[0];
              const bh = pBox[3] - pBox[1];

              const cropBox = computeCropBox(cx, cy, bw, bh, 0.25);
              const cropTensor = extractCropTensor(sourceElement, cropBox.cx, cropBox.cy, cropBox.size, cfg.posePrecision);

              const tPose0 = performance.now();
              const poseOutputs = await state.poseSession.run({ [state.poseInputName]: cropTensor });
              totalPoseInfer += (performance.now() - tPose0);

              const poseResult = decodeRtmPoseOutputs(poseOutputs, cropBox.cx, cropBox.cy, cropBox.size);

              // Anatomical Clutter Gate
              if (poseResult.meanConf >= 0.20 && poseResult.validKptsCount >= 3) {
                const kptBox = computeKptBbox(poseResult.keypoints) || pBox;
                trk.applyObservation(kptBox, poseResult.keypoints, poseResult.handProb);
                finalDetections.push({
                  origBox: trk.smoothedBbox,
                  score: poseResult.meanConf,
                  classId: side,
                  keypoints: poseResult.keypoints
                });
              } else {
                trk.extrapolateUnobserved();
              }
            }
          }

          // One-Euro Temporal Smoothing
          const nowSec = performance.now() / 1000.0;
          for (const det of finalDetections) {
            if (state.filterEnabled) {
              det.keypoints = state.smoothers[det.classId].smooth(det.keypoints, nowSec);
            }
          }

          renderDetections(sourceElement, finalDetections);

          const e2e = performance.now() - frameStart;
          const totalInfer = tDet + totalPoseInfer;
          const totalProc = e2e - totalInfer;

          updateTelemetry(totalInfer, totalProc, e2e, finalDetections.length, tDet, totalPoseInfer);

        } else if (cfg.type === 'yolo' && state.yoloSession) {
          // ── Single-Stage YOLO Pipeline ──
          const tPre0 = performance.now();
          const prep = preprocessYoloLetterbox(sourceElement, cfg.precision);
          const tPre = performance.now() - tPre0;

          if (prep) {
            const tInfer0 = performance.now();
            const outputs = await state.yoloSession.run({ [state.yoloInputName]: prep.tensor });
            const tInfer = performance.now() - tInfer0;

            const tPost0 = performance.now();
            const rawOutput = outputs[state.yoloSession.outputNames[0]];
            let dets = decodeYoloPose(rawOutput, prep, 0.25, 0.45);

            const nowSec = performance.now() / 1000.0;
            for (const det of dets) {
              det.classId = determineGeometricChirality13kp(det.keypoints);
              if (state.filterEnabled) {
                det.keypoints = state.smoothers[det.classId].smooth(det.keypoints, nowSec);
              }
            }
            const tPost = performance.now() - tPost0;

            renderDetections(sourceElement, dets);

            const e2e = performance.now() - frameStart;
            updateTelemetry(tInfer, tPre + tPost, e2e, dets.length, tInfer, 0);
          }
        }
      } catch (err) {
        console.error('Frame inference error:', err);
        state.modelError = err.message || String(err);
        showBanner('error', `❌ Runtime WASM Failure: ${state.modelError}`);
        renderDetections(sourceElement, []);
      }
    } else {
      renderDetections(sourceElement, []);
    }

    if (state.isPlaying) {
      animFrameId = requestAnimationFrame(processFrame);
    }
  }

  // ── Telemetry Helper ────────────────────────────────────────────────────────
  function updateTelemetry(inferMs, procMs, e2eMs, detCount, stageAMs = 0, stageBMs = 0) {
    const now = performance.now();
    const delta = now - state.lastFrameTime;
    state.lastFrameTime = now;
    const curFps = delta > 0 ? 1000.0 / delta : 0;

    state.inferTimes.push(inferMs);
    state.prePostTimes.push(procMs);
    state.e2eTimes.push(e2eMs);
    state.stageATimes.push(stageAMs);
    state.stageBTimes.push(stageBMs);

    if (state.inferTimes.length > 30) {
      state.inferTimes.shift();
      state.prePostTimes.shift();
      state.e2eTimes.shift();
      state.stageATimes.shift();
      state.stageBTimes.shift();
    }

    const sortedInfer = [...state.inferTimes].sort((a, b) => a - b);
    const medianInfer = sortedInfer[Math.floor(sortedInfer.length / 2)];
    const p95Infer = sortedInfer[Math.floor(sortedInfer.length * 0.95)];
    const avgProc = state.prePostTimes.reduce((a, b) => a + b, 0) / state.prePostTimes.length;
    const avgE2E = state.e2eTimes.reduce((a, b) => a + b, 0) / state.e2eTimes.length;
    const avgStageA = state.stageATimes.reduce((a, b) => a + b, 0) / state.stageATimes.length;
    const avgStageB = state.stageBTimes.reduce((a, b) => a + b, 0) / state.stageBTimes.length;

    hudFps.textContent = curFps.toFixed(1);
    overlayFpsPill.textContent = `${curFps.toFixed(1)} FPS`;
    hudInferCur.textContent = inferMs.toFixed(1);
    hudInferMed.textContent = medianInfer.toFixed(1);
    hudInferP95.textContent = p95Infer.toFixed(1);
    hudProc.textContent = avgProc.toFixed(1);
    hudE2E.textContent = avgE2E.toFixed(1);
    hudDetections.textContent = detCount;

    if (hudStageSplit) {
      if (state.activeModelConfig && state.activeModelConfig.type === 'twostage') {
        hudStageSplit.textContent = `${avgStageA.toFixed(1)}ms / ${avgStageB.toFixed(1)}ms`;
      } else {
        hudStageSplit.textContent = `N/A (Single)`;
      }
    }

    if (hudTrackingMode) {
      if (state.activeModelConfig && state.activeModelConfig.type === 'twostage') {
        hudTrackingMode.textContent = state.heartbeatEnabled ? 'Heartbeat 5 (Pose-Guided)' : 'Every-Frame (Baseline)';
      } else {
        hudTrackingMode.textContent = 'Single-Stage YOLO';
      }
    }

    if (hudSkipRate) {
      if (state.activeModelConfig && state.activeModelConfig.type === 'twostage') {
        hudSkipRate.textContent = `${state.adaptiveDetector.getSkipRatePct().toFixed(1)}%`;
      } else {
        hudSkipRate.textContent = '0.0%';
      }
    }
  }

  // ── Camera and Video Source Handlers ────────────────────────────────────────
  let currentStream = null;

  async function startVideoSource() {
    stopCurrentStream();
    video.srcObject = null;
    video.src = 'video.mp4';
    video.loop = true;
    flipCameraBtn.style.display = 'none';
    restartBtn.style.display = 'flex';

    try {
      await video.play();
      state.isPlaying = true;
      updatePlayPauseUI(true);
      processFrame();
    } catch (err) {
      console.warn('Autoplay prevented, waiting for user interaction:', err);
      updatePlayPauseUI(false);
    }
  }

  async function startCameraSource() {
    stopCurrentStream();
    video.pause();
    video.removeAttribute('src');
    video.load();

    flipCameraBtn.style.display = 'flex';
    restartBtn.style.display = 'none';

    try {
      const constraints = {
        audio: false,
        video: {
          facingMode: state.facingMode,
          width: { ideal: 640 },
          height: { ideal: 480 }
        }
      };

      currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = currentStream;
      await video.play();
      state.isPlaying = true;
      updatePlayPauseUI(true);
      processFrame();
      showBanner('info', `Camera started (${state.facingMode})`);
      setTimeout(() => { if (!state.modelError) hideBanner(); }, 3000);
    } catch (err) {
      console.error('Camera access error:', err);
      showBanner('error', `Camera Error: ${err.name} - ${err.message}. Check camera permissions.`);
      sourceButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.source === 'video');
      });
      startVideoSource();
    }
  }

  function stopCurrentStream() {
    if (currentStream) {
      currentStream.getTracks().forEach(track => track.stop());
      currentStream = null;
    }
  }

  function updatePlayPauseUI(isPlaying) {
    state.isPlaying = isPlaying;
    playPauseIcon.textContent = isPlaying ? '⏸' : '▶';
    playPauseText.textContent = isPlaying ? 'Pause' : 'Play';
  }

  // ── Event Listeners ─────────────────────────────────────────────────────────
  allModelButtons.forEach(btn => {
    btn.addEventListener('click', async () => {
      if (btn.classList.contains('active')) return;
      allModelButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const modelKey = btn.dataset.model;
      state.inferTimes = [];
      state.prePostTimes = [];
      state.e2eTimes = [];
      state.stageATimes = [];
      state.stageBTimes = [];
      state.smoothers[0].reset();
      state.smoothers[1].reset();

      await loadModel(modelKey);
    });
  });

  sourceButtons.forEach(btn => {
    btn.addEventListener('click', async () => {
      if (btn.classList.contains('active')) return;
      sourceButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const sourceKey = btn.dataset.source;
      state.currentSource = sourceKey;

      if (sourceKey === 'camera') {
        await startCameraSource();
      } else {
        await startVideoSource();
      }
    });
  });

  if (heartbeatToggle) {
    heartbeatToggle.addEventListener('change', () => {
      state.heartbeatEnabled = heartbeatToggle.checked;
      state.adaptiveDetector.reset();
    });
  }

  filterToggle.addEventListener('change', () => {
    state.filterEnabled = filterToggle.checked;
    state.smoothers[0].reset();
    state.smoothers[1].reset();
  });

  playPauseBtn.addEventListener('click', () => {
    if (state.isPlaying) {
      video.pause();
      updatePlayPauseUI(false);
    } else {
      video.play().then(() => {
        updatePlayPauseUI(true);
        processFrame();
      }).catch(err => {
        console.error('Play error:', err);
      });
    }
  });

  flipCameraBtn.addEventListener('click', async () => {
    state.facingMode = state.facingMode === 'environment' ? 'user' : 'environment';
    await startCameraSource();
  });

  restartBtn.addEventListener('click', () => {
    if (state.currentSource === 'video') {
      video.currentTime = 0;
      if (!state.isPlaying) {
        video.play();
        updatePlayPauseUI(true);
        processFrame();
      }
    }
  });

  // ── Initial Start ───────────────────────────────────────────────────────────
  window.addEventListener('DOMContentLoaded', async () => {
    // 1. Load default model (twostage_fp32_fp32 - Pure FP32 Reference)
    await loadModel('twostage_fp32_fp32');
    // 2. Start video
    await startVideoSource();
  });

})();
