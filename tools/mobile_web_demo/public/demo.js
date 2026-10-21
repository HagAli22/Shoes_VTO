// demo.js - Shoes VTO Stage-A Mobile Benchmark (WASM Only)

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

  // ── Keypoint Schema & Connections ──────────────────────────────────────────
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
    bbox: '#22c55e',      // Bright Green
    skeleton: '#16a34a',
    kptVis: '#4ade80',
    kptOcc: '#86efac',
    label: 'Left Foot'
  };

  const COLOR_RIGHT = {
    bbox: '#38bdf8',      // Bright Cyan/Blue
    skeleton: '#0284c7',
    kptVis: '#67e8f9',
    kptOcc: '#a5f3fc',
    label: 'Right Foot'
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

  // ── State Management ────────────────────────────────────────────────────────
  const state = {
    currentModel: 'int8', // 'int8' or 'fp16'
    currentSource: 'video', // 'video' or 'camera'
    filterEnabled: true,
    facingMode: 'environment', // 'environment' or 'user'
    isPlaying: false,
    session: null,
    sessionInputName: null,
    sessionInputType: null,
    loadStartTime: null,
    loadDuration: null,
    modelError: null,
    smoothers: {
      0: new KeypointSmoother(13), // Left
      1: new KeypointSmoother(13)  // Right
    },
    // Telemetry rolling window
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
  const modelButtons = document.querySelectorAll('#modelControl button');
  const sourceButtons = document.querySelectorAll('#sourceControl button');
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
  const hudDetections = document.getElementById('hudDetections');
  const overlayModelPill = document.getElementById('overlayModelPill');
  const overlayFpsPill = document.getElementById('overlayFpsPill');

  // ── Offscreen Preprocessing Canvas ──────────────────────────────────────────
  const offscreenCanvas = document.createElement('canvas');
  offscreenCanvas.width = 256;
  offscreenCanvas.height = 256;
  const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });

  // ── Helper: Float32 to Float16 (Uint16) converter ───────────────────────────
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

  // ── Diagnostic Banner Helper ────────────────────────────────────────────────
  function showBanner(type, message) {
    banner.className = `banner ${type}`;
    banner.textContent = message;
    banner.style.display = 'block';
  }

  function hideBanner() {
    banner.style.display = 'none';
  }

  // ── Model Loader ────────────────────────────────────────────────────────────
  async function loadModel(modelKey) {
    state.currentModel = modelKey;
    state.modelError = null;
    overlayModelPill.textContent = `${modelKey.toUpperCase()} WASM`;

    const modelPath = modelKey === 'int8'
      ? 'models/stage-a-256-13kp-int8.onnx'
      : 'models/stage-a-256-13kp-fp16.onnx';

    showBanner('info', `Loading ${modelKey.toUpperCase()} model (${modelPath}) via WASM backend...`);

    const t0 = performance.now();
    try {
      if (typeof ort === 'undefined') {
        throw new Error('ONNX Runtime Web library failed to load.');
      }

      const sessionOptions = {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      };

      state.session = await ort.InferenceSession.create(modelPath, sessionOptions);
      state.sessionInputName = state.session.inputNames[0];
      
      // Determine input tensor type
      // ORT Web exposes input metadata
      const inputMeta = state.session.inputs ? state.session.inputs[0] : null;
      state.sessionInputType = inputMeta ? inputMeta.type : (modelKey === 'fp16' ? 'float16' : 'float32');

      const dt = (performance.now() - t0).toFixed(0);
      state.loadDuration = dt;
      hudLoadTime.textContent = `${dt} ms`;

      showBanner('info', `✓ ${modelKey.toUpperCase()} loaded successfully in ${dt} ms. Execution Provider: WASM (Single-Threaded).`);
      setTimeout(() => {
        if (!state.modelError) hideBanner();
      }, 4000);

      // Warm up model with a dummy tensor
      await warmUpModel();

    } catch (err) {
      console.error('Session creation failed:', err);
      state.modelError = err.message || String(err);
      state.session = null;
      showBanner('error', `❌ ${modelKey.toUpperCase()} WASM Error: ${state.modelError}\n(ONNX Runtime Web WASM does not support this model precision or operator set)`);
    }
  }

  async function warmUpModel() {
    if (!state.session) return;
    try {
      const isFp16 = state.currentModel === 'fp16';
      let dummyTensor;
      if (isFp16) {
        const totalElements = 1 * 3 * 256 * 256;
        const f16Data = (typeof Float16Array !== 'undefined')
          ? new Float16Array(totalElements)
          : new Uint16Array(totalElements);
        dummyTensor = new ort.Tensor('float16', f16Data, [1, 3, 256, 256]);
      } else {
        const f32 = new Float32Array(1 * 3 * 256 * 256);
        dummyTensor = new ort.Tensor('float32', f32, [1, 3, 256, 256]);
      }
      await state.session.run({ [state.sessionInputName]: dummyTensor });
      console.log('Warmup inference completed successfully.');
    } catch (err) {
      console.warn('Warmup error (operator or precision incompatibility):', err);
      state.modelError = err.message || String(err);
      showBanner('error', `❌ ${state.currentModel.toUpperCase()} WASM Warm-up Failed: ${state.modelError}`);
    }
  }

  // ── Input Preprocessing (Letterbox 256x256 with fill 114) ───────────────────
  function preprocessFrame(sourceElement) {
    const srcW = sourceElement.videoWidth || sourceElement.width;
    const srcH = sourceElement.videoHeight || sourceElement.height;
    if (!srcW || !srcH) return null;

    const targetSize = 256;
    const ratio = Math.min(targetSize / srcW, targetSize / srcH);
    const nw = Math.round(srcW * ratio);
    const nh = Math.round(srcH * ratio);
    const padX = (targetSize - nw) / 2;
    const padY = (targetSize - nh) / 2;

    // Fill offscreen canvas with 114
    offscreenCtx.fillStyle = 'rgb(114, 114, 114)';
    offscreenCtx.fillRect(0, 0, targetSize, targetSize);

    // Draw source video centered
    offscreenCtx.drawImage(sourceElement, 0, 0, srcW, srcH, padX, padY, nw, nh);

    const imgData = offscreenCtx.getImageData(0, 0, targetSize, targetSize).data;
    const isFp16 = state.currentModel === 'fp16';

    const numPixels = targetSize * targetSize;
    let tensorData;
    let tensorType;

    if (isFp16) {
      tensorType = 'float16';
      const rOffset = 0;
      const gOffset = numPixels;
      const bOffset = numPixels * 2;

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
    } else {
      tensorType = 'float32';
      tensorData = new Float32Array(3 * numPixels);
      const rOffset = 0;
      const gOffset = numPixels;
      const bOffset = numPixels * 2;

      for (let i = 0; i < numPixels; i++) {
        const srcIdx = i * 4;
        tensorData[rOffset + i] = imgData[srcIdx] / 255.0;
        tensorData[gOffset + i] = imgData[srcIdx + 1] / 255.0;
        tensorData[bOffset + i] = imgData[srcIdx + 2] / 255.0;
      }
    }

    const inputTensor = new ort.Tensor(tensorType, tensorData, [1, 3, targetSize, targetSize]);
    return {
      tensor: inputTensor,
      ratio,
      padX,
      padY,
      srcW,
      srcH
    };
  }

  // ── Postprocessing: Decode Output [1, 45, 1344] ─────────────────────────────
  function sigmoid(v) {
    return 1.0 / (1.0 + Math.exp(-v));
  }

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

  function decodeYoloPose(rawOutput, prepMeta, confThresh = 0.25, nmsThresh = 0.45) {
    // rawOutput shape is [1, 45, 1344]
    // In flat array: index [c * 1344 + a]
    const numChannels = 45;
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

    // NMS
    const boxes = candidates.map(c => c.boxPx);
    const scores = candidates.map(c => c.score);
    const keepIndices = nms(boxes, scores, nmsThresh);

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

    for (const det of detections) {
      const clsId = det.classId;
      const cfg = clsId === 0 ? COLOR_LEFT : COLOR_RIGHT;
      const b = det.origBox;
      const kpts = det.keypoints;

      // 1. Bounding box
      ctx.strokeStyle = cfg.bbox;
      ctx.lineWidth = 3;
      ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);

      // Badge
      ctx.fillStyle = cfg.bbox;
      const badgeText = `${cfg.label} ${(det.score * 100).toFixed(0)}%`;
      ctx.font = 'bold 16px sans-serif';
      const textWidth = ctx.measureText(badgeText).width;
      ctx.fillRect(b[0], Math.max(0, b[1] - 24), textWidth + 12, 24);
      ctx.fillStyle = '#0f172a';
      ctx.fillText(badgeText, b[0] + 6, Math.max(18, b[1] - 6));

      // 2. Skeleton
      ctx.strokeStyle = cfg.skeleton;
      ctx.lineWidth = 2.5;
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
      for (const kp of kpts) {
        if (kp.visibility < 0.15) continue;
        ctx.beginPath();
        if (kp.visibility >= 0.35) {
          // Solid dot
          ctx.fillStyle = cfg.kptVis;
          ctx.arc(kp.x, kp.y, 5, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = '#000';
          ctx.lineWidth = 1;
          ctx.stroke();
        } else {
          // Hollow ring for occluded
          ctx.strokeStyle = cfg.kptOcc;
          ctx.lineWidth = 2;
          ctx.arc(kp.x, kp.y, 4, 0, Math.PI * 2);
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

    if (state.session && !state.modelError) {
      try {
        // Step 1: Preprocess
        const tPre0 = performance.now();
        const prep = preprocessFrame(sourceElement);
        const tPre = performance.now() - tPre0;

        if (prep) {
          // Step 2: Model Inference
          const tInfer0 = performance.now();
          const outputs = await state.session.run({ [state.sessionInputName]: prep.tensor });
          const tInfer = performance.now() - tInfer0;

          // Step 3: Postprocess & Chirality & Smoothing
          const tPost0 = performance.now();
          const rawOutput = outputs[state.session.outputNames[0]];
          let dets = decodeYoloPose(rawOutput, prep, 0.25, 0.45);

          const nowSec = performance.now() / 1000.0;
          for (const det of dets) {
            det.classId = determineGeometricChirality13kp(det.keypoints);
            if (state.filterEnabled) {
              det.keypoints = state.smoothers[det.classId].smooth(det.keypoints, nowSec);
            }
          }
          const tPost = performance.now() - tPost0;

          // Step 4: Render
          renderDetections(sourceElement, dets);

          const e2e = performance.now() - frameStart;

          // Telemetry updates
          updateTelemetry(tInfer, tPre + tPost, e2e, dets.length);
        }
      } catch (err) {
        console.error('Frame inference error:', err);
        state.modelError = err.message || String(err);
        showBanner('error', `❌ Runtime WASM Failure: ${state.modelError}`);
        // Render raw video frame without detections
        renderDetections(sourceElement, []);
      }
    } else {
      // No session or model error: just show video feed
      renderDetections(sourceElement, []);
    }

    if (state.isPlaying) {
      animFrameId = requestAnimationFrame(processFrame);
    }
  }

  // ── Telemetry Helper ────────────────────────────────────────────────────────
  function updateTelemetry(inferMs, procMs, e2eMs, detCount) {
    const now = performance.now();
    const delta = now - state.lastFrameTime;
    state.lastFrameTime = now;
    const curFps = delta > 0 ? 1000.0 / delta : 0;

    // Rolling windows of 30 frames
    state.inferTimes.push(inferMs);
    state.prePostTimes.push(procMs);
    state.e2eTimes.push(e2eMs);
    if (state.inferTimes.length > 30) {
      state.inferTimes.shift();
      state.prePostTimes.shift();
      state.e2eTimes.shift();
    }

    // Stats calculations
    const sortedInfer = [...state.inferTimes].sort((a, b) => a - b);
    const medianInfer = sortedInfer[Math.floor(sortedInfer.length / 2)];
    const p95Infer = sortedInfer[Math.floor(sortedInfer.length * 0.95)];
    const avgProc = state.prePostTimes.reduce((a, b) => a + b, 0) / state.prePostTimes.length;
    const avgE2E = state.e2eTimes.reduce((a, b) => a + b, 0) / state.e2eTimes.length;

    hudFps.textContent = curFps.toFixed(1);
    overlayFpsPill.textContent = `${curFps.toFixed(1)} FPS`;
    hudInferCur.textContent = inferMs.toFixed(1);
    hudInferMed.textContent = medianInfer.toFixed(1);
    hudInferP95.textContent = p95Infer.toFixed(1);
    hudProc.textContent = avgProc.toFixed(1);
    hudE2E.textContent = avgE2E.toFixed(1);
    hudDetections.textContent = detCount;
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
      // Revert to video
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
  modelButtons.forEach(btn => {
    btn.addEventListener('click', async () => {
      if (btn.classList.contains('active')) return;
      modelButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const modelKey = btn.dataset.model;
      state.inferTimes = [];
      state.prePostTimes = [];
      state.e2eTimes = [];
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
    // 1. Load default model (INT8)
    await loadModel('int8');
    // 2. Start video
    await startVideoSource();
  });

})();

