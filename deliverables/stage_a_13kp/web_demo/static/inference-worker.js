/**
 * inference-worker.js
 * Dedicated Web Worker for background ONNX Runtime inference
 * Automatic execution provider selection: WebGL (GPU) -> WASM SIMD Multi-thread -> WASM Single-thread
 * Keeps the main UI thread completely free for 60 FPS camera preview
 */

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.min.js');

let session = null;
let currentPrecision = 'int8';
let activeBackend = 'WASM';
let activeThreads = 1;

let CONFIG = {
    inputSize: 256,
    numAnchors: 1344, // Default 256x256 (1344). 192x192 has 756. 160x160 has 525.
    confThreshold: 0.45,
    iouThreshold: 0.50,
    maxDetections: 2,
    keypointVisThreshold: 0.3
};

const KEYPOINT_NAMES = [
    "toe_ground", "heel_back", "heel_ground", "ball_medial", "ball_lateral",
    "instep_top", "arch_medial", "midfoot_lateral", "malleolus_medial",
    "malleolus_lateral", "toe_tip", "throat", "achilles"
];

// Preallocated OffscreenCanvas and input tensor buffer
let offscreenCanvas = new OffscreenCanvas(CONFIG.inputSize, CONFIG.inputSize);
let offCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
let inputTensorBuffer = new Float32Array(3 * CONFIG.inputSize * CONFIG.inputSize);

function updateInputSize(newSize, newAnchors) {
    if (CONFIG.inputSize !== newSize || CONFIG.numAnchors !== newAnchors) {
        CONFIG.inputSize = newSize;
        CONFIG.numAnchors = newAnchors;
        offscreenCanvas = new OffscreenCanvas(newSize, newSize);
        offCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
        inputTensorBuffer = new Float32Array(3 * newSize * newSize);
        console.log(`[Worker] Adapted canvas & buffers to ${newSize}x${newSize} (${newAnchors} anchors)`);
    }
}

function sigmoid(x) {
    return 1.0 / (1.0 + Math.exp(-x));
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

/**
 * Initialize ONNX Runtime session with automatic backend fallback
 */
async function loadModel(precisionKey, modelBuffer, modelMeta) {
    currentPrecision = precisionKey;
    
    // Release previous session if exists
    if (session) {
        try { await session.release(); } catch (e) {}
        session = null;
    }

    // Configure WASM paths and threading
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';
    ort.env.wasm.simd = true;

    if (self.crossOriginIsolated) {
        activeThreads = Math.min(4, navigator.hardwareConcurrency || 4);
        ort.env.wasm.numThreads = activeThreads;
        console.log(`[Worker] Unlocked multi-threading: ${activeThreads} threads via SharedArrayBuffer`);
    } else {
        activeThreads = 1;
        ort.env.wasm.numThreads = 1;
        console.log(`[Worker] Single-thread mode`);
    }

    if (precisionKey === 'int8' || precisionKey === 'turbo192' || precisionKey.includes('int8')) {
        // INT8 dynamic quantization runs fastest on multi-threaded CPU SIMD
        // WebGPU/WebGL lack native integer convolution kernels and cause high overhead
        session = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        activeBackend = self.crossOriginIsolated 
            ? `WASM (${activeThreads}T Multi-Core)` 
            : `WASM (1T Single-Core)`;
    } else if (precisionKey === 'fp16') {
        // Try WebGPU first, then WebGL, then fail gracefully
        let created = false;
        const providersToTry = [
            { name: 'webgpu', label: 'WebGPU (GPU)' },
            { name: 'webgl', label: 'WebGL (GPU)' }
        ];

        for (const ep of providersToTry) {
            try {
                session = await ort.InferenceSession.create(modelBuffer, {
                    executionProviders: [ep.name],
                    graphOptimizationLevel: 'all'
                });
                activeBackend = ep.label;
                created = true;
                break;
            } catch (err) {
                console.warn(`[Worker] ${ep.label} failed for FP16:`, err);
            }
        }

        if (!created) {
            throw new Error('FP16 requires WebGPU or WebGL GPU acceleration. Please select INT8 or FP32.');
        }
    } else {
        // FP32: try WebGPU (GPU) -> WebGL (GPU) -> WASM MT/ST
        let created = false;
        const providersToTry = [
            { name: 'webgpu', label: 'WebGPU (GPU)' },
            { name: 'webgl', label: 'WebGL (GPU)' },
            { name: 'wasm', label: `WASM (${activeThreads}T CPU)` }
        ];

        for (const ep of providersToTry) {
            try {
                session = await ort.InferenceSession.create(modelBuffer, {
                    executionProviders: [ep.name],
                    graphOptimizationLevel: 'all'
                });
                activeBackend = ep.label;
                created = true;
                console.log(`[Worker] Activated ${ep.label} for FP32!`);
                break;
            } catch (err) {
                console.warn(`[Worker] ${ep.label} failed for FP32:`, err.message || err);
            }
        }

        if (!created) {
            session = await ort.InferenceSession.create(modelBuffer, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'basic'
            });
            activeBackend = self.crossOriginIsolated 
                ? `WASM (${activeThreads}T Multi-Core)` 
                : `WASM (1T Single-Core)`;
        }
    }

    if (modelMeta && modelMeta.inputSize) {
        updateInputSize(modelMeta.inputSize, modelMeta.numAnchors);
    }

    return {
        precisionKey,
        backend: activeBackend,
        threads: activeThreads,
        isolated: self.crossOriginIsolated
    };
}

/**
 * Preprocess ImageBitmap into 256x256 model input
 */
function preprocess(imageBitmap) {
    const vW = imageBitmap.width;
    const vH = imageBitmap.height;
    const S = CONFIG.inputSize;

    const scale = Math.min(S / vW, S / vH);
    const nW = Math.round(vW * scale);
    const nH = Math.round(vH * scale);
    const dw = Math.floor((S - nW) / 2);
    const dh = Math.floor((S - nH) / 2);

    // Fill gray padding
    offCtx.fillStyle = 'rgb(114, 114, 114)';
    offCtx.fillRect(0, 0, S, S);

    // Draw scaled frame
    offCtx.drawImage(imageBitmap, 0, 0, vW, vH, dw, dh, nW, nH);

    const imgData = offCtx.getImageData(0, 0, S, S);
    const pixels = imgData.data;
    const channelSize = S * S;
    const inv255 = 1.0 / 255.0;

    for (let i = 0; i < channelSize; i++) {
        const p = i * 4;
        inputTensorBuffer[i] = pixels[p] * inv255;
        inputTensorBuffer[channelSize + i] = pixels[p + 1] * inv255;
        inputTensorBuffer[2 * channelSize + i] = pixels[p + 2] * inv255;
    }

    const inputName = session.inputNames[0];
    const feeds = {};
    feeds[inputName] = new ort.Tensor('float32', inputTensorBuffer, [1, 3, S, S]);

    return { feeds, scale, pad: [dw, dh] };
}

/**
 * Postprocess raw model output into 2 clean feet with 13 keypoints
 */
function postprocess(output, scale, pad) {
    const data = output.data;
    const detections = [];
    const numAnchors = CONFIG.numAnchors; // 1344 for 256x256

    for (let i = 0; i < numAnchors; i++) {
        const cx = data[i];
        const cy = data[numAnchors + i];
        const w = data[2 * numAnchors + i];
        const h = data[3 * numAnchors + i];

        const sLeft = sigmoid(data[4 * numAnchors + i]);
        const sRight = sigmoid(data[5 * numAnchors + i]);
        const maxScore = Math.max(sLeft, sRight);

        if (maxScore < CONFIG.confThreshold) continue;

        const classId = sLeft >= sRight ? 0 : 1;
        const x1 = (cx - w / 2 - pad[0]) / scale;
        const y1 = (cy - h / 2 - pad[1]) / scale;
        const x2 = (cx + w / 2 - pad[0]) / scale;
        const y2 = (cy + h / 2 - pad[1]) / scale;

        const keypoints = [];
        for (let k = 0; k < 13; k++) {
            const kx = (data[(6 + k * 3) * numAnchors + i] - pad[0]) / scale;
            const ky = (data[(6 + k * 3 + 1) * numAnchors + i] - pad[1]) / scale;
            const kv = sigmoid(data[(6 + k * 3 + 2) * numAnchors + i]);
            keypoints.push({ index: k, name: KEYPOINT_NAMES[k], x: kx, y: ky, visibility: kv });
        }

        detections.push({
            bbox: [x1, y1, x2, y2],
            classId: classId,
            className: classId === 0 ? 'left_foot' : 'right_foot',
            confidence: maxScore,
            keypoints: keypoints
        });
    }

    detections.sort((a, b) => b.confidence - a.confidence);

    const kept = [];
    for (const det of detections) {
        let isDup = false;
        for (const k of kept) {
            if (computeIoU(det.bbox, k.bbox) > CONFIG.iouThreshold) {
                isDup = true;
                break;
            }
        }
        if (!isDup) kept.push(det);
        if (kept.length >= CONFIG.maxDetections) break;
    }

    return kept;
}

/**
 * Handle incoming messages from UI thread
 */
self.onmessage = async function(e) {
    const msg = e.data;

    if (msg.type === 'LOAD_MODEL') {
        try {
            const res = await loadModel(msg.precisionKey, msg.modelBuffer, msg.modelMeta);
            self.postMessage({
                type: 'MODEL_LOADED',
                precisionKey: res.precisionKey,
                backend: res.backend,
                threads: res.threads
            });
        } catch (err) {
            self.postMessage({
                type: 'ERROR',
                message: (err && (err.message || String(err))) || 'Failed to load model'
            });
        }
    } else if (msg.type === 'RUN_INFERENCE') {
        if (!session) {
            if (msg.imageBitmap) msg.imageBitmap.close();
            return;
        }

        const t0 = performance.now();
        try {
            const tStartPre = performance.now();
            const { feeds, scale, pad } = preprocess(msg.imageBitmap);
            msg.imageBitmap.close(); // Clean up image bitmap immediately
            const tPre = performance.now() - tStartPre;

            const tStartInf = performance.now();
            const results = await session.run(feeds);
            const tInf = performance.now() - tStartInf;

            const tStartPost = performance.now();
            const output = results[session.outputNames[0]];
            const detections = postprocess(output, scale, pad);
            const tPost = performance.now() - tStartPost;

            const latency = performance.now() - t0;

            self.postMessage({
                type: 'INFERENCE_RESULT',
                detections: detections,
                latency: latency,
                breakdown: {
                    pre: Math.round(tPre),
                    inf: Math.round(tInf),
                    post: Math.round(tPost)
                },
                id: msg.id
            });
        } catch (err) {
            if (msg.imageBitmap) msg.imageBitmap.close();
            console.error('[Worker Inference Error]', err);
            self.postMessage({
                type: 'INFERENCE_ERROR',
                message: err.message
            });
        }
    }
};
