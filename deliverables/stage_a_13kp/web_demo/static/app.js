/**
 * app.js
 * Main UI thread controller
 * Decoupled 60 FPS camera display with zero-copy Web Worker background inference
 */

const MODELS = {
    int8: {
        id: 'int8',
        name: 'INT8 (256×256)',
        filename: 'stage-a-256-13kp-int8.onnx',
        sizeMB: '3.30 MB',
        expectedBytes: 3460553,
        inputSize: 256,
        numAnchors: 1344
    },
    turbo192: {
        id: 'turbo192',
        name: '🚀 Turbo 192 (Fastest)',
        filename: 'stage-a-192-13kp-int8.onnx',
        sizeMB: '3.28 MB',
        expectedBytes: 3438654,
        inputSize: 192,
        numAnchors: 756
    },
    fp16: {
        id: 'fp16',
        name: 'FP16 (256×256)',
        filename: 'stage-a-256-13kp-fp16.onnx',
        sizeMB: '6.22 MB',
        expectedBytes: 6525939,
        inputSize: 256,
        numAnchors: 1344
    },
    orig320: {
        id: 'orig320',
        name: '🏛️ Original 320 (Unquantized)',
        filename: 'stage-a-320-13kp-v7.onnx',
        sizeMB: '12.39 MB',
        expectedBytes: 12990118,
        inputSize: 320,
        numAnchors: 2100
    },
    fp32: {
        id: 'fp32',
        name: 'FP32 (256×256)',
        filename: 'stage-a-256-13kp-fp32.onnx',
        sizeMB: '12.36 MB',
        expectedBytes: 12965875,
        inputSize: 256,
        numAnchors: 1344
    }
};

const SKELETON_PAIRS = [
    [2, 1], [2, 6], [6, 3], [3, 10], [10, 0], [0, 4], [4, 7], [7, 2],
    [10, 5], [5, 11], [8, 11], [9, 11], [8, 12], [9, 12], [1, 12]
];

const COLORS = {
    left: { bbox: [56, 189, 248], skeleton: [14, 165, 233], kpt: [2, 132, 199] },
    right: { bbox: [244, 114, 182], skeleton: [236, 72, 153], kpt: [219, 39, 119] }
};

// Global state
let worker = null;
let currentPrecision = 'int8';
let video = null;
let canvas = null;
let ctx = null;

let isCameraRunning = false;
let isWorkerBusy = false;
let isSwitching = false;
let frameId = 0;

let latestDetections = [];
let stats = {
    latency: 0,
    fps: 0,
    camFps: 60,
    detections: 0
};

const benchmarkHistory = {
    int8: { sum: 0, count: 0, avg: 0 },
    turbo192: { sum: 0, count: 0, avg: 0 },
    fp16: { sum: 0, count: 0, avg: 0 },
    orig320: { sum: 0, count: 0, avg: 0 },
    fp32: { sum: 0, count: 0, avg: 0 }
};

/**
 * Fetch model bytes with streaming progress bar
 */
async function fetchModelBuffer(filename, expectedBytes) {
    const urls = [
        `../models/${filename}`,
        `/models/${filename}`,
        `models/${filename}`
    ];

    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    if (progressContainer) progressContainer.style.display = 'block';

    let lastError = null;
    for (const url of urls) {
        try {
            console.log(`[Main] Downloading ${url}...`);
            const response = await fetch(url, { cache: 'no-cache' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const totalBytes = parseInt(response.headers.get('Content-Length'), 10) || expectedBytes;

            if (!response.body || !response.body.getReader) {
                const buffer = await response.arrayBuffer();
                if (progressBar) progressBar.style.width = '100%';
                return buffer;
            }

            const reader = response.body.getReader();
            let receivedBytes = 0;
            const chunks = [];

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
                receivedBytes += value.length;

                const pct = Math.min(100, Math.round((receivedBytes / totalBytes) * 100));
                const mbR = (receivedBytes / (1024 * 1024)).toFixed(1);
                const mbT = (totalBytes / (1024 * 1024)).toFixed(1);

                if (progressBar) progressBar.style.width = `${pct}%`;
                updateStatus(`Downloading ${MODELS[currentPrecision].name}: ${pct}% (${mbR}/${mbT} MB)...`, 'info');
            }

            const combined = new Uint8Array(receivedBytes);
            let offset = 0;
            for (const chunk of chunks) {
                combined.set(chunk, offset);
                offset += chunk.length;
            }
            return combined.buffer;
        } catch (err) {
            console.warn(`[Main] URL ${url} failed:`, err);
            lastError = err;
        }
    }
    throw new Error(`Failed to load ${filename}: ${lastError ? lastError.message : 'Not found'}`);
}

/**
 * Switch & load model precision
 */
async function loadPrecisionModel(precisionKey) {
    if (isSwitching) return;
    isSwitching = true;
    currentPrecision = precisionKey;
    const modelMeta = MODELS[precisionKey];

    // Update active UI styles
    ['int8', 'turbo192', 'fp16', 'orig320', 'fp32'].forEach(k => {
        const btn = document.getElementById(`btn-${k}`);
        const row = document.getElementById(`row-${k}`);
        if (btn) btn.classList.toggle('active', k === precisionKey);
        if (row) row.classList.toggle('active-row', k === precisionKey);
    });

    const activeLabel = document.getElementById('activeModelName');
    if (activeLabel) activeLabel.textContent = `${modelMeta.name} Active`;

    try {
        updateStatus(`Downloading ${modelMeta.name} (${modelMeta.sizeMB})...`, 'info');
        const buffer = await fetchModelBuffer(modelMeta.filename, modelMeta.expectedBytes);

        updateStatus(`Initializing ${modelMeta.name} with GPU/SIMD acceleration...`, 'info');
        worker.postMessage({
            type: 'LOAD_MODEL',
            precisionKey: precisionKey,
            modelBuffer: buffer,
            modelMeta: {
                inputSize: modelMeta.inputSize || 256,
                numAnchors: modelMeta.numAnchors || 1344
            }
        }, [buffer]); // Zero-copy transfer to worker

    } catch (err) {
        console.error('[Load Error]', err);
        const progressContainer = document.getElementById('progress-container');
        if (progressContainer) progressContainer.style.display = 'none';
        const msg = (err && (err.message || String(err))) || 'Download failed';
        updateStatus(`Failed to load ${modelMeta.name}: ${msg}`, 'error');
        isSwitching = false;
    }
}

window.switchPrecision = async function(precisionKey) {
    if (precisionKey === currentPrecision && !isSwitching) return;
    await loadPrecisionModel(precisionKey);
};

/**
 * Dispatch next camera frame to worker for inference
 */
async function scheduleNextFrame() {
    if (!isCameraRunning || isWorkerBusy || isSwitching) return;
    if (!video || video.paused || video.ended) return;

    try {
        isWorkerBusy = true;
        // Hardware zero-copy frame capture
        const bitmap = await createImageBitmap(video);

        worker.postMessage({
            type: 'RUN_INFERENCE',
            imageBitmap: bitmap,
            id: ++frameId
        }, [bitmap]); // Transferable ownership to worker
    } catch (e) {
        isWorkerBusy = false;
    }
}

/**
 * Draw detections on overlay canvas
 */
function renderOverlay() {
    if (!ctx || !canvas) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const det of latestDetections) {
        const color = det.classId === 0 ? COLORS.left : COLORS.right;
        const [x1, y1, x2, y2] = det.bbox.map(v => Math.round(v));

        // Bounding box
        ctx.strokeStyle = `rgb(${color.bbox.join(',')})`;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Label
        const label = `${det.className}: ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 15px Arial';
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = `rgb(${color.bbox.join(',')})`;
        ctx.fillRect(x1, Math.max(0, y1 - 24), textWidth + 10, 24);
        ctx.fillStyle = 'white';
        ctx.fillText(label, x1 + 5, Math.max(16, y1 - 7));

        // Skeleton
        const kpts = det.keypoints;
        for (const [k1, k2] of SKELETON_PAIRS) {
            if (kpts[k1] && kpts[k2] && kpts[k1].visibility > 0.3 && kpts[k2].visibility > 0.3) {
                ctx.beginPath();
                ctx.moveTo(kpts[k1].x, kpts[k1].y);
                ctx.lineTo(kpts[k2].x, kpts[k2].y);
                ctx.strokeStyle = `rgb(${color.skeleton.join(',')})`;
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        }

        // Keypoints
        for (const kp of kpts) {
            if (kp.visibility > 0.3) {
                ctx.beginPath();
                ctx.arc(kp.x, kp.y, 4, 0, 2 * Math.PI);
                ctx.fillStyle = `rgb(${color.kpt.join(',')})`;
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }
    }
}

/**
 * Smooth 60 FPS camera display loop (unblocked by inference)
 */
let lastCamTime = performance.now();
let camFrameCount = 0;
function displayLoop() {
    if (!isCameraRunning) return;

    renderOverlay();

    camFrameCount++;
    const now = performance.now();
    if (now - lastCamTime >= 1000) {
        stats.camFps = Math.round((camFrameCount * 1000) / (now - lastCamTime));
        camFrameCount = 0;
        lastCamTime = now;
        const camEl = document.getElementById('cameraFps');
        if (camEl) camEl.textContent = stats.camFps;
    }

    requestAnimationFrame(displayLoop);
}

/**
 * Update numbers and comparison table
 */
function updateStatsDisplay(latencyMs, breakdown) {
    const latEl = document.getElementById('latency');
    const fpsEl = document.getElementById('fps');
    const detEl = document.getElementById('detections');

    if (latEl) latEl.textContent = Math.round(latencyMs);
    if (fpsEl) fpsEl.textContent = (1000 / latencyMs).toFixed(1);
    if (detEl) detEl.textContent = stats.detections;

    const hist = benchmarkHistory[currentPrecision];
    hist.sum += latencyMs;
    hist.count++;
    hist.avg = hist.sum / hist.count;

    const rowLat = document.getElementById(`lat-${currentPrecision}`);
    const rowFps = document.getElementById(`fps-${currentPrecision}`);
    if (rowLat) rowLat.textContent = `${Math.round(hist.avg)} ms`;
    if (rowFps) rowFps.textContent = `${(1000 / hist.avg).toFixed(1)} fps`;

    if (breakdown) {
        const bdText = `Pre: ${breakdown.pre}ms | Inference: ${breakdown.inf}ms | Post: ${breakdown.post}ms`;
        const activeLabel = document.getElementById('activeModelName');
        if (activeLabel) activeLabel.textContent = `${MODELS[currentPrecision].name} (${bdText})`;
    }
}

function updateStatus(message, type = 'info') {
    const statusEl = document.getElementById('status');
    if (statusEl) {
        statusEl.textContent = message;
        statusEl.className = `status ${type}`;
    }
}

/**
 * Camera handlers
 */
async function startCamera() {
    try {
        updateStatus('Opening camera at 60 FPS...', 'info');

        let stream = null;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: 'environment' },
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 60, min: 30 }
                }
            });
        } catch (e) {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
        }

        video.srcObject = stream;
        await video.play();

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        isCameraRunning = true;
        document.getElementById('startCamera').style.display = 'none';
        document.getElementById('stopCamera').style.display = 'block';

        updateStatus(`Camera active at 60 FPS — Running ${MODELS[currentPrecision].name}!`, 'success');

        // Start independent 60 FPS UI loop
        requestAnimationFrame(displayLoop);

        // Trigger worker inference pipeline
        scheduleNextFrame();

    } catch (err) {
        console.error('[Camera Error]', err);
        updateStatus(`Camera error: ${err.message}`, 'error');
    }
}

function stopCamera() {
    isCameraRunning = false;
    isWorkerBusy = false;
    if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }
    latestDetections = [];
    if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);

    document.getElementById('startCamera').style.display = 'block';
    document.getElementById('stopCamera').style.display = 'none';
    updateStatus('Camera stopped.', 'info');
}

/**
 * Initialize Web Worker and start default model
 */
function initWorker() {
    worker = new Worker('../static/inference-worker.js?v=256_v7');

    worker.onmessage = function(e) {
        const msg = e.data;

        if (msg.type === 'MODEL_LOADED') {
            isSwitching = false;
            const progressContainer = document.getElementById('progress-container');
            if (progressContainer) progressContainer.style.display = 'none';

            const badge = document.getElementById('backendBadge');
            if (badge) badge.textContent = msg.backend;

            updateStatus(`Ready! Active: ${MODELS[msg.precisionKey].name} (${msg.backend}). Tap "Start Camera"`, 'success');
            document.getElementById('startCamera').disabled = false;

            if (isCameraRunning) {
                scheduleNextFrame();
            }
        } else if (msg.type === 'INFERENCE_RESULT') {
            isWorkerBusy = false;
            latestDetections = msg.detections;
            stats.latency = msg.latency;
            stats.fps = 1000 / msg.latency;
            stats.detections = latestDetections.length;

            updateStatsDisplay(msg.latency, msg.breakdown);

            // Queue next frame
            if (isCameraRunning) {
                scheduleNextFrame();
            }
        } else if (msg.type === 'ERROR') {
            isSwitching = false;
            const progressContainer = document.getElementById('progress-container');
            if (progressContainer) progressContainer.style.display = 'none';
            updateStatus(`${msg.message}`, 'error');
        } else if (msg.type === 'INFERENCE_ERROR') {
            isWorkerBusy = false;
            if (isCameraRunning) {
                setTimeout(scheduleNextFrame, 200);
            }
        }
    };

    worker.onerror = function(err) {
        console.error('[Worker Fatal]', err);
        updateStatus(`Worker error: ${err.message || 'Script failed'}`, 'error');
    };
}

async function init() {
    console.log('[App] Initializing High-Performance 256x256 Client-Side Engine...');
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    document.getElementById('startCamera').addEventListener('click', startCamera);
    document.getElementById('stopCamera').addEventListener('click', stopCamera);

    initWorker();
    await loadPrecisionModel('int8');
}

window.addEventListener('DOMContentLoaded', init);
