const fs = require('fs');
const path = require('path');
const rootDir = path.resolve(__dirname, '..');
const ort = require(path.join(rootDir, 'tools/mobile_web_demo/public/vendor/ort.min.js'));

// Read raw frame 172 RGB (1920x1080, 3 channels)
const rawBytes = fs.readFileSync('scratch/frame172_raw.bin');
const srcW = 1920;
const srcH = 1080;

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

function decodeNanoDet(boxesTensor, scoresTensor, prepMeta, confThresh = 0.40, nmsThresh = 0.45) {
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

      const origX1 = Math.max(0, Math.min(prepMeta.srcW, (x1 - prepMeta.padX) / prepMeta.ratio));
      const origY1 = Math.max(0, Math.min(prepMeta.srcH, (y1 - prepMeta.padY) / prepMeta.ratio));
      const origX2 = Math.max(0, Math.min(prepMeta.srcW, (x2 - prepMeta.padX) / prepMeta.ratio));
      const origY2 = Math.max(0, Math.min(prepMeta.srcH, (y2 - prepMeta.padY) / prepMeta.ratio));

      candidates.push({
        boxPx: [x1, y1, x2, y2],
        origBox: [origX1, origY1, origX2, origY2],
        score: score
      });
    }
  }

  console.log(`[decodeNanoDet] candidates >= ${confThresh}: ${candidates.length}`);
  if (candidates.length === 0) return [];

  const keepIndices = nms(candidates.map(c => c.boxPx), candidates.map(c => c.score), nmsThresh);
  let detections = keepIndices.map(i => candidates[i]);
  if (detections.length > 2) {
    detections.sort((a, b) => b.score - a.score);
    detections = detections.slice(0, 2);
  }
  console.log(`[decodeNanoDet] kept after NMS: ${detections.length}`);
  return detections;
}

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
    const kptOrigX = cx + (resX.coord - 128.0) * (size / 256.0);
    const kptOrigY = cy + (resY.coord - 128.0) * (size / 256.0);
    keypoints.push({ index: k, x: kptOrigX, y: kptOrigY, visibility: conf });
    sumConf += conf;
    if (conf >= 0.20) validKptsCount++;
  }

  const meanConf = sumConf / 13.0;
  const handProb = handData ? getFloatValue(handData, 0) : 0.5;
  return { keypoints, meanConf, validKptsCount, handProb };
}

async function run() {
  console.log('Loading ONNX sessions in WASM...');
  const detBuf = fs.readFileSync('tools/mobile_web_demo/public/models/nanodet-improved4-int8.onnx');
  const detSession = await ort.InferenceSession.create(detBuf.buffer, { executionProviders: ['wasm'] });
  const poseBuf = fs.readFileSync('tools/mobile_web_demo/public/models/rtmpose-13kp-hand-fp16.onnx');
  const poseSession = await ort.InferenceSession.create(poseBuf.buffer, { executionProviders: ['wasm'] });

  // Simulate letterbox 256x256
  const targetSize = 256;
  const ratio = Math.min(targetSize / srcW, targetSize / srcH);
  const nw = Math.round(srcW * ratio);
  const nh = Math.round(srcH * ratio);
  const padX = (targetSize - nw) / 2;
  const padY = (targetSize - nh) / 2;
  console.log(`Letterbox: ratio=${ratio.toFixed(4)}, nw=${nw}, nh=${nh}, padX=${padX}, padY=${padY}`);

  // Create input tensor (simple bilinear or nearest for test)
  const planarF32 = new Float32Array(3 * 256 * 256);
  planarF32.fill(114.0 / 255.0); // background
  for (let y = 0; y < nh; y++) {
    const srcY = Math.floor(y / ratio);
    for (let x = 0; x < nw; x++) {
      const srcX = Math.floor(x / ratio);
      const srcIdx = (srcY * srcW + srcX) * 3;
      const dstIdx = (padY + y) * 256 + (padX + x);
      planarF32[dstIdx] = rawBytes[srcIdx] / 255.0;
      planarF32[256 * 256 + dstIdx] = rawBytes[srcIdx + 1] / 255.0;
      planarF32[2 * 256 * 256 + dstIdx] = rawBytes[srcIdx + 2] / 255.0;
    }
  }

  const prepTensor = new ort.Tensor('float32', planarF32, [1, 3, 256, 256]);
  const prepMeta = { ratio, padX, padY, srcW, srcH };

  // Step A: NanoDet
  const detOutputs = await detSession.run({ [detSession.inputNames[0]]: prepTensor });
  const rawDetections = decodeNanoDet(detOutputs['boxes'], detOutputs['scores'], prepMeta, 0.40, 0.45);

  // Step B: Pose
  const validCandidates = [];
  for (const det of rawDetections) {
    const b = det.origBox;
    const bw = b[2] - b[0];
    const bh = b[3] - b[1];
    const cx = (b[0] + b[2]) / 2.0;
    const cy = (b[1] + b[3]) / 2.0;
    const ar = Math.max(bw / bh, bh / bw);
    const extraMargin = ar > 1.3 ? 0.15 : 0.05;
    const size = Math.max(bw, bh) * (1.0 + 0.25 + extraMargin);

    // Extract crop
    const cropX1 = cx - size / 2.0;
    const cropY1 = cy - size / 2.0;
    const cropTensorData = new Uint16Array(3 * 256 * 256);
    cropTensorData.fill(float32ToFloat16(114.0 / 255.0));

    const scale = 256.0 / size;
    for (let dy = 0; dy < 256; dy++) {
      const sy = Math.floor(cropY1 + dy / scale);
      if (sy < 0 || sy >= srcH) continue;
      for (let dx = 0; dx < 256; dx++) {
        const sx = Math.floor(cropX1 + dx / scale);
        if (sx < 0 || sx >= srcW) continue;
        const srcIdx = (sy * srcW + sx) * 3;
        const dstIdx = dy * 256 + dx;
        cropTensorData[dstIdx] = float32ToFloat16(rawBytes[srcIdx] / 255.0);
        cropTensorData[256 * 256 + dstIdx] = float32ToFloat16(rawBytes[srcIdx + 1] / 255.0);
        cropTensorData[2 * 256 * 256 + dstIdx] = float32ToFloat16(rawBytes[srcIdx + 2] / 255.0);
      }
    }

    const cropTensor = new ort.Tensor('float16', cropTensorData, [1, 3, 256, 256]);
    const poseOutputs = await poseSession.run({ [poseSession.inputNames[0]]: cropTensor });
    const poseResult = decodeRtmPoseOutputs(poseOutputs, cx, cy, size);

    console.log(`Det score=${det.score.toFixed(3)}, meanConf=${poseResult.meanConf.toFixed(3)}, validKpts=${poseResult.validKptsCount}/13, handProb=${poseResult.handProb.toFixed(3)}`);
    if (poseResult.meanConf < 0.20 || poseResult.validKptsCount < 3) {
      console.log('  -> REJECTED BY CLUTTER GATE!');
      continue;
    }
    validCandidates.push({
      origBox: det.origBox,
      score: det.score,
      keypoints: poseResult.keypoints,
      handProb: poseResult.handProb,
      jointScore: det.score * (0.5 + 0.5 * poseResult.meanConf),
      cx: cx
    });
  }

  console.log(`validCandidates count: ${validCandidates.length}`);
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

  const bestLeft = leftCands.length > 0 ? leftCands.reduce((prev, curr) => (curr.jointScore > prev.jointScore ? curr : prev)) : null;
  const bestRight = rightCands.length > 0 ? rightCands.reduce((prev, curr) => (curr.jointScore > prev.jointScore ? curr : prev)) : null;

  const finalDetections = [];
  if (bestLeft) finalDetections.push({ label: 'Left Foot', box: bestLeft.origBox });
  if (bestRight) finalDetections.push({ label: 'Right Foot', box: bestRight.origBox });

  console.log('FINAL DETECTIONS:', finalDetections);
}

run().catch(console.error);
