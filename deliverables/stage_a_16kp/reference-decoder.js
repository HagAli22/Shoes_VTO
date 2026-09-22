/**
 * reference-decoder.js
 * ────────────────────
 * Production JavaScript Reference Decoder for Shoes VTO Stage A 16-Keypoint ONNX Models.
 * 
 * Features:
 *   - Decodes [1, 54, 2100] output tensor.
 *   - Independent Per-Class NMS (Preserves both feet during overlapping & crossing).
 *   - Inverts letterbox transform back to original image dimensions.
 *   - Returns structured 16-keypoint anatomical landmarks with confidence scores.
 */

const KEYPOINT_NAMES_16 = [
  "toe_ground",        // Index 0
  "heel_back",         // Index 1
  "heel_ground",       // Index 2
  "ball_medial",       // Index 3
  "ball_lateral",      // Index 4
  "ball_top",          // Index 5
  "instep_top",        // Index 6
  "arch_medial",       // Index 7
  "midfoot_lateral",   // Index 8
  "malleolus_medial",  // Index 9
  "malleolus_lateral", // Index 10
  "toe_tip",           // Index 11
  "ankle_center",      // Index 12
  "throat",            // Index 13
  "achilles",          // Index 14
  "shin_mid"           // Index 15
];


function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function computeIoU(b1, b2) {
  const x1 = Math.max(b1.x1, b2.x1);
  const y1 = Math.max(b1.y1, b2.y1);
  const x2 = Math.min(b1.x2, b2.x2);
  const y2 = Math.min(b1.y2, b2.y2);

  const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const b1Area = (b1.x2 - b1.x1) * (b1.y2 - b1.y1);
  const b2Area = (b2.x2 - b2.x1) * (b2.y2 - b2.y1);
  const unionArea = b1Area + b2Area - interArea;

  return unionArea > 0 ? interArea / unionArea : 0;
}

/**
 * Decodes the raw Stage A 16-KP tensor into detections.
 * @param {Float32Array} tensorData - Flat array of size 54 * 2100 = 113400 elements.
 * @param {number} origWidth - Original image width in pixels.
 * @param {number} origHeight - Original image height in pixels.
 * @param {object} options - Thresholds: { confThresh: 0.25, nmsThresh: 0.45, kptThresh: 0.3 }
 * @returns {Array<object>} Array of detected foot objects.
 */
export function decodeStageA16KP(tensorData, origWidth, origHeight, options = {}) {
  const confThresh = options.confThresh ?? 0.25;
  const nmsThresh = options.nmsThresh ?? 0.45;
  const kptThresh = options.kptThresh ?? 0.30;
  const imgSize = 320;
  const numAnchors = 2100;
  const numChannels = 54;

  // Compute letterbox padding scale & offsets
  const scale = Math.min(imgSize / origWidth, imgSize / origHeight);
  const padX = (imgSize - origWidth * scale) / 2;
  const padY = (imgSize - origHeight * scale) / 2;

  const candidates = [];

  for (let a = 0; a < numAnchors; a++) {
    const cx = tensorData[0 * numAnchors + a];
    const cy = tensorData[1 * numAnchors + a];
    const w  = tensorData[2 * numAnchors + a];
    const h  = tensorData[3 * numAnchors + a];

    // Sigmoid class scores
    const scoreLeft = sigmoid(tensorData[4 * numAnchors + a]);
    const scoreRight = sigmoid(tensorData[5 * numAnchors + a]);

    const maxScore = Math.max(scoreLeft, scoreRight);
    if (maxScore < confThresh) continue;

    const classId = scoreLeft >= scoreRight ? 0 : 1;
    const className = classId === 0 ? "left_foot" : "right_foot";

    // Bounding box in letterbox coords -> convert to original image coords
    const lbX1 = cx - w / 2;
    const lbY1 = cy - h / 2;
    const lbX2 = cx + w / 2;
    const lbY2 = cy + h / 2;

    const origX1 = Math.max(0, Math.min(origWidth, (lbX1 - padX) / scale));
    const origY1 = Math.max(0, Math.min(origHeight, (lbY1 - padY) / scale));
    const origX2 = Math.max(0, Math.min(origWidth, (lbX2 - padX) / scale));
    const origY2 = Math.max(0, Math.min(origHeight, (lbY2 - padY) / scale));

    // Decode 16 Keypoints
    const keypoints = [];
    for (let k = 0; k < 16; k++) {
      const kxRaw = tensorData[(6 + k * 3) * numAnchors + a];
      const kyRaw = tensorData[(7 + k * 3) * numAnchors + a];
      const kvRaw = tensorData[(8 + k * 3) * numAnchors + a];

      const kxOrig = (kxRaw - padX) / scale;
      const kyOrig = (kyRaw - padY) / scale;
      const conf = sigmoid(kvRaw);

      keypoints.push({
        index: k,
        name: KEYPOINT_NAMES_16[k],
        x: Math.max(0, Math.min(origWidth, kxOrig)),
        y: Math.max(0, Math.min(origHeight, kyOrig)),
        confidence: conf,
        visible: conf >= kptThresh
      });
    }

    candidates.push({
      classId,
      className,
      confidence: maxScore,
      scoreLeft,
      scoreRight,
      bbox: {
        x1: origX1,
        y1: origY1,
        x2: origX2,
        y2: origY2,
        width: origX2 - origX1,
        height: origY2 - origY1
      },
      keypoints
    });
  }

  // INDEPENDENT PER-CLASS NMS (Crucial: never cross-suppress left vs right foot)
  const finalDetections = [];
  for (const targetClassId of [0, 1]) {
    const classCandidates = candidates
      .filter(c => c.classId === targetClassId)
      .sort((a, b) => b.confidence - a.confidence);

    const keep = [];
    for (const cand of classCandidates) {
      let suppressed = false;
      for (const kept of keep) {
        if (computeIoU(cand.bbox, kept.bbox) > nmsThresh) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) {
        keep.push(cand);
      }
    }
    finalDetections.push(...keep);
  }

  return finalDetections.sort((a, b) => b.confidence - a.confidence);
}
