/**
 * stage_a_decoder.js
 * Reference JavaScript Decoder for Stage A Foot Detector.
 * Target Runtime: onnxruntime-web (WebGPU EP / WASM EP)
 */

export class StageADecoder {
  constructor(confThreshold = 0.30, iouThreshold = 0.35) {
    this.confThreshold = confThreshold;
    this.iouThreshold = iouThreshold;
    this.numAnchors = 2100;
  }

  /**
   * Decodes raw [1, 18, 2100] output tensor from ONNXRuntime.
   * @param {Float32Array} rawOutput - Output tensor data
   * @returns {Array<Object>} List of detected feet (Max 2: left & right)
   */
  decode(rawOutput) {
    const candidates = [];
    const N = this.numAnchors;

    for (let i = 0; i < N; i++) {
      const scoreLeft = rawOutput[4 * N + i];
      const scoreRight = rawOutput[5 * N + i];
      const maxScore = Math.max(scoreLeft, scoreRight);

      if (maxScore >= this.confThreshold) {
        const cls = scoreLeft >= scoreRight ? 0 : 1;
        const cx = rawOutput[0 * N + i];
        const cy = rawOutput[1 * N + i];
        const w  = rawOutput[2 * N + i];
        const h  = rawOutput[3 * N + i];

        const x1 = cx - w / 2;
        const y1 = cy - h / 2;
        const x2 = cx + w / 2;
        const y2 = cy + h / 2;

        // 4 Coarse Keypoints (toe_tip, heel_back, ball_medial, ball_lateral)
        const coarseKpts = [];
        for (let k = 0; k < 4; k++) {
          const kx = rawOutput[(6 + k * 3) * N + i];
          const ky = rawOutput[(7 + k * 3) * N + i];
          const kv = rawOutput[(8 + k * 3) * N + i];
          coarseKpts.push({ x: kx, y: ky, v: kv });
        }

        candidates.push({
          classId: cls,
          className: cls === 0 ? 'left_foot' : 'right_foot',
          confidence: maxScore,
          box: [x1, y1, x2, y2],
          coarseKpts: coarseKpts
        });
      }
    }

    candidates.sort((a, b) => b.confidence - a.confidence);

    // Cross-Class Spatial NMS (Max 2 feet)
    const selected = [];
    for (const cand of candidates) {
      let overlap = false;
      for (const sel of selected) {
        const iou = this.computeIoU(cand.box, sel.box);
        if (iou > this.iouThreshold) {
          overlap = true;
          break;
        }
      }
      if (!overlap && selected.length < 2) {
        selected.push(cand);
      }
    }

    return selected;
  }

  computeIoU(b1, b2) {
    const xA = Math.max(b1[0], b2[0]);
    const yA = Math.max(b1[1], b2[1]);
    const xB = Math.min(b1[2], b2[2]);
    const yB = Math.min(b1[3], b2[3]);
    const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const areaA = Math.max(1e-5, (b1[2] - b1[0]) * (b1[3] - b1[1]));
    const areaB = Math.max(1e-5, (b2[2] - b2[0]) * (b2[3] - b2[1]));
    return inter / (areaA + areaB - inter);
  }
}
