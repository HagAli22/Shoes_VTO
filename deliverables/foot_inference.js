/**
 * foot_inference.js
 * Reference JavaScript Decoder for Shoes VTO (Stage A - 16 Keypoint Foot Detector)
 * Runtime: onnxruntime-web (WebGPU EP / WASM EP)
 * 
 * Usage:
 *   import { FootDetector } from './foot_inference.js';
 *   const detector = new FootDetector();
 *   await detector.loadModel('./models/stage_a_16kp_fp32.onnx');
 *   const detections = await detector.detect(videoElementOrCanvas);
 */

export class FootDetector {
    constructor(options = {}) {
        this.inputSize = 320;
        this.boxConfThresh = options.boxConfThresh || 0.30;
        this.crossNmsIou = options.crossNmsIou || 0.35;
        this.kpConfThresh = options.kpConfThresh || 0.30;
        this.session = null;

        // Channel index inside model output to Anatomical Keypoint Definition
        this.channelMapping = {
            0:  { index: 0,  name: 'toe_tip',           zone: 'sole' },
            11: { index: 1,  name: 'toe_ground',        zone: 'sole' },
            13: { index: 2,  name: 'heel_back',         zone: 'sole' },
            12: { index: 3,  name: 'heel_ground',       zone: 'sole' },
            3:  { index: 4,  name: 'ball_medial',       zone: 'ball' },
            4:  { index: 5,  name: 'ball_lateral',      zone: 'ball' },
            5:  { index: 6,  name: 'ball_top',          zone: 'ball' },
            6:  { index: 7,  name: 'instep_top',        zone: 'mid' },
            7:  { index: 8,  name: 'arch_medial',       zone: 'mid' },
            8:  { index: 9,  name: 'midfoot_lateral',   zone: 'mid' },
            9:  { index: 10, name: 'malleolus_medial',  zone: 'ankle' },
            10: { index: 11, name: 'malleolus_lateral', zone: 'ankle' },
            14: { index: 14, name: 'achilles',          zone: 'ankle' },
            15: { index: 15, name: 'shin_mid',          zone: 'leg' }
        };
    }

    async loadModel(modelPathOrBuffer, ort) {
        const options = {
            executionProviders: ['webgpu', 'wasm'],
            graphOptimizationLevel: 'all'
        };
        this.session = await ort.InferenceSession.create(modelPathOrBuffer, options);
    }

    /**
     * Preprocesses input image/video to 320x320 letterbox Float32Tensor [1, 3, 320, 320]
     */
    preprocess(imageSource, ort) {
        const srcW = imageSource.videoWidth || imageSource.width;
        const srcH = imageSource.videoHeight || imageSource.height;

        const scale = Math.min(this.inputSize / srcW, this.inputSize / srcH);
        const padW = Math.round((this.inputSize - srcW * scale) / 2);
        const padH = Math.round((this.inputSize - srcH * scale) / 2);

        const canvas = document.createElement('canvas');
        canvas.width = this.inputSize;
        canvas.height = this.inputSize;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#727272'; // 114 gray
        ctx.fillRect(0, 0, this.inputSize, this.inputSize);
        ctx.drawImage(imageSource, padW, padH, srcW * scale, srcH * scale);

        const imgData = ctx.getImageData(0, 0, this.inputSize, this.inputSize).data;
        const floatData = new Float32Array(3 * this.inputSize * this.inputSize);

        const planeSize = this.inputSize * this.inputSize;
        for (let i = 0; i < planeSize; i++) {
            floatData[i] = imgData[i * 4] / 255.0;                      // R
            floatData[planeSize + i] = imgData[i * 4 + 1] / 255.0;        // G
            floatData[2 * planeSize + i] = imgData[i * 4 + 2] / 255.0;    // B
        }

        const tensor = new ort.Tensor('float32', floatData, [1, 3, this.inputSize, this.inputSize]);
        return { tensor, scale, padW, padH, srcW, srcH };
    }

    /**
     * Runs inference and decodes output tensor [1, 54, 2100]
     */
    async detect(imageSource, ort) {
        if (!this.session) throw new Error("Model session not loaded.");

        const { tensor, scale, padW, padH, srcW, srcH } = this.preprocess(imageSource, ort);
        const results = await this.session.run({ images: tensor });
        const output = results.output0.data; // Float32Array [54 * 2100]

        const numAnchors = 2100;
        const candidates = [];

        // Parse anchors
        for (let a = 0; a < numAnchors; a++) {
            const leftScore = output[4 * numAnchors + a];
            const rightScore = output[5 * numAnchors + a];
            const maxScore = Math.max(leftScore, rightScore);
            const cls = leftScore >= rightScore ? 0 : 1;

            if (maxScore < this.boxConfThresh) continue;

            const cx = output[0 * numAnchors + a];
            const cy = output[1 * numAnchors + a];
            const w = output[2 * numAnchors + a];
            const h = output[3 * numAnchors + a];

            // Convert letterbox coords -> full original image pixel coords
            const x1 = Math.max(0, Math.min(srcW, (cx - w / 2 - padW) / scale));
            const y1 = Math.max(0, Math.min(srcH, (cy - h / 2 - padH) / scale));
            const x2 = Math.max(0, Math.min(srcW, (cx + w / 2 - padW) / scale));
            const y2 = Math.max(0, Math.min(srcH, (cy + h / 2 - padH) / scale));

            // Extract keypoints
            const keypoints = {};
            for (let k = 0; k < 16; k++) {
                const kx_lb = output[(6 + k * 3) * numAnchors + a];
                const ky_lb = output[(6 + k * 3 + 1) * numAnchors + a];
                const kv = output[(6 + k * 3 + 2) * numAnchors + a];

                const mapping = this.channelMapping[k];
                if (mapping) {
                    const px = (kx_lb - padW) / scale;
                    const py = (ky_lb - padH) / scale;
                    keypoints[mapping.name] = {
                        index: mapping.index,
                        x: px,
                        y: py,
                        visibility: kv,
                        zone: mapping.zone
                    };
                }
            }

            candidates.push({
                bbox: [x1, y1, x2, y2],
                classId: cls,
                className: cls === 0 ? 'left_foot' : 'right_foot',
                confidence: maxScore,
                keypoints
            });
        }

        // Apply Cross-Class Spatial NMS (suppress duplicate boxes on the same foot)
        candidates.sort((a, b) => b.confidence - a.confidence);
        const finalDetections = [];

        for (const cand of candidates) {
            let duplicate = false;
            for (const existing of finalDetections) {
                if (this.computeIoU(cand.bbox, existing.bbox) > this.crossNmsIou) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate && finalDetections.length < 2) {
                finalDetections.push(cand);
            }
        }

        return finalDetections;
    }

    computeIoU(boxA, boxB) {
        const xA = Math.max(boxA[0], boxB[0]);
        const yA = Math.max(boxA[1], boxB[1]);
        const xB = Math.min(boxA[2], boxB[2]);
        const yB = Math.min(boxA[3], boxB[3]);
        const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
        const boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
        const boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
        return interArea / (boxAArea + boxBArea - interArea);
    }
}
