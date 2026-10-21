/**
 * crop-transforms.js
 * ──────────────────
 * Coordinate transformations between original image space and letterboxed tensor space,
 * plus Stage A -> Stage B foot crop transformation utilities.
 * Stage A Model Input: 256x256 RGB.
 */

'use strict';

/**
 * Computes letterbox scale and padding offsets for an image fitting into targetSize (256x256).
 * @param {number} origW - Original image width in pixels.
 * @param {number} origH - Original image height in pixels.
 * @param {number} targetSize - Target letterbox square size (default: 256).
 * @returns {{ scale: number, padX: number, padY: number, newW: number, newH: number }}
 */
function getLetterboxParams(origW, origH, targetSize = 256) {
  const scale = Math.min(targetSize / origW, targetSize / origH);
  const newW = Math.round(origW * scale);
  const newH = Math.round(origH * scale);
  const padX = (targetSize - newW) / 2.0;
  const padY = (targetSize - newH) / 2.0;
  return { scale, padX, padY, newW, newH };
}

/**
 * Alias for getLetterboxParams for backwards/SDK compatibility.
 */
function computeLetterboxTransform(origW, origH, targetW = 256, targetH = 256) {
  const targetSize = Math.max(targetW, targetH);
  return getLetterboxParams(origW, origH, targetSize);
}

/**
 * Transforms coordinates from letterbox tensor space [0..256] back to original image space.
 * @param {number} x - Letterbox X coordinate.
 * @param {number} y - Letterbox Y coordinate.
 * @param {{ scale: number, padX: number, padY: number }} params - Letterbox parameters.
 * @returns {[number, number]} - [origX, origY]
 */
function unletterboxPoint(x, y, params) {
  const origX = (x - params.padX) / params.scale;
  const origY = (y - params.padY) / params.scale;
  return [origX, origY];
}

/**
 * Transforms bounding box [x1, y1, x2, y2] from letterbox space back to original image space.
 * @param {[number, number, number, number]} bbox - Letterbox bbox.
 * @param {{ scale: number, padX: number, padY: number }} params - Letterbox parameters.
 * @returns {[number, number, number, number]} - [origX1, origY1, origX2, origY2]
 */
function unletterboxBBox(bbox, params) {
  const [x1, y1] = unletterboxPoint(bbox[0], bbox[1], params);
  const [x2, y2] = unletterboxPoint(bbox[2], bbox[3], params);
  return [x1, y1, x2, y2];
}

/**
 * Transforms detections with letterbox-space boxes/keypoints to original image space.
 * (Note: decodeStageA256 already does this automatically when origW and origH are provided).
 * @param {Array<object>} detections - Detections array.
 * @param {{ scale: number, padX: number, padY: number }} params - Letterbox parameters.
 * @returns {Array<object>} - Transformed detections.
 */
function transformDetectionsToOriginal(detections, params) {
  return detections.map(det => {
    const unbox = unletterboxBBox(det.bbox, params);
    const unkpts = (det.keypoints || []).map(kp => {
      const [ux, uy] = unletterboxPoint(kp.x, kp.y, params);
      return { ...kp, x: ux, y: uy };
    });
    return {
      ...det,
      bbox: unbox,
      keypoints: unkpts
    };
  });
}

/**
 * Extracts a square foot crop from a canvas/image for Stage B alignment.
 * @param {HTMLCanvasElement|HTMLImageElement} imageSource - Video frame or canvas.
 * @param {{ x1: number, y1: number, x2: number, y2: number, width: number, height: number }} bbox - Foot bounding box.
 * @param {number} targetSize - Output crop resolution (e.g. 224 or 256).
 * @param {number} marginFactor - Padding multiplier (default: 1.25).
 */
function extractFootCrop(imageSource, bbox, targetSize = 224, marginFactor = 1.25) {
  const bx1 = bbox.x1 !== undefined ? bbox.x1 : bbox[0];
  const by1 = bbox.y1 !== undefined ? bbox.y1 : bbox[1];
  const bx2 = bbox.x2 !== undefined ? bbox.x2 : bbox[2];
  const by2 = bbox.y2 !== undefined ? bbox.y2 : bbox[3];
  const bw = bx2 - bx1;
  const bh = by2 - by1;

  const cx = (bx1 + bx2) / 2.0;
  const cy = (by1 + by2) / 2.0;
  const maxDim = Math.max(bw, bh) * marginFactor;

  const cropX1 = cx - maxDim / 2.0;
  const cropY1 = cy - maxDim / 2.0;

  return {
    cropX1,
    cropY1,
    cropSize: maxDim,
    targetSize,
    scale: maxDim / targetSize
  };
}

/**
 * Projects a point from Stage B crop space back to the original full video frame.
 * @param {{ x: number, y: number, confidence?: number, visibility?: number }} cropKpt
 * @param {{ cropX1: number, cropY1: number, scale: number }} transform
 */
function projectCropToFrame(cropKpt, transform) {
  return {
    ...cropKpt,
    x: transform.cropX1 + cropKpt.x * transform.scale,
    y: transform.cropY1 + cropKpt.y * transform.scale
  };
}

module.exports = {
  getLetterboxParams,
  computeLetterboxTransform,
  unletterboxPoint,
  unletterboxBBox,
  transformDetectionsToOriginal,
  extractFootCrop,
  projectCropToFrame
};
