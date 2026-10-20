/**
 * crop-transforms.js
 * ──────────────────
 * Coordinate transformations between original image space and letterboxed tensor space.
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

module.exports = {
  getLetterboxParams,
  unletterboxPoint,
  unletterboxBBox
};
