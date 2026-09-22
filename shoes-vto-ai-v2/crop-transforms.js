/**
 * crop-transforms.js
 * ──────────────────
 * Standardized Crop Transformation Utilities for Stage A -> Stage B Cropping.
 */

export function extractFootCrop(imageCanvas, bbox, targetSize = 224, marginFactor = 1.25) {
  const cx = (bbox.x1 + bbox.x2) / 2;
  const cy = (bbox.y1 + bbox.y2) / 2;
  const maxDim = Math.max(bbox.width, bbox.height) * marginFactor;

  const cropX1 = cx - maxDim / 2;
  const cropY1 = cy - maxDim / 2;

  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = targetSize;
  cropCanvas.height = targetSize;
  const ctx = cropCanvas.getContext("2d");

  ctx.drawImage(imageCanvas, cropX1, cropY1, maxDim, maxDim, 0, 0, targetSize, targetSize);

  return {
    canvas: cropCanvas,
    transform: {
      cropX1,
      cropY1,
      cropSize: maxDim,
      targetSize
    }
  };
}

export function projectCropToFrame(cropKpt, transform) {
  const scale = transform.cropSize / transform.targetSize;
  return {
    x: transform.cropX1 + cropKpt.x * scale,
    y: transform.cropY1 + cropKpt.y * scale,
    confidence: cropKpt.confidence
  };
}
