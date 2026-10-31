const fs = require('fs');
const path = require('path');
const rootDir = path.resolve('.');
const ort = require(path.join(rootDir, 'tools/mobile_web_demo/public/vendor/ort.min.js'));
const rawBytes = fs.readFileSync('scratch/frame172_raw.bin');
const srcW = 1920;
const srcH = 1080;

function float16ToFloat32(h) {
  const s = (h >> 15) & 0x0001;
  const e = (h >> 10) & 0x001f;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  else if (e === 31) return f ? NaN : ((s ? -1 : 1) * Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function getFloatValue(array, index) {
  if (array instanceof Float32Array || array instanceof Float64Array) return array[index];
  if (array instanceof Uint16Array) return float16ToFloat32(array[index]);
  return Number(array[index]);
}

async function testDirectResize() {
  const detBuf = fs.readFileSync('tools/mobile_web_demo/public/models/nanodet-improved4-int8.onnx');
  const detSession = await ort.InferenceSession.create(detBuf.buffer, { executionProviders: ['wasm'] });

  const planarF32 = new Float32Array(3 * 256 * 256);
  for (let dy = 0; dy < 256; dy++) {
    const sy = Math.floor(dy * srcH / 256.0);
    for (let dx = 0; dx < 256; dx++) {
      const sx = Math.floor(dx * srcW / 256.0);
      const srcIdx = (sy * srcW + sx) * 3;
      const dstIdx = dy * 256 + dx;
      planarF32[dstIdx] = rawBytes[srcIdx] / 255.0;
      planarF32[256*256 + dstIdx] = rawBytes[srcIdx+1] / 255.0;
      planarF32[2*256*256 + dstIdx] = rawBytes[srcIdx+2] / 255.0;
    }
  }
  const detOutputs = await detSession.run({ [detSession.inputNames[0]]: new ort.Tensor('float32', planarF32, [1, 3, 256, 256]) });
  const boxesData = detOutputs['boxes'].data;
  const scoresData = detOutputs['scores'].data;
  const scaleX = srcW / 256.0;
  const scaleY = srcH / 256.0;

  console.log('--- Direct Resize NanoDet Output ---');
  let count = 0;
  for (let a = 0; a < 1344; a++) {
    const sc = getFloatValue(scoresData, a);
    if (sc >= 0.40) {
      count++;
      const x1 = getFloatValue(boxesData, a*4+0) * scaleX;
      const y1 = getFloatValue(boxesData, a*4+1) * scaleY;
      const x2 = getFloatValue(boxesData, a*4+2) * scaleX;
      const y2 = getFloatValue(boxesData, a*4+3) * scaleY;
      console.log('  Anchor ' + a + ': score=' + sc.toFixed(3) + ', box=[' + x1.toFixed(1) + ', ' + y1.toFixed(1) + ', ' + x2.toFixed(1) + ', ' + y2.toFixed(1) + ']');
    }
  }
  console.log('Total anchors >= 0.40: ' + count);
}
testDirectResize();

