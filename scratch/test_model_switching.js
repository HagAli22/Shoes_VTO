const path = require('path');
const fs = require('fs');
const rootDir = path.resolve(__dirname, '..');
const ort = require(path.join(rootDir, 'tools/mobile_web_demo/public/vendor/ort.min.js'));

const MODEL_CONFIGS = {
  'twostage_int8_fp16': {
    type: 'twostage',
    name: 'NanoDet(INT8) + RTMPose(FP16)',
    detPath: path.join(rootDir, 'tools/mobile_web_demo/public/models/nanodet-improved4-int8.onnx'),
    detPrecision: 'float32',
    posePath: path.join(rootDir, 'tools/mobile_web_demo/public/models/rtmpose-13kp-hand-fp16.onnx'),
    posePrecision: 'float16'
  },
  'twostage_fp32_fp16': {
    type: 'twostage',
    name: 'NanoDet(FP32) + RTMPose(FP16)',
    detPath: path.join(rootDir, 'tools/mobile_web_demo/public/models/nanodet-improved4-fp32.onnx'),
    detPrecision: 'float32',
    posePath: path.join(rootDir, 'tools/mobile_web_demo/public/models/rtmpose-13kp-hand-fp16.onnx'),
    posePrecision: 'float16'
  },
  'twostage_fp16_fp16': {
    type: 'twostage',
    name: 'NanoDet(FP16) + RTMPose(FP16)',
    detPath: path.join(rootDir, 'tools/mobile_web_demo/public/models/nanodet-improved4-fp16.onnx'),
    detPrecision: 'float16',
    posePath: path.join(rootDir, 'tools/mobile_web_demo/public/models/rtmpose-13kp-hand-fp16.onnx'),
    posePrecision: 'float16'
  },
  'yolo_int8': {
    type: 'yolo',
    name: 'YOLOv8n (INT8)',
    path: path.join(rootDir, 'tools/mobile_web_demo/public/models/stage-a-256-13kp-int8.onnx'),
    precision: 'float32'
  }
};

const state = {
  activeModelKey: null,
  activeModelConfig: null,
  detSession: null,
  detInputName: null,
  poseSession: null,
  poseInputName: null,
  yoloSession: null,
  yoloInputName: null,
  isLoadingModel: false,
  modelError: null
};

async function createWasmSession(path) {
  const buf = fs.readFileSync(path);
  return await ort.InferenceSession.create(buf.buffer, { executionProviders: ['wasm'] });
}

function createDummyTensor(type, dims) {
  const total = dims.reduce((a, b) => a * b, 1);
  if (type === 'float16') {
    return new ort.Tensor('float16', new Uint16Array(total), dims);
  } else {
    return new ort.Tensor('float32', new Float32Array(total), dims);
  }
}

async function loadModel(modelKey) {
  console.log(`\n[*] Switching to ${modelKey}...`);
  const config = MODEL_CONFIGS[modelKey];
  state.isLoadingModel = true;

  try {
    if (config.type === 'twostage') {
      const newDetSession = await createWasmSession(config.detPath);
      const newDetInputName = newDetSession.inputNames[0];

      const isDetFp16 = config.detPrecision === 'float16';
      const detWarmup = createDummyTensor(isDetFp16 ? 'float16' : 'float32', [1, 3, 256, 256]);
      await newDetSession.run({ [newDetInputName]: detWarmup });

      let newPoseSession = state.poseSession;
      let newPoseInputName = state.poseInputName;
      if (!newPoseSession) {
        newPoseSession = await createWasmSession(config.posePath);
        newPoseInputName = newPoseSession.inputNames[0];
        const poseWarmup = createDummyTensor('float16', [1, 3, 256, 256]);
        await newPoseSession.run({ [newPoseInputName]: poseWarmup });
      }

      // Commit
      state.yoloSession = null;
      state.detSession = newDetSession;
      state.detInputName = newDetInputName;
      state.poseSession = newPoseSession;
      state.poseInputName = newPoseInputName;
      state.activeModelKey = modelKey;
      state.activeModelConfig = config;

    } else {
      const newYoloSession = await createWasmSession(config.path);
      const newYoloInputName = newYoloSession.inputNames[0];
      const isFp16 = config.precision === 'float16';
      const yoloWarmup = createDummyTensor(isFp16 ? 'float16' : 'float32', [1, 3, 256, 256]);
      await newYoloSession.run({ [newYoloInputName]: yoloWarmup });

      state.detSession = null;
      state.poseSession = null;
      state.yoloSession = newYoloSession;
      state.yoloInputName = newYoloInputName;
      state.activeModelKey = modelKey;
      state.activeModelConfig = config;
    }

    state.isLoadingModel = false;
    console.log(`[+] Successfully loaded ${config.name}`);
  } catch (err) {
    state.isLoadingModel = false;
    state.modelError = err.message;
    console.error(`[-] Load failed: ${err.message}`);
    throw err;
  }
}

async function simulateFrame() {
  if (state.isLoadingModel) {
    console.log('  [frame] Skipped because model is loading');
    return;
  }
  const cfg = state.activeModelConfig;
  if (cfg.type === 'twostage') {
    const isDetFp16 = cfg.detPrecision === 'float16';
    const detTensor = createDummyTensor(isDetFp16 ? 'float16' : 'float32', [1, 3, 256, 256]);
    const detOut = await state.detSession.run({ [state.detInputName]: detTensor });

    const poseTensor = createDummyTensor('float16', [1, 3, 256, 256]);
    const poseOut = await state.poseSession.run({ [state.poseInputName]: poseTensor });
    console.log(`  [frame] Two-Stage (${cfg.name}): detOut boxes=${detOut['boxes'].dims}, poseOut simcc_x=${poseOut['simcc_x'].dims} -> SUCCESS`);
  } else {
    const isFp16 = cfg.precision === 'float16';
    const yoloTensor = createDummyTensor(isFp16 ? 'float16' : 'float32', [1, 3, 256, 256]);
    const yoloOut = await state.yoloSession.run({ [state.yoloInputName]: yoloTensor });
    console.log(`  [frame] YOLO (${cfg.name}): output=${yoloOut[state.yoloSession.outputNames[0]].dims} -> SUCCESS`);
  }
}

async function main() {
  console.log('============================================================');
  console.log('TESTING MODEL SWITCHING ACROSS ALL PRECISIONS');
  console.log('============================================================');

  // 1. Initial Exp 1 (INT8)
  await loadModel('twostage_int8_fp16');
  await simulateFrame();

  // 2. Switch to Exp 2 (FP32)
  await loadModel('twostage_fp32_fp16');
  await simulateFrame();

  // 3. Switch to Exp 3 (FP16) - THIS WAS THE FAILING TEST
  await loadModel('twostage_fp16_fp16');
  await simulateFrame();

  // 4. Switch to YOLO INT8
  await loadModel('yolo_int8');
  await simulateFrame();

  // 5. Switch back to Exp 3 (FP16)
  await loadModel('twostage_fp16_fp16');
  await simulateFrame();

  console.log('\n============================================================');
  console.log('ALL MODEL SWITCHING TRANSITIONS PASSED WITH ZERO ERRORS!');
  console.log('============================================================');
}

main().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
