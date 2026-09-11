// Pin the JS and WASM to the same release; no bundler is required.
const CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const video = document.querySelector('#camera');
const canvas = document.querySelector('#overlay');
let ctx = canvas.getContext('2d');
const start = document.querySelector('#start');
const debug = document.querySelector('#debug');
const status = document.querySelector('#status');
const fps = document.querySelector('#fps');
const menu = document.querySelector('#menu');
const ar = document.querySelector('#ar');
const back = document.querySelector('#back');
const PRESETS = {
  gentleman: { skin: '#ffe39a', body: '#8c7dff', arm: '#9effca', accent: '#ff927e', iris: '#46d9c5', bg: '#eee7fd' },
  robot: { skin: '#bce7f4', body: '#507ddb', arm: '#82c8e3', accent: '#ffd475', iris: '#377fe0', bg: '#e2edf9' },
  cat: { skin: '#ffc17e', body: '#ed866c', arm: '#ffb865', accent: '#ffe7a3', iris: '#709e50', bg: '#fbe8d7' },
  bunny: { skin: '#ffd7e5', body: '#be83be', arm: '#f2b6d4', accent: '#e4d8ff', iris: '#ba67ba', bg: '#f5e3f0' },
  alien: { skin: '#b9e69a', body: '#649777', arm: '#bce8af', accent: '#ddd586', iris: '#7655cf', bg: '#e5efdc' },
  bear: { skin: '#c99972', body: '#629ba8', arm: '#d6ac87', accent: '#ffe39a', iris: '#825637', bg: '#eae5df' }
};
const DEFAULT_CATALOG = [
  { id: 'gentleman', name: 'ひげダンディ', description: 'くるりんひげの人気者', template: 'gentleman' },
  { id: 'robot', name: 'ピコロボ', description: '未来からきた相棒', template: 'robot' },
  { id: 'cat', name: 'ミケねこ', description: '気まぐれな冒険家', template: 'cat' },
  { id: 'bunny', name: 'ももウサギ', description: 'ふわふわ、はずむ笑顔', template: 'bunny' },
  { id: 'alien', name: 'そらマメ', description: '宇宙からこんにちは', template: 'alien' },
  { id: 'bear', name: 'くまポン', description: 'のんびりやさしい友だち', template: 'bear' }
];
let catalog = DEFAULT_CATALOG.map(item => ({ ...item }));
let selected = catalog[0];
let theme = { ...PRESETS[selected.template] };
let startGeneration = 0, modelLoading = null;
const REQUIRED = [0, 11, 12, 13, 14, 15, 16];
const LINKS = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24]];
const smooth = Array.from({ length: 33 }, () => ({ x: 0, y: 0 }));
const points = Array.from({ length: 33 }, () => ({ x: 0, y: 0 }));
const faceSmooth = Array.from({ length: 478 }, () => ({ x: 0, y: 0 }));
const facePoints = Array.from({ length: 478 }, () => ({ x: 0, y: 0 }));
// Anatomical sides are preserved; only the screen projection is mirrored.
const EYES = [
  { corners: [362, 263], iris: 473, blink: 'eyeBlinkLeft', color: '#46d9c5' },
  { corners: [33, 133], iris: 468, blink: 'eyeBlinkRight', color: '#8c7dff' }
];
const FACE_LOOPS = [
  [33, 160, 158, 133, 153, 144, 33],
  [362, 385, 387, 263, 373, 380, 362],
  [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91, 61],
  [78, 81, 13, 311, 308, 402, 14, 178, 78]
];
const expression = { eyeBlinkLeft: 0, eyeBlinkRight: 0, jawOpen: 0 };
const expressionTarget = { ...expression };
let landmarker, faceLandmarker, stream, raw, faceRaw;
let faceVisible = false, lastPoseInference = -Infinity;
let running = false, visible = false, showDebug = false, raf = 0;
let width = 0, height = 0, lastVideo = -1, lastInference = 0, lastDraw = 0;
let lastResult = 0, fpsStart = 0, detections = 0;
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, low = 0, high = 1) => Math.min(high, Math.max(low, v));
const distance = (a, b) => Math.hypot(b.x - a.x, b.y - a.y);
const confidence = p => p && Number.isFinite(p.x) && Number.isFinite(p.y)
  && (p.visibility ?? 0) >= 0.55 && (p.presence ?? 1) >= 0.55;
function say(message) { if (status.textContent !== message) status.textContent = message; }

function resize() {
  width = canvas.clientWidth;
  height = canvas.clientHeight;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
new ResizeObserver(resize).observe(canvas);

function loadModel() {
  if (!modelLoading) modelLoading = createModels().finally(() => { modelLoading = null; });
  return modelLoading;
}
async function createModels() {
  if (landmarker && faceLandmarker) return;
  const { FilesetResolver, PoseLandmarker, FaceLandmarker } = await import(`${CDN}/vision_bundle.mjs`);
  const files = await FilesetResolver.forVisionTasks(`${CDN}/wasm`);
  const options = {
    baseOptions: { modelAssetPath: MODEL, delegate: 'GPU' },
    runningMode: 'VIDEO', numPoses: 1,
    minPoseDetectionConfidence: 0.6, minPosePresenceConfidence: 0.6,
    minTrackingConfidence: 0.6, outputSegmentationMasks: false
  };
  async function createTask(Task, settings) {
    try { return await Task.createFromOptions(files, settings); }
    catch {
      say('CPUでのトラッキングを試しています…');
      settings.baseOptions.delegate = 'CPU';
      return Task.createFromOptions(files, settings);
    }
  }
  try {
    if (!landmarker) landmarker = await createTask(PoseLandmarker, options);
    say('目と口のモデルを読み込んでいます…');
    if (!faceLandmarker) faceLandmarker = await createTask(FaceLandmarker, {
      baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'GPU' },
      runningMode: 'VIDEO', numFaces: 1,
      minFaceDetectionConfidence: 0.6, minFacePresenceConfidence: 0.6,
      minTrackingConfidence: 0.6, outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: false
    });
  } catch (error) {
    closeModels();
    throw error;
  }
}

function closeModels() {
  landmarker?.close(); faceLandmarker?.close();
  landmarker = faceLandmarker = null;
}

function stopCamera() {
  startGeneration++;
  start.disabled = false;
  running = false;
  cancelAnimationFrame(raf);
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  raw = null;
  faceRaw = null;
  faceVisible = false;
  visible = false;
  ctx.clearRect(0, 0, width, height);
  start.textContent = 'カメラを開始';
  fps.textContent = '— FPS';
}

start.addEventListener('click', async () => {
  if (running) { stopCamera(); say('カメラを停止しました'); return; }
  const generation = ++startGeneration;
  start.disabled = true;
  try {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      throw new Error('カメラを使用するにはHTTPSまたはlocalhostで開いてください。');
    }
    say('体と顔のモデルを読み込んでいます…');
    await loadModel();
    if (generation !== startGeneration) return;
    say('カメラへのアクセスを許可してください…');
    const acquired = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 } }
    });
    if (generation !== startGeneration) { acquired.getTracks().forEach(track => track.stop()); return; }
    stream = acquired;
    video.srcObject = stream;
    await video.play();
    if (generation !== startGeneration) return;
    stream.getVideoTracks()[0].addEventListener('ended', () => {
      stopCamera(); say('カメラの接続が切れました。もう一度開始してください。');
    }, { once: true });
    running = true;
    lastVideo = -1;
    lastPoseInference = -Infinity;
    lastInference = lastDraw = lastResult = 0;
    fpsStart = performance.now();
    detections = 0;
    start.textContent = 'カメラを停止';
    say('顔と上半身を探しています…');
    raf = requestAnimationFrame(frame);
  } catch (error) {
    if (generation !== startGeneration) return;
    stopCamera();
    console.error(error);
    say(error.name === 'NotAllowedError' ? 'カメラへのアクセスが拒否されました。許可してから再度お試しください。'
      : error.name === 'NotFoundError' ? 'カメラが見つかりません。'
      : error.name === 'NotReadableError' ? 'カメラを使用できません。他のカメラアプリを閉じてください。'
      : '起動できませんでした。HTTPSまたはlocalhostで開き、インターネット接続を確認して再度お試しください。');
  } finally { if (generation === startGeneration) start.disabled = false; }
});
back.addEventListener('click', () => {
  stopCamera();
  ar.hidden = true; menu.hidden = false;
  document.querySelector('#menu-title').focus();
  refreshCatalog();
});
debug.addEventListener('click', () => {
  showDebug = !showDebug;
  debug.setAttribute('aria-pressed', String(showDebug));
});
window.addEventListener('pagehide', () => { stopCamera(); closeModels(); });
document.addEventListener('visibilitychange', () => {
  if (document.hidden && running) { stopCamera(); say('カメラを一時停止しました。開始ボタンで再開できます。'); }
});

function frame(now) {
  if (!running) return;
  try {
    if (video.readyState >= 2 && video.currentTime !== lastVideo && now - lastInference >= 1000 / 30) {
      lastVideo = video.currentTime;
      lastInference = now;
      // Face gets priority for short blinks; run the body at up to 15 Hz.
      updateFace(faceLandmarker.detectForVideo(video, now));
      if (now - lastPoseInference >= 1000 / 15) {
        lastPoseInference = now;
        const result = landmarker.detectForVideo(video, now);
        raw = result.landmarks[0] ?? null;
        const good = !!raw && REQUIRED.every(i => confidence(raw[i]))
          && Math.hypot((raw[11].x - raw[12].x) * video.videoWidth,
            (raw[11].y - raw[12].y) * video.videoHeight) > 25;
        if (good && !visible) {
          raw.forEach((p, i) => { smooth[i].x = p.x; smooth[i].y = p.y; });
        }
        visible = good;
      }
      lastResult = now;
      say(faceVisible
        ? visible ? '体と顔を追跡中 · まばたきや口の開閉を試してください！' : '顔を追跡中 · 腕が映るように少し離れてください'
        : visible ? '体を追跡中 · カメラに顔を向けてください' : 'カメラに顔を映してください');
      detections++;
      if (now - fpsStart >= 1000) {
        fps.textContent = `${Math.round(detections * 1000 / (now - fpsStart))} FPS`;
        detections = 0;
        fpsStart = now;
      }
    }
    if (now - lastResult > 500) {
      visible = faceVisible = false; raw = faceRaw = null;
      say('カメラ映像を待っています…');
    }
    const dt = Math.min((now - lastDraw) / 1000, 0.1);
    const alpha = 1 - Math.exp(-dt / 0.065);
    lastDraw = now;
    ctx.clearRect(0, 0, width, height);
    if (raw) {
      project(raw, smooth, points, alpha);
    }
    if (faceVisible) {
      project(faceRaw, faceSmooth, facePoints, 1 - Math.exp(-dt / 0.035));
      for (const name of Object.keys(expression)) {
        // Fast expression response preserves blinks while positions stay smooth.
        expression[name] = lerp(expression[name], expressionTarget[name], 1 - Math.exp(-dt / 0.018));
      }
    }
    if (visible) drawCharacter();
    if (faceVisible) drawFace();
    if (showDebug) { if (raw) drawDebug(); if (faceVisible) drawFaceDebug(); }
    raf = requestAnimationFrame(frame);
  } catch (error) {
    console.error(error);
    stopCamera();
    closeModels();
    say('トラッキングに失敗しました。カメラを再度開始してください。');
  }
}

function updateFace(result) {
  const candidate = result.faceLandmarks[0];
  const valid = candidate?.length >= 478 && candidate.every(p => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (!valid) { faceVisible = false; faceRaw = null; return; }
  faceRaw = candidate;
  for (const name of Object.keys(expressionTarget)) expressionTarget[name] = 0;
  for (const category of result.faceBlendshapes[0]?.categories ?? []) {
    if (Object.hasOwn(expressionTarget, category.categoryName)) {
      expressionTarget[category.categoryName] = clamp(category.score);
    }
  }
  if (!faceVisible) {
    faceRaw.forEach((p, i) => { faceSmooth[i].x = p.x; faceSmooth[i].y = p.y; });
    Object.assign(expression, expressionTarget);
  }
  faceVisible = true;
}

function project(source, filtered, output, alpha) {
  // Both models share the video's mirrored object-fit: cover transform.
  const scale = Math.max(width / video.videoWidth, height / video.videoHeight);
  const dw = video.videoWidth * scale, dh = video.videoHeight * scale;
  for (let i = 0; i < source.length; i++) {
    filtered[i].x = lerp(filtered[i].x, source[i].x, alpha);
    filtered[i].y = lerp(filtered[i].y, source[i].y, alpha);
    output[i].x = width - (filtered[i].x * dw + (width - dw) / 2);
    output[i].y = filtered[i].y * dh + (height - dh) / 2;
  }
}

function circle(x, y, radius, fill, stroke = '#142330') {
  ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = fill; ctx.fill();
  ctx.strokeStyle = stroke; ctx.lineWidth = Math.max(2, radius * 0.13); ctx.stroke();
}
function segment(a, b, thickness, color, squared = false) {
  ctx.lineCap = squared ? 'butt' : 'round';
  ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
  ctx.strokeStyle = '#142330'; ctx.lineWidth = thickness + 5; ctx.stroke();
  ctx.strokeStyle = color; ctx.lineWidth = thickness; ctx.stroke();
}
function drawCharacter() {
  const l = points[11], r = points[12], nose = points[0];
  const x = (l.x + r.x) / 2, y = (l.y + r.y) / 2;
  const size = Math.hypot(r.x - l.x, r.y - l.y);
  // In a mirrored frontal view the person's left shoulder is screen-left.
  const angle = Math.atan2(r.y - l.y, r.x - l.x);
  // Torso local origin sits exactly at the shoulder midpoint.
  ctx.save(); ctx.translate(x, y); ctx.rotate(angle); ctx.scale(size, size);
  ctx.beginPath(); ctx.moveTo(-0.48, -0.09); ctx.lineTo(0.48, -0.09);
  ctx.lineTo(0.35, 1.05); ctx.quadraticCurveTo(0, 1.2, -0.35, 1.05); ctx.closePath();
  ctx.fillStyle = theme.body; ctx.fill(); ctx.strokeStyle = '#142330'; ctx.lineWidth = 0.025; ctx.stroke();
  ctx.fillStyle = theme.accent; ctx.fillRect(-0.22, 0.22, 0.44, 0.32);
  ctx.fillStyle = '#142330'; ctx.fillRect(-0.12, 0.32, 0.24, 0.05);
  ctx.restore();
  // Anatomical left = mint; anatomical right = coral. Elbows stay articulated.
  for (const [s, e, w, color] of [[11, 13, 15, theme.arm], [12, 14, 16, theme.accent]]) {
    segment(points[s], points[e], size * 0.19, color);
    segment(points[e], points[w], size * 0.14, theme.skin, true);
    circle(points[s].x, points[s].y, size * 0.12, color);
    circle(points[e].x, points[e].y, size * 0.10, '#f3f7f8');
    circle(points[w].x, points[w].y, size * 0.10, color);
  }
  // Face graphics are drawn separately, so they also work without visible arms.
  if (faceVisible) {
    segment({ x, y }, facePoints[152], size * 0.13, theme.skin);
  }
}

function drawFace() {
  const p = facePoints;
  const left = p[263], right = p[33];
  const angle = Math.atan2(right.y - left.y, right.x - left.x);
  const headWidth = distance(p[234], p[454]);
  const headHeight = distance(p[10], p[152]);
  if (headWidth < 10 || headHeight < 10) return;
  const cx = (p[234].x + p[454].x) / 2;
  const cy = (p[10].y + p[152].y) / 2;
  ctx.save(); ctx.translate(cx, cy); ctx.rotate(angle);
  drawAccessories(headWidth, headHeight);
  ctx.beginPath();
  if (selected.template === 'robot') ctx.roundRect(-headWidth * 0.57, -headHeight * 0.54, headWidth * 1.14, headHeight * 1.08, headWidth * 0.15);
  else ctx.ellipse(0, 0, headWidth * 0.57, headHeight * 0.57, 0, 0, Math.PI * 2);
  ctx.fillStyle = theme.skin; ctx.fill();
  ctx.strokeStyle = '#142330'; ctx.lineWidth = Math.max(2, headWidth * 0.02); ctx.stroke();
  ctx.restore();
  for (const eye of EYES) drawEye(eye);
  drawMouth();

  if (selected.template === 'cat') drawWhiskers();
  if (selected.template !== 'gentleman') return;
  // Keep the moustache above the upper lip as the mouth opens.
  const nose = p[1], lip = p[0];
  const size = distance(p[61], p[291]) * 2.0;
  ctx.save();
  ctx.translate(lerp(nose.x, lip.x, 0.70), lerp(nose.y, lip.y, 0.70));
  ctx.rotate(angle); ctx.scale(size, size);
  ctx.fillStyle = '#302031'; ctx.strokeStyle = '#142330'; ctx.lineWidth = 0.012;
  for (const side of [-1, 1]) {
    ctx.save(); ctx.scale(side, 1);
    ctx.beginPath(); ctx.moveTo(0, 0.01);
    ctx.bezierCurveTo(0.06, -0.07, 0.12, -0.035, 0.17, 0.005);
    ctx.bezierCurveTo(0.22, 0.045, 0.26, 0.01, 0.265, -0.035);
    ctx.bezierCurveTo(0.30, 0.12, 0.13, 0.16, 0, 0.065);
    ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.restore();
  }
  ctx.restore();
}

function drawEye(eye) {
  let a = facePoints[eye.corners[0]], b = facePoints[eye.corners[1]];
  if (a.x > b.x) [a, b] = [b, a];
  const x = (a.x + b.x) / 2, y = (a.y + b.y) / 2;
  const angle = Math.atan2(b.y - a.y, b.x - a.x);
  const w = Math.max(3, distance(a, b) * 0.76);
  // Each anatomical eye has its own real eyeBlink blendshape score.
  const open = 1 - clamp((expression[eye.blink] - 0.08) / 0.72);
  const h = w * 0.72 * Math.max(0.025, open);
  const iris = facePoints[eye.iris];
  const dx = iris.x - x, dy = iris.y - y;
  const ix = clamp(dx * Math.cos(angle) + dy * Math.sin(angle), -w * 0.38, w * 0.38);
  const iy = clamp(-dx * Math.sin(angle) + dy * Math.cos(angle), -w * 0.20, w * 0.20);
  ctx.save(); ctx.translate(x, y); ctx.rotate(angle);
  ctx.lineWidth = Math.max(2, w * 0.075); ctx.strokeStyle = '#142330';
  ctx.beginPath(); ctx.ellipse(0, 0, w, h, 0, 0, Math.PI * 2);
  ctx.fillStyle = '#fff'; ctx.fill(); ctx.stroke();
  if (open > 0.08) {
    ctx.save(); ctx.clip();
    circle(ix, iy, w * 0.48, theme.iris);
    circle(ix, iy, w * 0.23, '#142330');
    circle(ix - w * 0.12, iy - w * 0.16, w * 0.09, '#fff', '#fff');
    ctx.restore();
  }
  // Expressive illustrated eyebrow follows this eye's position and rotation.
  ctx.beginPath(); ctx.moveTo(-w * 0.8, -w * 0.95);
  ctx.quadraticCurveTo(0, -w * (1.18 + 0.12 * open), w * 0.8, -w * 0.95);
  ctx.strokeStyle = '#302031'; ctx.lineWidth = Math.max(3, w * 0.12);
  ctx.lineCap = 'round'; ctx.stroke();
  ctx.restore();
}

function drawMouth() {
  const p = facePoints;
  let a = p[61], b = p[291];
  if (a.x > b.x) [a, b] = [b, a];
  const x = (p[13].x + p[14].x) / 2, y = (p[13].y + p[14].y) / 2;
  const angle = Math.atan2(b.y - a.y, b.x - a.x);
  const w = Math.max(4, distance(a, b) * 0.62);
  // Lip gap preserves actual lip motion; jawOpen exaggerates it for the cartoon.
  const lipGap = distance(p[13], p[14]);
  const opening = clamp(Math.max(lipGap / (w * 1.25), expression.jawOpen));
  const h = w * (0.055 + opening * 0.85);
  ctx.save(); ctx.translate(x, y); ctx.rotate(angle);
  ctx.lineWidth = Math.max(3, w * 0.13); ctx.strokeStyle = '#ee6d83';
  ctx.beginPath(); ctx.ellipse(0, 0, w, h, 0, 0, Math.PI * 2);
  ctx.fillStyle = '#381c35'; ctx.fill(); ctx.stroke();
  if (opening > 0.12) {
    ctx.save(); ctx.clip();
    ctx.fillStyle = '#fff8e9'; ctx.fillRect(-w * 0.78, -h, w * 1.56, h * 0.43);
    ctx.beginPath(); ctx.ellipse(0, h * 0.82, w * 0.56, h * 0.46, 0, 0, Math.PI * 2);
    ctx.fillStyle = '#ff83a7'; ctx.fill();
    ctx.restore();
  }
  ctx.restore();
}

function drawFaceDebug() {
  ctx.strokeStyle = '#ff65d3'; ctx.lineWidth = 1.5;
  for (const loop of FACE_LOOPS) {
    ctx.beginPath();
    loop.forEach((index, i) => {
      const p = facePoints[index];
      if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
  }
  for (const index of [1, 13, 14, 468, 473]) {
    const p = facePoints[index]; circle(p.x, p.y, 2, '#ff65d3');
  }
}

function drawDebug() {
  ctx.lineWidth = 2; ctx.strokeStyle = '#45efff'; ctx.lineCap = 'round';
  for (const [a, b] of LINKS) {
    if (!confidence(raw[a]) || !confidence(raw[b])) continue;
    ctx.beginPath(); ctx.moveTo(points[a].x, points[a].y);
    ctx.lineTo(points[b].x, points[b].y); ctx.stroke();
  }
  for (let i = 0; i < raw.length; i++) {
    if (confidence(raw[i])) circle(points[i].x, points[i].y, 3, '#45efff');
  }
}

function drawAccessories(w, h) {
  ctx.fillStyle = theme.skin; ctx.strokeStyle = '#142330'; ctx.lineWidth = Math.max(2, w * 0.02);
  for (const side of [-1, 1]) {
    ctx.beginPath();
    if (selected.template === 'cat') {
      ctx.moveTo(side * w * 0.18, -h * 0.39);
      ctx.lineTo(side * w * 0.49, -h * 0.81);
      ctx.lineTo(side * w * 0.58, -h * 0.2);
    } else if (selected.template === 'bunny') {
      ctx.ellipse(side * w * 0.29, -h * 0.64, w * 0.14, h * 0.45, side * 0.17, 0, Math.PI * 2);
    } else if (selected.template === 'bear') {
      ctx.arc(side * w * 0.45, -h * 0.43, w * 0.22, 0, Math.PI * 2);
    } else if (selected.template === 'alien') {
      segment({ x: side * w * 0.24, y: -h * 0.4 }, { x: side * w * 0.38, y: -h * 0.76 }, w * 0.045, theme.skin);
      circle(side * w * 0.38, -h * 0.76, w * 0.10, theme.accent);
      continue;
    } else if (selected.template === 'robot') {
      ctx.rect(side * w * 0.55 - w * 0.1, -h * 0.15, w * 0.2, h * 0.28);
    } else continue;
    ctx.closePath(); ctx.fillStyle = theme.skin; ctx.fill(); ctx.stroke();
  }
  if (selected.template === 'robot') {
    segment({x: 0, y: -h * 0.45}, {x: 0, y: -h * 0.78}, w * 0.04, theme.accent);
    circle(0, -h * 0.78, w * 0.08, theme.accent);
  }
}

function drawWhiskers() {
  const p = facePoints, nose = p[1];
  const w = distance(p[234], p[454]);
  ctx.save(); ctx.translate(nose.x, nose.y);
  ctx.rotate(Math.atan2(p[33].y - p[263].y, p[33].x - p[263].x));
  ctx.strokeStyle = '#593728'; ctx.lineWidth = Math.max(1.5, w * 0.012);
  for (const side of [-1, 1]) for (const offset of [-1, 1]) {
    ctx.beginPath(); ctx.moveTo(side * w * 0.2, w * 0.05);
    ctx.lineTo(side * w * 0.58, w * (0.05 + offset * 0.10)); ctx.stroke();
  }
  ctx.restore();
}

// The previews use the same geometry and palettes as the AR renderer.
function renderPreview(target, item) {
  const previous = selected, previousTheme = theme, previousCtx = ctx, previousFace = faceVisible;
  const previousExpression = { ...expression };
  Object.assign(expression, { eyeBlinkLeft: 0, eyeBlinkRight: 0, jawOpen: 0.15 });
  selected = item; theme = { ...PRESETS[item.template], ...item.colors }; ctx = target.getContext('2d');
  target.width = 460; target.height = 400; ctx.scale(2, 2);
  facePoints.forEach(p => { p.x = 115; p.y = 93; });
  for (const [i, x, y] of [[263, 89, 80], [362, 104, 80], [33, 141, 80], [133, 126, 80], [473, 96, 80], [468, 134, 80], [234, 73, 94], [454, 157, 94], [10, 115, 48], [152, 115, 135], [61, 100, 113], [291, 130, 113], [13, 115, 110], [14, 115, 121], [1, 115, 98], [0, 115, 107]]) Object.assign(facePoints[i], {x,y});
  for (const [i,x,y] of [[11,72,143],[12,158,143],[13,47,164],[14,181,130],[15,35,142],[16,191,102]]) Object.assign(points[i], {x,y});
  faceVisible = true;
  ctx.save(); ctx.translate(11.5, 12); ctx.scale(0.9, 0.9); drawCharacter(); drawFace(); ctx.restore();
  ctx = previousCtx; selected = previous; theme = previousTheme; faceVisible = previousFace;
  Object.assign(expression, previousExpression);
}

let refreshVersion = 0;
function validateCatalog(data) {
  if (data?.version !== 1 || !Array.isArray(data.characters) || data.characters.length < 1 || data.characters.length > 50) throw new Error('1〜50件のキャラクターを含む設定ファイルを選んでください。');
  const ids = new Set();
  return data.characters.map(item => {
    if (!item || typeof item.id !== 'string' || !/^[a-zA-Z0-9_-]{1,80}$/.test(item.id) || ids.has(item.id)
      || typeof item.name !== 'string' || !item.name.trim() || item.name.length > 30
      || typeof item.description !== 'string' || item.description.length > 70
      || !Object.hasOwn(PRESETS, item.template)) throw new Error('設定の名前・ID・キャラクターの形を確認してください。');
    ids.add(item.id);
    const colors = {};
    for (const key of ['skin', 'body', 'arm', 'accent', 'iris', 'bg']) {
      if (item.colors?.[key] !== undefined) {
        if (!/^#[0-9a-fA-F]{6}$/.test(item.colors[key])) throw new Error('色は #RRGGBB 形式で指定してください。');
        colors[key] = item.colors[key];
      }
    }
    return { id: item.id, name: item.name.trim(), description: item.description, template: item.template, colors };
  });
}
function renderMenu() {
  const container = document.querySelector('#characters');
  container.replaceChildren();
  document.querySelector('#character-count').textContent = `全${catalog.length}種類`;
  catalog.forEach((item, index) => {
    const button = document.createElement('button'); button.className = 'character-card';
    button.setAttribute('aria-label', `${item.name}でARを開始`);
    const art = document.createElement('div'); art.className = 'character-art';
    art.style.setProperty('--card-bg', item.colors?.bg ?? PRESETS[item.template].bg);
    const preview = document.createElement('canvas'); preview.setAttribute('aria-hidden', 'true'); art.append(preview);
    const number = document.createElement('span'); number.className = 'card-number'; number.textContent = String(index + 1).padStart(2, '0'); art.append(number);
    const info = document.createElement('span'); info.className = 'card-info';
    const title = document.createElement('span'); title.className = 'card-title'; title.textContent = item.name;
    const arrow = document.createElement('span'); arrow.className = 'card-arrow'; arrow.textContent = '↗'; title.append(arrow);
    const desc = document.createElement('span'); desc.className = 'card-description'; desc.textContent = item.description;
    info.append(title, desc); button.append(art, info); container.append(button);
    renderPreview(preview, item);
    button.addEventListener('click', () => {
      selected = item; theme = { ...PRESETS[item.template], ...item.colors };
      menu.hidden = true; ar.hidden = false; resize();
      document.querySelector('#character-name').textContent = item.name;
      start.focus(); start.click();
    });
  });
}
async function refreshCatalog() {
  const revision = ++refreshVersion;
  try {
    const response = await fetch('characters.json', { cache: 'no-store' });
    if (!response.ok) throw new Error('キャラクター設定を読み込めませんでした。');
    const next = validateCatalog(await response.json());
    if (revision !== refreshVersion || menu.hidden) return;
    catalog = next;
    renderMenu();
  } catch (error) {
    // Keep the current/default characters usable if the catalog is unavailable.
    console.warn('キャラクター設定の読み込みに失敗しました。', error);
  }
}
window.addEventListener('focus', () => { if (!menu.hidden) refreshCatalog(); });
renderMenu();
refreshCatalog();
