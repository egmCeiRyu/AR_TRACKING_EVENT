// Pin the JS and WASM to the same release; no bundler is required.
const CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const video = document.querySelector('#camera');
const canvas = document.querySelector('#overlay');
const ctx = canvas.getContext('2d');
const start = document.querySelector('#start');
const debug = document.querySelector('#debug');
const status = document.querySelector('#status');
const fps = document.querySelector('#fps');
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

async function loadModel() {
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
      say('Tentando rastreamento pela CPU…');
      settings.baseOptions.delegate = 'CPU';
      return Task.createFromOptions(files, settings);
    }
  }
  try {
    if (!landmarker) landmarker = await createTask(PoseLandmarker, options);
    say('Carregando olhos e boca…');
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
  start.textContent = 'Start camera';
  fps.textContent = '— FPS';
}

start.addEventListener('click', async () => {
  if (running) { stopCamera(); say('Camera stopped'); return; }
  start.disabled = true;
  try {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      throw new Error('Camera requires HTTPS or localhost.');
    }
    say('Carregando corpo e rosto…');
    await loadModel();
    say('Allow camera access…');
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 } }
    });
    video.srcObject = stream;
    await video.play();
    stream.getVideoTracks()[0].addEventListener('ended', () => {
      stopCamera(); say('Camera disconnected. Start again.');
    }, { once: true });
    running = true;
    lastVideo = -1;
    lastPoseInference = -Infinity;
    lastInference = lastDraw = lastResult = 0;
    fpsStart = performance.now();
    detections = 0;
    start.textContent = 'Stop camera';
    say('Looking for your upper body…');
    raf = requestAnimationFrame(frame);
  } catch (error) {
    stopCamera();
    console.error(error);
    say(error.name === 'NotAllowedError' ? 'Camera permission denied. Allow access and retry.'
      : error.name === 'NotFoundError' ? 'No camera found.'
      : error.name === 'NotReadableError' ? 'Camera unavailable. Close other camera apps.'
      : `Could not start: ${error.message}`);
  } finally { start.disabled = false; }
});
debug.addEventListener('click', () => {
  showDebug = !showDebug;
  debug.setAttribute('aria-pressed', String(showDebug));
});
window.addEventListener('pagehide', () => { stopCamera(); closeModels(); });
document.addEventListener('visibilitychange', () => {
  if (document.hidden && running) { stopCamera(); say('Camera paused. Start to resume.'); }
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
        ? visible ? 'Corpo + rosto · pisque e abra a boca!' : 'Rosto ativo · afaste-se para mostrar os braços'
        : visible ? 'Corpo ativo · olhe para a câmera' : 'Mostre seu rosto para a câmera');
      detections++;
      if (now - fpsStart >= 1000) {
        fps.textContent = `${Math.round(detections * 1000 / (now - fpsStart))} FPS`;
        detections = 0;
        fpsStart = now;
      }
    }
    if (now - lastResult > 500) {
      visible = faceVisible = false; raw = faceRaw = null;
      say('Aguardando imagens da câmera…');
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
    say('Tracking failed. Start camera to retry.');
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
  ctx.fillStyle = '#8c7dff'; ctx.fill(); ctx.strokeStyle = '#142330'; ctx.lineWidth = 0.025; ctx.stroke();
  ctx.fillStyle = '#d4ceff'; ctx.fillRect(-0.22, 0.22, 0.44, 0.32);
  ctx.fillStyle = '#142330'; ctx.fillRect(-0.12, 0.32, 0.24, 0.05);
  ctx.restore();
  // Anatomical left = mint; anatomical right = coral. Elbows stay articulated.
  for (const [s, e, w, color] of [[11, 13, 15, '#9effca'], [12, 14, 16, '#ff927e']]) {
    segment(points[s], points[e], size * 0.19, color);
    segment(points[e], points[w], size * 0.14, '#ffe39a', true);
    circle(points[s].x, points[s].y, size * 0.12, color);
    circle(points[e].x, points[e].y, size * 0.10, '#f3f7f8');
    circle(points[w].x, points[w].y, size * 0.10, color);
  }
  // Face graphics are drawn separately, so they also work without visible arms.
  if (faceVisible) {
    segment({ x, y }, facePoints[152], size * 0.13, '#ffe39a');
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
  ctx.beginPath(); ctx.ellipse(0, 0, headWidth * 0.57, headHeight * 0.57, 0, 0, Math.PI * 2);
  ctx.fillStyle = '#ffe39a'; ctx.fill();
  ctx.strokeStyle = '#142330'; ctx.lineWidth = Math.max(2, headWidth * 0.02); ctx.stroke();
  ctx.restore();
  for (const eye of EYES) drawEye(eye);
  drawMouth();

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
    circle(ix, iy, w * 0.48, eye.color);
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
