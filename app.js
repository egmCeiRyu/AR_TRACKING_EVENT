// Pin the JS and WASM to the same release; no bundler is required.
const CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
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
let landmarker, stream, raw;
let running = false, visible = false, showDebug = false, raf = 0;
let width = 0, height = 0, lastVideo = -1, lastInference = 0, lastDraw = 0;
let lastResult = 0, fpsStart = 0, detections = 0;
const lerp = (a, b, t) => a + (b - a) * t;
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
  if (landmarker) return;
  const { FilesetResolver, PoseLandmarker } = await import(`${CDN}/vision_bundle.mjs`);
  const files = await FilesetResolver.forVisionTasks(`${CDN}/wasm`);
  const options = {
    baseOptions: { modelAssetPath: MODEL, delegate: 'GPU' },
    runningMode: 'VIDEO', numPoses: 1,
    minPoseDetectionConfidence: 0.6, minPosePresenceConfidence: 0.6,
    minTrackingConfidence: 0.6, outputSegmentationMasks: false
  };
  try { landmarker = await PoseLandmarker.createFromOptions(files, options); }
  catch {
    say('Trying CPU tracking…');
    options.baseOptions.delegate = 'CPU';
    landmarker = await PoseLandmarker.createFromOptions(files, options);
  }
}

function stopCamera() {
  running = false;
  cancelAnimationFrame(raf);
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  raw = null;
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
    say('Loading pose model…');
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
window.addEventListener('pagehide', () => { stopCamera(); landmarker?.close(); landmarker = null; });
document.addEventListener('visibilitychange', () => {
  if (document.hidden && running) { stopCamera(); say('Camera paused. Start to resume.'); }
});

function frame(now) {
  if (!running) return;
  try {
    if (video.readyState >= 2 && video.currentTime !== lastVideo && now - lastInference >= 1000 / 30) {
      lastVideo = video.currentTime;
      lastInference = now;
      const result = landmarker.detectForVideo(video, now);
      raw = result.landmarks[0] ?? null;
      lastResult = now;
      const good = !!raw && REQUIRED.every(i => confidence(raw[i]))
        && Math.hypot((raw[11].x - raw[12].x) * video.videoWidth,
          (raw[11].y - raw[12].y) * video.videoHeight) > 25;
      if (good && !visible) {
        raw.forEach((p, i) => { smooth[i].x = p.x; smooth[i].y = p.y; });
      }
      visible = good;
      say(good ? 'Tracking · move your arms!' : 'Show your head, shoulders, elbows, and wrists');
      detections++;
      if (now - fpsStart >= 1000) {
        fps.textContent = `${Math.round(detections * 1000 / (now - fpsStart))} FPS`;
        detections = 0;
        fpsStart = now;
      }
    }
    if (now - lastResult > 500) { visible = false; raw = null; say('Waiting for camera frames…'); }
    const alpha = 1 - Math.exp(-Math.min((now - lastDraw) / 1000, 0.1) / 0.065);
    lastDraw = now;
    ctx.clearRect(0, 0, width, height);
    if (raw) {
      // Match object-fit: cover, center crop, and CSS scaleX(-1) exactly.
      const scale = Math.max(width / video.videoWidth, height / video.videoHeight);
      const dw = video.videoWidth * scale, dh = video.videoHeight * scale;
      for (let i = 0; i < raw.length; i++) {
        smooth[i].x = lerp(smooth[i].x, raw[i].x, alpha);
        smooth[i].y = lerp(smooth[i].y, raw[i].y, alpha);
        points[i].x = width - (smooth[i].x * dw + (width - dw) / 2);
        points[i].y = smooth[i].y * dh + (height - dh) / 2;
      }
      if (visible) drawCharacter();
      if (showDebug) drawDebug();
    }
    raf = requestAnimationFrame(frame);
  } catch (error) {
    console.error(error);
    stopCamera();
    landmarker?.close(); landmarker = null;
    say('Tracking failed. Start camera to retry.');
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
  const hx = lerp(x, nose.x, 0.85), hy = nose.y - size * 0.04;
  segment({ x, y }, { x: hx, y: hy }, size * 0.13, '#ffe39a');
  ctx.save(); ctx.translate(hx, hy); ctx.rotate(angle * 0.5);
  ctx.beginPath(); ctx.ellipse(0, 0, size * 0.31, size * 0.34, 0, 0, Math.PI * 2);
  ctx.fillStyle = '#ffe39a'; ctx.fill(); ctx.strokeStyle = '#142330'; ctx.lineWidth = size * 0.025; ctx.stroke();
  circle(-size * 0.105, -size * 0.035, size * 0.035, '#142330');
  circle(size * 0.105, -size * 0.035, size * 0.035, '#142330');
  ctx.beginPath(); ctx.arc(0, size * 0.055, size * 0.12, 0.15, Math.PI - 0.15);
  ctx.lineWidth = size * 0.025; ctx.stroke();
  ctx.restore();
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
