/**
 * MediaPipe 21 3D Hand Landmark Tracking & Perception Module
 * Extracts spatial coordinates, finger curvature, palm orientation, and bounding boxes.
 */

class MediaPipeTracker {
  constructor(videoElement, canvasElement, onLandmarksCallback) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.onLandmarks = onLandmarksCallback;
    this.hands = null;
    this.camera = null;
    this.isRunning = false;
    this.lastFrameTime = performance.now();
    this.fps = 60;
    this.lastLandmarks = null;

    this._initMediaPipe();
  }

  _initMediaPipe() {
    if (typeof Hands === 'undefined') {
      console.warn('[MediaPipeTracker] MediaPipe Hands library not loaded yet.');
      return;
    }

    this.hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    this.hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.6
    });

    this.hands.onResults((results) => this._onResults(results));
  }

  async startCamera() {
    if (!this.hands) {
      this._initMediaPipe();
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
      });
      this.video.srcObject = stream;
      await this.video.play();
      this.isRunning = true;
      this._processVideoLoop();
      return true;
    } catch (err) {
      console.warn('[MediaPipeTracker] Camera access failed or denied:', err);
      return false;
    }
  }

  stopCamera() {
    this.isRunning = false;
    if (this.video && this.video.srcObject) {
      const tracks = this.video.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      this.video.srcObject = null;
    }
  }

  async _processVideoLoop() {
    if (!this.isRunning) return;

    if (this.video.readyState >= 2 && this.hands) {
      const now = performance.now();
      const delta = (now - this.lastFrameTime) / 1000;
      this.fps = Math.round(1 / delta);
      this.lastFrameTime = now;

      await this.hands.send({ image: this.video });
    }

    requestAnimationFrame(() => this._processVideoLoop());
  }

  _onResults(results) {
    this.canvas.width = this.video.videoWidth || 640;
    this.canvas.height = this.video.videoHeight || 480;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    let perceptionData = {
      handDetected: false,
      handedness: 'Right',
      landmarks: null,
      palmTiltDeg: 0,
      pinchDist: 0.8,
      motionSpeed: 'Steady',
      bbox: null,
      croppedFrameB64: null,
      fps: this.fps
    };

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      perceptionData.handDetected = true;
      const landmarks = results.multiHandLandmarks[0];
      perceptionData.landmarks = landmarks;

      if (results.multiHandedness && results.multiHandedness.length > 0) {
        perceptionData.handedness = results.multiHandedness[0].label;
      }

      // Draw futuristic neon skeletal overlay
      this._drawNeonSkeleton(ctx, landmarks, this.canvas.width, this.canvas.height);

      // Compute geometric metrics
      perceptionData.palmTiltDeg = this._calculatePalmTilt(landmarks);
      perceptionData.pinchDist = this._calculatePinchDistance(landmarks);
      perceptionData.motionSpeed = this._calculateMotionVelocity(landmarks);

      // Compute bounding box
      perceptionData.bbox = this._calculateBoundingBox(landmarks, this.canvas.width, this.canvas.height);
      this._drawBoundingBox(ctx, perceptionData.bbox);

      // Extract cropped bounding box for server-side ML model
      perceptionData.croppedFrameB64 = this._cropHandRegion(this.video, perceptionData.bbox);
      this.lastLandmarks = landmarks;
    }

    if (this.onLandmarks) {
      this.onLandmarks(perceptionData);
    }
  }

  _drawNeonSkeleton(ctx, landmarks, width, height) {
    const CONNECTIONS = [
      [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8], // Index
      [5, 9], [9, 10], [10, 11], [11, 12], // Middle
      [9, 13], [13, 14], [14, 15], [15, 16], // Ring
      [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
      [0, 17] // Palm Base
    ];

    // Draw connecting lines with neon cyan glow
    ctx.save();
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#00f2fe';
    ctx.shadowColor = '#00f2fe';
    ctx.shadowBlur = 10;

    for (const [start, end] of CONNECTIONS) {
      const p1 = landmarks[start];
      const p2 = landmarks[end];
      ctx.beginPath();
      ctx.moveTo(p1.x * width, p1.y * height);
      ctx.lineTo(p2.x * width, p2.y * height);
      ctx.stroke();
    }

    // Draw landmark joint points
    for (let i = 0; i < landmarks.length; i++) {
      const p = landmarks[i];
      ctx.beginPath();
      ctx.arc(p.x * width, p.y * height, i % 4 === 0 ? 5 : 3.5, 0, 2 * Math.PI);
      ctx.fillStyle = i === 8 || i === 4 ? '#7f5af0' : '#10b981';
      ctx.shadowColor = '#10b981';
      ctx.shadowBlur = 8;
      ctx.fill();
    }
    ctx.restore();
  }

  _drawBoundingBox(ctx, bbox) {
    if (!bbox) return;
    ctx.save();
    ctx.strokeStyle = 'rgba(0, 242, 254, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 6]);
    ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);
    ctx.restore();
  }

  _calculateBoundingBox(landmarks, width, height) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const lm of landmarks) {
      const px = lm.x * width;
      const py = lm.y * height;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    }

    const padding = 30;
    const x = Math.max(0, minX - padding);
    const y = Math.max(0, minY - padding);
    const w = Math.min(width - x, (maxX - minX) + padding * 2);
    const h = Math.min(height - y, (maxY - minY) + padding * 2);

    return { x, y, width: Math.max(w, h), height: Math.max(w, h) };
  }

  _cropHandRegion(video, bbox) {
    if (!bbox || bbox.width <= 0 || bbox.height <= 0) return null;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 120;
    tempCanvas.height = 120;
    const tempCtx = tempCanvas.getContext('2d');
    
    try {
      tempCtx.drawImage(
        video,
        bbox.x, bbox.y, bbox.width, bbox.height,
        0, 0, 120, 120
      );
      return tempCanvas.toDataURL('image/jpeg', 0.85);
    } catch (e) {
      return null;
    }
  }

  _calculatePalmTilt(landmarks) {
    const wrist = landmarks[0];
    const middleBase = landmarks[9];
    const dx = middleBase.x - wrist.x;
    const dy = middleBase.y - wrist.y;
    const rad = Math.atan2(dy, dx);
    let deg = Math.round(rad * (180 / Math.PI));
    return Math.abs(deg);
  }

  _calculatePinchDistance(landmarks) {
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const dist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);
    return Math.round(dist * 100) / 100;
  }

  _calculateMotionVelocity(landmarks) {
    if (!this.lastLandmarks) return 'Steady';
    const wristCurrent = landmarks[0];
    const wristPrev = this.lastLandmarks[0];
    const delta = Math.hypot(wristCurrent.x - wristPrev.x, wristCurrent.y - wristPrev.y);
    if (delta > 0.08) return 'Fast Motion';
    if (delta > 0.03) return 'Transitioning';
    return 'Steady Hold';
  }
}
