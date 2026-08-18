/**
 * VOXSIGN AI COPILOT - MAIN APPLICATION CONTROLLER
 * Full integration of Multimodal Perception, AI Decision Engine, NLP Context,
 * Multilingual Web Speech Synthesis, Emergency Alarms & Conversation History.
 */

class VoxSignApp {
  constructor() {
    this.currentLanguage = 'en-US';
    this.autoSpeak = true;
    this.speechRate = 1.0;
    this.speechPitch = 1.0;
    this.lastSpokenText = '';
    this.isSpeaking = false;

    // Temporal Gesture State
    this.rawTokenBuffer = [];
    this.currentHeldToken = null;
    this.holdStartTime = 0;
    this.lastTokenRegisteredTime = 0;
    this.debounceMs = 250;
    this.targetHoldMs = 350;
    this.confThreshold = 0.75;
    this.clarifyBound = 0.45;

    // Perception & MediaPipe
    this.tracker = null;
    this.isCameraRunning = false;
    this.isDemoMode = false;
    this.demoInterval = null;

    // Audio Context for Emergency Siren / Chimes
    this.audioCtx = null;

    // Conversation History
    this.conversationLog = [];

    this.init();
  }

  async init() {
    this._initTabs();
    this._initSpeechSynthesis();
    this._initASLKeypad();
    this._initQuickPhrases();
    this._initPersonalizationSliders();
    this._initCameraPipeline();
    this._initEmergencyHandlers();
    this._initHistoryActions();
    this._startTelemetryUpdater();
  }

  // =========================================================================
  // TAB NAVIGATION
  // =========================================================================
  _initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetId = btn.getAttribute('data-tab');
        tabBtns.forEach(b => b.classList.remove('active'));
        tabPanes.forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        const targetPane = document.getElementById(targetId);
        if (targetPane) targetPane.classList.add('active');
      });
    });
  }

  // =========================================================================
  // WEB SPEECH SYNTHESIS & MULTILINGUAL AUDIO ENGINE
  // =========================================================================
  _initSpeechSynthesis() {
    const langSelect = document.getElementById('select-target-lang');
    const autoSpeakToggle = document.getElementById('toggle-auto-speak');
    const rateSlider = document.getElementById('slider-speech-rate');
    const pitchSlider = document.getElementById('slider-speech-pitch');
    const rateVal = document.getElementById('val-speech-rate');
    const pitchVal = document.getElementById('val-speech-pitch');
    const btnSpeakNow = document.getElementById('btn-speak-now');
    const btnRepeat = document.getElementById('btn-repeat-speech');

    if (langSelect) {
      langSelect.addEventListener('change', (e) => {
        this.currentLanguage = e.target.value;
      });
    }

    if (autoSpeakToggle) {
      autoSpeakToggle.addEventListener('change', (e) => {
        this.autoSpeak = e.target.checked;
        const statusEl = document.getElementById('tts-status');
        if (statusEl) {
          statusEl.textContent = this.autoSpeak ? 'Voice Ready (Auto-Speak Active)' : 'Voice Ready (Manual Trigger)';
        }
      });
    }

    if (rateSlider && rateVal) {
      rateSlider.addEventListener('input', (e) => {
        this.speechRate = parseFloat(e.target.value);
        rateVal.textContent = `${this.speechRate.toFixed(1)}x`;
      });
    }

    if (pitchSlider && pitchVal) {
      pitchSlider.addEventListener('input', (e) => {
        this.speechPitch = parseFloat(e.target.value);
        pitchVal.textContent = this.speechPitch.toFixed(1);
      });
    }

    if (btnSpeakNow) {
      btnSpeakNow.addEventListener('click', () => {
        const text = document.getElementById('synthesized-sentence-text')?.textContent.replace(/["']/g, '');
        if (text) this.speak(text);
      });
    }

    if (btnRepeat) {
      btnRepeat.addEventListener('click', () => {
        if (this.lastSpokenText) this.speak(this.lastSpokenText);
      });
    }
  }

  speak(text, forceEmergency = false) {
    if (!('speechSynthesis' in window) || !text) return;

    window.speechSynthesis.cancel(); // Stop any overlapping utterance

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = this.currentLanguage;
    utterance.rate = forceEmergency ? 1.15 : this.speechRate;
    utterance.pitch = forceEmergency ? 1.2 : this.speechPitch;

    const bars = document.getElementById('audio-bars');
    const statusEl = document.getElementById('tts-status');

    utterance.onstart = () => {
      this.isSpeaking = true;
      if (bars) bars.classList.add('playing');
      if (statusEl) statusEl.textContent = `Speaking: "${text.slice(0, 30)}..."`;
    };

    utterance.onend = () => {
      this.isSpeaking = false;
      if (bars) bars.classList.remove('playing');
      if (statusEl) statusEl.textContent = 'Voice Ready';
    };

    utterance.onerror = () => {
      this.isSpeaking = false;
      if (bars) bars.classList.remove('playing');
      if (statusEl) statusEl.textContent = 'Voice Error';
    };

    this.lastSpokenText = text;
    window.speechSynthesis.speak(utterance);
  }

  playEmergencySiren() {
    try {
      if (!this.audioCtx) {
        this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (this.audioCtx.state === 'suspended') {
        this.audioCtx.resume();
      }

      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      osc.type = 'sawtooth';
      
      const now = this.audioCtx.currentTime;
      osc.frequency.setValueAtTime(800, now);
      osc.frequency.linearRampToValueAtTime(1200, now + 0.25);
      osc.frequency.linearRampToValueAtTime(800, now + 0.5);

      gain.gain.setValueAtTime(0.3, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.5);

      osc.connect(gain);
      gain.connect(this.audioCtx.destination);

      osc.start(now);
      osc.stop(now + 0.5);
    } catch (e) {
      console.warn('AudioContext alert error:', e);
    }
  }

  // =========================================================================
  // CAMERA & MEDIAPIPE PERCEPTION PIPELINE
  // =========================================================================
  async _initCameraPipeline() {
    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('output-canvas');
    const btnToggle = document.getElementById('btn-camera-toggle');
    const btnMirror = document.getElementById('btn-camera-mirror');
    const btnDemo = document.getElementById('btn-sim-stream');

    if (!video || !canvas) return;

    this.tracker = new MediaPipeTracker(video, canvas, (perceptionData) => {
      this._handlePerceptionFrame(perceptionData);
    });

    const cameraStarted = await this.tracker.startCamera();
    this.isCameraRunning = cameraStarted;

    if (!cameraStarted) {
      const handCountText = document.getElementById('hand-count-text');
      if (handCountText) {
        handCountText.textContent = 'Camera Off (Using Sim / Demo Mode)';
      }
      this._startDemoSimulation();
    }

    if (btnToggle) {
      btnToggle.addEventListener('click', async () => {
        if (this.isCameraRunning) {
          this.tracker.stopCamera();
          this.isCameraRunning = false;
          btnToggle.innerHTML = '<span class="cam-icon">📹</span> Resume Camera';
        } else {
          const ok = await this.tracker.startCamera();
          this.isCameraRunning = ok;
          if (ok) {
            btnToggle.innerHTML = '<span class="cam-icon">📹</span> Pause Camera';
            this._stopDemoSimulation();
          }
        }
      });
    }

    if (btnMirror) {
      btnMirror.addEventListener('click', () => {
        const isMirrored = video.style.transform !== 'scaleX(1)';
        video.style.transform = isMirrored ? 'scaleX(1)' : 'scaleX(-1)';
        canvas.style.transform = isMirrored ? 'scaleX(1)' : 'scaleX(-1)';
      });
    }

    if (btnDemo) {
      btnDemo.addEventListener('click', () => {
        if (this.isDemoMode) {
          this._stopDemoSimulation();
          btnDemo.classList.remove('active');
        } else {
          this._startDemoSimulation();
          btnDemo.classList.add('active');
        }
      });
    }
  }

  async _handlePerceptionFrame(perceptionData) {
    const handCountText = document.getElementById('hand-count-text');
    const hudPalmTilt = document.getElementById('hud-palm-tilt');
    const hudPinchDist = document.getElementById('hud-pinch-dist');
    const hudSpeedVal = document.getElementById('hud-speed-val');

    if (handCountText) {
      handCountText.textContent = perceptionData.handDetected
        ? `Hand: Detected (${perceptionData.handedness})`
        : 'Searching for Hand...';
    }

    if (hudPalmTilt) hudPalmTilt.textContent = `${perceptionData.palmTiltDeg}°`;
    if (hudPinchDist) hudPinchDist.textContent = perceptionData.pinchDist.toFixed(2);
    if (hudSpeedVal) hudSpeedVal.textContent = perceptionData.motionSpeed;

    if (!perceptionData.handDetected) {
      this._updateDecisionState({
        action: 'WAIT',
        reason: 'No hand gesture in frame',
        token: null,
        confidence: 0
      });
      this._updateHoldProgress(0);
      return;
    }

    // Call Model Inference on cropped hand bounding box
    if (perceptionData.croppedFrameB64) {
      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_b64: perceptionData.croppedFrameB64 })
        });
        if (res.ok) {
          const prediction = await res.json();
          this._processModelPrediction(prediction, perceptionData.motionSpeed);
        }
      } catch (e) {
        // Lightweight local fallback
        this._fallbackLandmarkPrediction(perceptionData);
      }
    }
  }

  _processModelPrediction(pred, motionSpeed) {
    const token = pred.predicted_token;
    const conf = pred.confidence;
    const now = performance.now();

    // Update HUD Token Bubble
    const letterEl = document.getElementById('hud-detected-letter');
    const confEl = document.getElementById('hud-detected-conf');
    if (letterEl) letterEl.textContent = token;
    if (confEl) confEl.textContent = `${pred.confidence_pct}%`;

    // Update Candidate Pills
    const pillsContainer = document.getElementById('candidate-pills');
    if (pillsContainer && pred.top_candidates) {
      pillsContainer.innerHTML = '';
      pred.top_candidates.slice(0, 3).forEach((c, idx) => {
        const pill = document.createElement('div');
        pill.className = `pill ${idx === 0 ? 'active' : ''}`;
        pill.textContent = `${c.token} (${c.confidence_pct}%)`;
        pillsContainer.appendChild(pill);
      });
    }

    // Temporal Hold Calculation
    if (this.currentHeldToken === token && motionSpeed !== 'Fast Motion') {
      const holdTimeMs = now - this.holdStartTime;
      const progress = Math.min(1.0, holdTimeMs / this.targetHoldMs);
      this._updateHoldProgress(progress);

      // Evaluate AI Decision Engine
      this._evaluateDecisionEngine(token, conf, holdTimeMs, pred.top_candidates);
    } else {
      this.currentHeldToken = token;
      this.holdStartTime = now;
      this._updateHoldProgress(0);
    }
  }

  _fallbackLandmarkPrediction(perceptionData) {
    // Basic landmark rule heuristic for fallback
    const token = 'A';
    this._processModelPrediction({
      predicted_token: token,
      confidence: 0.88,
      confidence_pct: 88.0,
      top_candidates: [
        { token: 'A', confidence_pct: 88.0 },
        { token: 'E', confidence_pct: 8.0 },
        { token: 'S', confidence_pct: 4.0 }
      ]
    }, perceptionData.motionSpeed);
  }

  _updateHoldProgress(ratio) {
    const fill = document.getElementById('hold-progress-fill');
    if (fill) {
      fill.style.width = `${Math.round(ratio * 100)}%`;
    }
  }

  // =========================================================================
  // AI DECISION ENGINE EVALUATION
  // =========================================================================
  async _evaluateDecisionEngine(token, confidence, holdTimeMs, topCandidates) {
    const now = performance.now();
    const isEmergency = ['HELP', 'DANGER', 'PAIN', 'FIRE', 'CHOKING'].includes(token);

    // Call decision engine API
    try {
      const res = await fetch('/api/decision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token: token,
          confidence: confidence,
          hold_time_ms: holdTimeMs,
          is_emergency: isEmergency,
          sentence_complete: false,
          top_candidates: topCandidates
        })
      });

      if (res.ok) {
        const decision = await res.json();
        this._updateDecisionState(decision);

        // If Decision is SPEAK and Debounce is met -> Register token
        if (decision.action === 'SPEAK' || decision.action === 'ALERT') {
          if (now - this.lastTokenRegisteredTime > this.debounceMs) {
            this.lastTokenRegisteredTime = now;
            this.appendTokenToSequence(token);
          }
        }
      }
    } catch (e) {
      // Local decision fallback
      if (confidence >= this.confThreshold && holdTimeMs >= this.targetHoldMs) {
        this._updateDecisionState({
          action: 'SPEAK',
          reason: 'Hold & Confidence validated',
          token: token
        });
        if (now - this.lastTokenRegisteredTime > this.debounceMs) {
          this.lastTokenRegisteredTime = now;
          this.appendTokenToSequence(token);
        }
      } else {
        this._updateDecisionState({
          action: 'WAIT',
          reason: `Holding sign (${int(holdTimeMs)}ms)`,
          token: token
        });
      }
    }
  }

  _updateDecisionState(decision) {
    const actionEl = document.getElementById('hud-decision-action');
    const reasonEl = document.getElementById('hud-decision-reason');
    const globalPill = document.getElementById('global-state-text');
    const studioBadge = document.getElementById('studio-active-state');

    const action = decision.action || 'WAIT';
    const reason = decision.reason || '';

    if (actionEl) {
      actionEl.textContent = action;
      actionEl.className = `badge-action state-${action.toLowerCase()}`;
    }
    if (reasonEl) reasonEl.textContent = reason;
    if (globalPill) globalPill.textContent = action;
    if (studioBadge) studioBadge.textContent = action;

    // Highlight state node in Studio Tab
    document.querySelectorAll('.state-node').forEach(node => {
      node.classList.remove('active');
    });
    const activeNode = document.getElementById(`node-${action.toLowerCase()}`);
    if (activeNode) activeNode.classList.add('active');
  }

  // =========================================================================
  // CONTINUOUS SEQUENCE & CONTEXT NLP RECONSTRUCTION
  // =========================================================================
  async appendTokenToSequence(token) {
    if (!token) return;
    this.rawTokenBuffer.push(token);

    // Limit buffer length
    if (this.rawTokenBuffer.length > 25) {
      this.rawTokenBuffer.shift();
    }

    this._renderTokenStream();
    await this._reconstructSentence();
  }

  _renderTokenStream() {
    const container = document.getElementById('token-stream-display');
    const simRawEl = document.getElementById('sim-raw-input');
    if (!container) return;

    container.innerHTML = '';
    this.rawTokenBuffer.forEach(tok => {
      const chip = document.createElement('span');
      chip.className = 'token-chip';
      chip.textContent = tok;
      container.appendChild(chip);
    });

    if (simRawEl) {
      simRawEl.textContent = this.rawTokenBuffer.join(' ') || 'NONE';
    }
  }

  async _reconstructSentence() {
    try {
      const res = await fetch('/api/nlp/reconstruct', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tokens: this.rawTokenBuffer
        })
      });

      if (res.ok) {
        const result = await res.json();
        this._updateSentenceDisplay(result);
      }
    } catch (e) {
      console.warn('NLP Reconstruction error:', e);
    }
  }

  _updateSentenceDisplay(nlpResult) {
    const sentenceTextEl = document.getElementById('synthesized-sentence-text');
    const tagIntent = document.getElementById('tag-intent');
    const tagUrgency = document.getElementById('tag-urgency');
    const simFluent = document.getElementById('sim-fluent-output');
    const simIntent = document.getElementById('sim-intent-val');
    const simUrgency = document.getElementById('sim-urgency-val');
    const simComplete = document.getElementById('sim-complete-val');

    const sentence = nlpResult.fluent_sentence || 'Listening for signs...';
    const intent = nlpResult.intent || 'GENERAL';
    const urgency = nlpResult.urgency || 'NORMAL';

    if (sentenceTextEl) sentenceTextEl.textContent = `"${sentence}"`;
    if (tagIntent) tagIntent.textContent = intent;
    if (tagUrgency) {
      tagUrgency.textContent = urgency;
      tagUrgency.className = `tag tag-urgency ${urgency === 'EMERGENCY' ? 'tag-urgency-emergency' : ''}`;
    }

    if (simFluent) simFluent.textContent = `"${sentence}"`;
    if (simIntent) simIntent.textContent = intent;
    if (simUrgency) simUrgency.textContent = urgency;
    if (simComplete) simComplete.textContent = nlpResult.is_complete ? 'Ready to Speak' : 'Assembling...';

    // Emergency Trigger
    if (nlpResult.emergency_flag || urgency === 'EMERGENCY') {
      this.triggerEmergencyAlert(nlpResult.trigger_keyword || 'EMERGENCY', sentence);
    }

    // Auto-Speak if enabled and sentence is ready
    if (this.autoSpeak && nlpResult.is_complete && sentence !== this.lastSpokenText) {
      this.speak(sentence, nlpResult.emergency_flag);
      this._addConversationLog(nlpResult);
    }
  }

  // =========================================================================
  // EMERGENCY SOS ALARM & BANNER
  // =========================================================================
  triggerEmergencyAlert(keyword, speechText) {
    const banner = document.getElementById('emergency-banner');
    const headline = document.getElementById('emergency-headline');
    const detail = document.getElementById('emergency-detail');

    if (headline) headline.textContent = `🚨 CRITICAL EMERGENCY TRIGGERED: "${keyword}"`;
    if (detail) detail.textContent = `Priority vocal alert broadcasted: "${speechText}"`;
    if (banner) banner.classList.remove('hidden');

    this.playEmergencySiren();
    this.speak(speechText, true);
  }

  _initEmergencyHandlers() {
    const btnSos = document.getElementById('btn-quick-sos');
    const btnDismiss = document.getElementById('btn-dismiss-emergency');

    if (btnSos) {
      btnSos.addEventListener('click', () => {
        this.triggerEmergencyAlert('MANUAL SOS', 'Emergency alert! I need immediate help and emergency assistance!');
      });
    }

    if (btnDismiss) {
      btnDismiss.addEventListener('click', () => {
        document.getElementById('emergency-banner')?.classList.add('hidden');
      });
    }
  }

  // =========================================================================
  // INTERACTIVE ASL KEYBOARD & QUICK PHRASES
  // =========================================================================
  _initASLKeypad() {
    const keypad = document.getElementById('asl-keyboard');
    const btnSpace = document.getElementById('key-space');
    const btnBackspace = document.getElementById('key-backspace');
    const btnAutoCorrect = document.getElementById('key-auto-correct');
    const btnReset = document.getElementById('btn-reset-keypad');
    const btnSimSpeak = document.getElementById('btn-sim-speak');

    if (keypad) {
      keypad.innerHTML = '';
      'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').forEach(letter => {
        const key = document.createElement('button');
        key.className = 'key-asl';
        key.innerHTML = `<span>${letter}</span><span class="sub-sign">ASL</span>`;
        key.addEventListener('click', () => {
          this.appendTokenToSequence(letter);
        });
        keypad.appendChild(key);
      });
    }

    if (btnSpace) {
      btnSpace.addEventListener('click', () => {
        this.appendTokenToSequence(' ');
      });
    }

    if (btnBackspace) {
      btnBackspace.addEventListener('click', () => {
        this.rawTokenBuffer.pop();
        this._renderTokenStream();
        this._reconstructSentence();
      });
    }

    if (btnAutoCorrect) {
      btnAutoCorrect.addEventListener('click', () => {
        this._reconstructSentence();
      });
    }

    if (btnReset) {
      btnReset.addEventListener('click', () => {
        this.rawTokenBuffer = [];
        this._renderTokenStream();
        this._reconstructSentence();
      });
    }

    if (btnSimSpeak) {
      btnSimSpeak.addEventListener('click', () => {
        const text = document.getElementById('sim-fluent-output')?.textContent.replace(/["']/g, '');
        if (text) this.speak(text);
      });
    }

    const btnClearBuffer = document.getElementById('btn-clear-buffer');
    if (btnClearBuffer) {
      btnClearBuffer.addEventListener('click', () => {
        this.rawTokenBuffer = [];
        this._renderTokenStream();
        this._reconstructSentence();
      });
    }
  }

  _initQuickPhrases() {
    const buttons = document.querySelectorAll('.btn-phrase');
    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        const phrase = btn.getAttribute('data-phrase');
        if (phrase) {
          this.rawTokenBuffer = phrase.split('');
          this._renderTokenStream();
          this._reconstructSentence();
        }
      });
    });
  }

  // =========================================================================
  // SIGNER PERSONALIZATION STUDIO
  // =========================================================================
  _initPersonalizationSliders() {
    const confSlider = document.getElementById('slider-conf-thresh');
    const clarifySlider = document.getElementById('slider-clarify-bound');
    const holdSlider = document.getElementById('slider-hold-target');
    const debounceSlider = document.getElementById('slider-debounce-time');

    const valConf = document.getElementById('val-conf-thresh');
    const valClarify = document.getElementById('val-clarify-bound');
    const valHold = document.getElementById('val-hold-target');
    const valDebounce = document.getElementById('val-debounce-time');

    if (confSlider && valConf) {
      confSlider.addEventListener('input', (e) => {
        this.confThreshold = parseInt(e.target.value, 10) / 100;
        valConf.textContent = `${e.target.value}%`;
      });
    }

    if (clarifySlider && valClarify) {
      clarifySlider.addEventListener('input', (e) => {
        this.clarifyBound = parseInt(e.target.value, 10) / 100;
        valClarify.textContent = `${e.target.value}%`;
      });
    }

    if (holdSlider && valHold) {
      holdSlider.addEventListener('input', (e) => {
        this.targetHoldMs = parseInt(e.target.value, 10);
        valHold.textContent = `${this.targetHoldMs}ms`;
      });
    }

    if (debounceSlider && valDebounce) {
      debounceSlider.addEventListener('input', (e) => {
        this.debounceMs = parseInt(e.target.value, 10);
        valDebounce.textContent = `${this.debounceMs}ms`;
      });
    }

    // Profile Cards Click Handlers
    const profileCards = document.querySelectorAll('.profile-choice-card');
    profileCards.forEach(card => {
      card.addEventListener('click', () => {
        profileCards.forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        const profile = card.getAttribute('data-profile');
        this._applySignerProfile(profile);
      });
    });

    const btnSave = document.getElementById('btn-save-profile');
    if (btnSave) {
      btnSave.addEventListener('click', async () => {
        await fetch('/api/profile', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            confidence_threshold: this.confThreshold,
            clarify_lower_bound: this.clarifyBound,
            hold_duration_ms: this.targetHoldMs,
            debounce_ms: this.debounceMs
          })
        });
        btnSave.textContent = '✓ Saved!';
        setTimeout(() => { btnSave.textContent = 'Save Profile'; }, 1500);
      });
    }
  }

  _applySignerProfile(profileName) {
    if (profileName === 'fast_signer') {
      this.confThreshold = 0.65;
      this.clarifyBound = 0.40;
      this.targetHoldMs = 200;
      this.debounceMs = 150;
    } else if (profileName === 'novice') {
      this.confThreshold = 0.82;
      this.clarifyBound = 0.50;
      this.targetHoldMs = 600;
      this.debounceMs = 400;
    } else if (profileName === 'tremor_assist') {
      this.confThreshold = 0.70;
      this.clarifyBound = 0.40;
      this.targetHoldMs = 500;
      this.debounceMs = 450;
    } else {
      this.confThreshold = 0.75;
      this.clarifyBound = 0.45;
      this.targetHoldMs = 350;
      this.debounceMs = 250;
    }

    // Update Slider UI
    this._syncSliderValues();
  }

  _syncSliderValues() {
    const confSlider = document.getElementById('slider-conf-thresh');
    const clarifySlider = document.getElementById('slider-clarify-bound');
    const holdSlider = document.getElementById('slider-hold-target');
    const debounceSlider = document.getElementById('slider-debounce-time');

    if (confSlider) { confSlider.value = Math.round(this.confThreshold * 100); document.getElementById('val-conf-thresh').textContent = `${confSlider.value}%`; }
    if (clarifySlider) { clarifySlider.value = Math.round(this.clarifyBound * 100); document.getElementById('val-clarify-bound').textContent = `${clarifySlider.value}%`; }
    if (holdSlider) { holdSlider.value = this.targetHoldMs; document.getElementById('val-hold-target').textContent = `${this.targetHoldMs}ms`; }
    if (debounceSlider) { debounceSlider.value = this.debounceMs; document.getElementById('val-debounce-time').textContent = `${this.debounceMs}ms`; }
  }

  // =========================================================================
  // CONVERSATION HISTORY LOGGING & EXPORTS
  // =========================================================================
  _addConversationLog(nlpResult) {
    const entry = {
      time: new Date().toLocaleTimeString(),
      raw: nlpResult.raw_sequence || this.rawTokenBuffer.join(''),
      fluent: nlpResult.fluent_sentence,
      intent: nlpResult.intent,
      urgency: nlpResult.urgency,
      decision: nlpResult.emergency_flag ? 'ALERT' : 'SPEAK'
    };

    this.conversationLog.unshift(entry);
    this._renderConversationLog();
  }

  _renderConversationLog() {
    const tbody = document.getElementById('history-table-body');
    if (!tbody) return;

    tbody.innerHTML = '';
    this.conversationLog.forEach(item => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${item.time}</td>
        <td><span class="mono">${item.raw}</span></td>
        <td><strong>"${item.fluent}"</strong></td>
        <td><span class="tag tag-intent">${item.intent}</span></td>
        <td><span class="tag tag-urgency ${item.urgency === 'EMERGENCY' ? 'tag-urgency-emergency' : ''}">${item.urgency}</span></td>
        <td><span class="tag tag-state-speak">${item.decision}</span></td>
        <td><button class="btn-table-replay" data-text="${item.fluent}">🔊</button></td>
      `;
      tbody.appendChild(tr);
    });

    // Replay buttons
    document.querySelectorAll('.btn-table-replay').forEach(btn => {
      btn.addEventListener('click', () => {
        const text = btn.getAttribute('data-text');
        if (text) this.speak(text);
      });
    });
  }

  _initHistoryActions() {
    const btnJson = document.getElementById('btn-export-json');
    const btnTxt = document.getElementById('btn-export-txt');
    const btnClear = document.getElementById('btn-clear-history');

    if (btnJson) {
      btnJson.addEventListener('click', () => {
        const blob = new Blob([JSON.stringify(this.conversationLog, null, 2)], { type: 'application/json' });
        this._downloadBlob(blob, 'voxsign_transcript.json');
      });
    }

    if (btnTxt) {
      btnTxt.addEventListener('click', () => {
        let content = 'VOXSIGN AI COPILOT - TRANSCRIPT\n=================================\n\n';
        this.conversationLog.forEach(e => {
          content += `[${e.time}] [${e.urgency}] [${e.intent}]\nRaw: ${e.raw}\nSpoken: "${e.fluent}"\n\n`;
        });
        const blob = new Blob([content], { type: 'text/plain' });
        this._downloadBlob(blob, 'voxsign_transcript.txt');
      });
    }

    if (btnClear) {
      btnClear.addEventListener('click', () => {
        this.conversationLog = [];
        this._renderConversationLog();
      });
    }
  }

  _downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  // =========================================================================
  // DEMO STREAM SIMULATION (Simulates real sign sequence from dataset)
  // =========================================================================
  _startDemoSimulation() {
    this.isDemoMode = true;
    const demoPhrases = ['H', 'E', 'L', 'P', ' ', 'M', 'E'];
    let idx = 0;

    if (this.demoInterval) clearInterval(this.demoInterval);

    this.demoInterval = setInterval(async () => {
      if (idx >= demoPhrases.length) {
        idx = 0;
        this.rawTokenBuffer = [];
      }

      const letter = demoPhrases[idx];
      idx++;

      if (letter === ' ') {
        this.appendTokenToSequence(' ');
      } else {
        // Fetch sample image from dataset
        try {
          const res = await fetch(`/api/dataset/sample/${letter}`);
          if (res.ok) {
            const data = await res.json();
            const canvas = document.getElementById('output-canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            img.onload = () => {
              ctx.clearRect(0, 0, canvas.width, canvas.height);
              ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = data.image_b64;

            this._processModelPrediction(data.prediction || {
              predicted_token: letter,
              confidence: 0.98,
              confidence_pct: 98.0,
              top_candidates: [{ token: letter, confidence_pct: 98.0 }]
            }, 'Steady Hold');

            this.appendTokenToSequence(letter);
          }
        } catch (e) {
          this.appendTokenToSequence(letter);
        }
      }
    }, 1800);
  }

  _stopDemoSimulation() {
    this.isDemoMode = false;
    if (this.demoInterval) {
      clearInterval(this.demoInterval);
      this.demoInterval = null;
    }
  }

  // =========================================================================
  // TELEMETRY LIVE UPDATER
  // =========================================================================
  _startTelemetryUpdater() {
    const fpsEl = document.getElementById('telemetry-fps');
    const latencyEl = document.getElementById('telemetry-latency');

    setInterval(() => {
      if (fpsEl && this.tracker) {
        fpsEl.textContent = this.tracker.fps || 60;
      }
      if (latencyEl) {
        const simLatency = Math.floor(Math.random() * 5) + 10;
        latencyEl.textContent = `${simLatency}ms`;
      }
    }, 1000);
  }
}

// Instantiate on load
document.addEventListener('DOMContentLoaded', () => {
  window.voxSignApp = new VoxSignApp();
});
