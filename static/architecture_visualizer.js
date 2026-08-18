/**
 * 3D 8-Layer Multimodal System Architecture Visualizer
 * Interactive deep-dive explanations and signal flow animations.
 */

const LAYER_DETAILS = {
  1: {
    title: "Layer 1: Input Layer",
    tag: "LIVE SENSORS & DATASET",
    content: `
      <strong>Function:</strong> Captures high-frame-rate visual streams and dataset samples.<br/>
      <strong>Submodules:</strong>
      <ul>
        <li><code>Live Camera Stream:</code> 60fps WebRTC / MediaDevices video input with resolution auto-scaling.</li>
        <li><code>Dataset Hub:</code> Direct access to 10,676 real-world ASL training samples (Classes A-Z).</li>
        <li><code>Virtual Sign Simulator:</code> Real-time ASL keyboard emulator for sequence simulation and testing.</li>
      </ul>
      <strong>Latency Profile:</strong> &lt; 2ms frame capture buffer.
    `
  },
  2: {
    title: "Layer 2: Multimodal Perception",
    tag: "21 3D LANDMARKS & VISION",
    content: `
      <strong>Function:</strong> Extracts rich anatomical landmark coordinates, spatial geometry, face affect, and kinematics.<br/>
      <strong>Submodules:</strong>
      <ul>
        <li><code>21 3D Hand Landmarks:</code> Joint-by-joint coordinates [x, y, z] tracking thumb, index, middle, ring, and pinky.</li>
        <li><code>Face & Affect Analysis:</code> Head orientation and facial emotion state (calm, attentive, distressed).</li>
        <li><code>Body Pose & Alignment:</code> Shoulder and torso position anchoring for frame-of-reference invariance.</li>
        <li><code>Kinematic Motion Velocity:</code> Real-time trajectory and optical velocity estimation.</li>
      </ul>
      <strong>Latency Profile:</strong> ~ 8ms GPU-accelerated inference.
    `
  },
  3: {
    title: "Layer 3: Data Processing & Normalization",
    tag: "NORMALIZATION & FILTERING",
    content: `
      <strong>Function:</strong> Removes camera noise, normalizes hand scale/rotation, and filters jitter.<br/>
      <strong>Submodules:</strong>
      <ul>
        <li><code>Wrist-Centric Normalization:</code> Shifts all 21 landmark vectors relative to landmark 0 (wrist) for translation invariance.</li>
        <li><code>Palm-Scale Normalization:</code> Scales coordinate bounds to [0, 1] based on palm radius.</li>
        <li><code>Exponential Smoothing:</code> Filters high-frequency hand tremors while preserving crisp gesture starts.</li>
        <li><code>Hold vs Stroke Segmenter:</code> Classifies whether the hand is in an active transition or a steady communicative hold.</li>
      </ul>
      <strong>Latency Profile:</strong> &lt; 1ms computation.
    `
  },
  4: {
    title: "Layer 4: Temporal Understanding",
    tag: "CONTINUOUS SEQUENCE MODEL",
    content: `
      <strong>Function:</strong> Bridges single gesture frames into continuous sign language sequences.<br/>
      <strong>Submodules:</strong>
      <ul>
        <li><code>Rolling Token Buffer:</code> Stores timestamped sign hypotheses across a moving temporal window.</li>
        <li><code>N-Gram Sequence Modeling:</code> Computes transition probabilities between consecutive ASL letters.</li>
        <li><code>Debounce & Coalescence:</code> Prevents rapid duplicate token generation during prolonged holds.</li>
        <li><code>Word Boundary Segmentation:</code> Automatically detects space boundaries when hand pauses or rests.</li>
      </ul>
      <strong>Latency Profile:</strong> &lt; 2ms.
    `
  },
  5: {
    title: "Layer 5: Context & Conversational Memory",
    tag: "WORKING MEMORY & NLP",
    content: `
      <strong>Function:</strong> Maintains multi-turn dialogue history and reconstructs raw tokens into grammatically fluent sentences.<br/>
      <strong>Submodules:</strong>
      <ul>
        <li><code>Working Memory:</code> Stores the last 20 conversational tokens and speaker context.</li>
        <li><code>Contextual Smoothing:</code> Biases ambiguous letters toward contextually probable words.</li>
        <li><code>Natural Language Sentence Reconstructor:</code> Generates full, polite natural English from sign abbreviations.</li>
      </ul>
      <strong>Latency Profile:</strong> &lt; 3ms.
    `
  },
  6: {
    title: "Layer 6: Semantic Interpretation",
    tag: "INTENT, URGENCY & EMOTION",
    content: `
      <strong>Function:</strong> Understands communicative intent, urgency levels, and emotional tone.<br/>
      <strong>Submodules:</strong>
      <ul>
        <li><code>Intent Classifier:</code> Categorizes expressions into Emergency, Medical, Request, Question, Greeting, Gratitude, or Statement.</li>
        <li><code>4-Tier Urgency Scoring:</code> Tier 1 (Normal) &rarr; Tier 2 (Important) &rarr; Tier 3 (Urgent) &rarr; Tier 4 (Emergency).</li>
        <li><code>Emergency Keyword Scanner:</code> Identifies critical safety words like 'HELP', 'PAIN', 'DANGER', 'HOSPITAL', 'FIRE', 'CHOKING'.</li>
      </ul>
      <strong>Latency Profile:</strong> &lt; 2ms.
    `
  },
  7: {
    title: "Layer 7: Verification & AI Decision Engine",
    tag: "5-WAY STATE MACHINE & PERSONALIZATION",
    content: `
      <strong>Function:</strong> The central decision brain that arbitrates actions: SPEAK, WAIT, CLARIFY, CORRECT, or ALERT.<br/>
      <strong>Submodules:</strong>
      <ul>
        <li><code>5-Way Decision State Machine:</code> Dynamically switches between vocalization, hold buffering, disambiguation, self-correction, and emergency alarm.</li>
        <li><code>Self-Correcting AI:</code> Automatically updates past interpretations when subsequent signs provide clearer context.</li>
        <li><code>Personalized Signer Intelligence:</code> Adapts dynamically to fast signers, novices, and users with motor tremors.</li>
        <li><code>Confidence-Aware Thresholding:</code> Validates high-confidence signs before triggering speech.</li>
      </ul>
      <strong>Latency Profile:</strong> &lt; 1ms.
    `
  },
  8: {
    title: "Layer 8: Output Layer",
    tag: "MULTILINGUAL TTS & DISPATCH",
    content: `
      <strong>Function:</strong> Delivers immediate audio-visual communication to the surrounding environment.<br/>
      <strong>Submodules:</strong>
      <ul>
        <li><code>Multilingual Real-Time Voice:</code> Web Speech API synthesis with 12+ international languages (English, Spanish, French, German, Hindi, Japanese, Chinese, Arabic, Telugu, Tamil, etc.).</li>
        <li><code>Voice Acoustics Customization:</code> Fine-tuned speech rate, pitch, and accent modulation.</li>
        <li><code>Live Caption HUD:</code> Real-time on-screen subtitle banner with color-coded urgency indicators.</li>
        <li><code>Emergency Dispatch Broadcast:</code> High-priority visual strobe banner and audible siren alarm.</li>
      </ul>
      <strong>Latency Profile:</strong> Immediate audio buffer playback.
    `
  }
};

function initArchitectureVisualizer() {
  const layerCards = document.querySelectorAll('.layer-card');
  const titleEl = document.getElementById('detail-layer-title');
  const tagEl = document.getElementById('detail-layer-tag');
  const bodyEl = document.getElementById('detail-layer-body');

  layerCards.forEach(card => {
    card.addEventListener('click', () => {
      layerCards.forEach(c => c.classList.remove('active-layer'));
      card.classList.add('active-layer');

      const layerNum = parseInt(card.getAttribute('data-layer'), 10);
      const details = LAYER_DETAILS[layerNum];
      if (details && titleEl && tagEl && bodyEl) {
        titleEl.textContent = details.title;
        tagEl.textContent = details.tag;
        bodyEl.innerHTML = details.content;
      }
    });
  });
}

document.addEventListener('DOMContentLoaded', initArchitectureVisualizer);
