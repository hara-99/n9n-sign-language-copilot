/**
 * Dataset Hub & ML Model Benchmark Lab
 * Interfaces with archive (1)/Data dataset, sample viewer, and custom image inference.
 */

class DatasetHub {
  constructor() {
    this.classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
    this.activeClass = 'A';
    this.datasetStats = null;
    this.init();
  }

  async init() {
    this._renderClassTabs();
    this._setupDropzone();
    await this.fetchDatasetStats();
    await this.loadSampleImage(this.activeClass);
  }

  _renderClassTabs() {
    const container = document.getElementById('dataset-class-bar');
    if (!container) return;
    container.innerHTML = '';

    this.classes.forEach(letter => {
      const btn = document.createElement('button');
      btn.className = `btn-class-tab ${letter === this.activeClass ? 'active' : ''}`;
      btn.textContent = letter;
      btn.addEventListener('click', () => {
        document.querySelectorAll('.btn-class-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.activeClass = letter;
        this.loadSampleImage(letter);
      });
      container.appendChild(btn);
    });

    const randomBtn = document.getElementById('btn-load-random-sample');
    if (randomBtn) {
      randomBtn.addEventListener('click', () => {
        const randomLetter = this.classes[Math.floor(Math.random() * this.classes.length)];
        this.activeClass = randomLetter;
        document.querySelectorAll('.btn-class-tab').forEach(b => {
          b.classList.toggle('active', b.textContent === randomLetter);
        });
        this.loadSampleImage(randomLetter);
      });
    }
  }

  async fetchDatasetStats() {
    try {
      const res = await fetch('/api/dataset/stats');
      if (res.ok) {
        this.datasetStats = await res.json();
        this._renderDistributionChart(this.datasetStats.class_distribution);
      }
    } catch (e) {
      console.warn('[DatasetHub] Failed to fetch dataset stats:', e);
    }
  }

  _renderDistributionChart(distribution) {
    const container = document.getElementById('dist-chart-bars');
    if (!container || !distribution) return;
    container.innerHTML = '';

    const maxCount = Math.max(...Object.values(distribution), 400);

    for (const [cls, count] of Object.entries(distribution)) {
      const col = document.createElement('div');
      col.className = 'dist-bar-col';
      col.title = `Class ${cls}: ${count} images`;

      const bar = document.createElement('div');
      bar.className = 'dist-bar';
      const pct = Math.round((count / maxCount) * 100);
      bar.style.height = `${pct}%`;

      const lbl = document.createElement('span');
      lbl.className = 'dist-lbl';
      lbl.textContent = cls;

      col.appendChild(bar);
      col.appendChild(lbl);
      container.appendChild(col);
    }
  }

  async loadSampleImage(className) {
    const imgEl = document.getElementById('sample-dataset-img');
    const metaEl = document.getElementById('sample-meta-info');
    const predClassEl = document.getElementById('sample-pred-class');
    const predConfEl = document.getElementById('sample-pred-conf');
    const barsContainer = document.getElementById('sample-candidate-bars');

    if (metaEl) metaEl.textContent = `Loading sample for '${className}'...`;

    try {
      const res = await fetch(`/api/dataset/sample/${className}`);
      if (!res.ok) throw new Error('Sample not found');
      const data = await res.json();

      if (imgEl && data.image_b64) {
        imgEl.src = data.image_b64;
      }
      if (metaEl) {
        metaEl.textContent = `Class: ${data.class_name} | File: ${data.filename} | Dataset: archive (1)/Data`;
      }

      if (data.prediction && predClassEl && predConfEl) {
        const pred = data.prediction;
        predClassEl.textContent = pred.predicted_token || '-';
        predConfEl.textContent = `(${pred.confidence_pct}% Confidence)`;

        if (barsContainer && pred.top_candidates) {
          barsContainer.innerHTML = '';
          pred.top_candidates.forEach(cand => {
            const row = document.createElement('div');
            row.className = 'prob-row';
            row.innerHTML = `
              <strong>${cand.token}</strong>
              <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width: ${cand.confidence_pct}%;"></div>
              </div>
              <span>${cand.confidence_pct}%</span>
            `;
            barsContainer.appendChild(row);
          });
        }
      }
    } catch (e) {
      if (metaEl) metaEl.textContent = `Error loading sample for class ${className}`;
      console.warn('[DatasetHub] Error:', e);
    }
  }

  _setupDropzone() {
    const dropzone = document.getElementById('upload-dropzone');
    const fileInput = document.getElementById('file-upload-input');

    if (!dropzone || !fileInput) return;

    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-over');
    });

    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        this._processUploadedFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        this._processUploadedFile(e.target.files[0]);
      }
    });
  }

  _processUploadedFile(file) {
    const reader = new FileReader();
    reader.onload = async (event) => {
      const b64 = event.target.result;
      const imgEl = document.getElementById('sample-dataset-img');
      const metaEl = document.getElementById('sample-meta-info');

      if (imgEl) imgEl.src = b64;
      if (metaEl) metaEl.textContent = `Uploaded Custom Image: ${file.name}`;

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_b64: b64 })
        });
        if (res.ok) {
          const pred = await res.json();
          const predClassEl = document.getElementById('sample-pred-class');
          const predConfEl = document.getElementById('sample-pred-conf');
          const barsContainer = document.getElementById('sample-candidate-bars');

          if (predClassEl) predClassEl.textContent = pred.predicted_token || '-';
          if (predConfEl) predConfEl.textContent = `(${pred.confidence_pct}% Confidence)`;

          if (barsContainer && pred.top_candidates) {
            barsContainer.innerHTML = '';
            pred.top_candidates.forEach(cand => {
              const row = document.createElement('div');
              row.className = 'prob-row';
              row.innerHTML = `
                <strong>${cand.token}</strong>
                <div class="prob-bar-track">
                  <div class="prob-bar-fill" style="width: ${cand.confidence_pct}%;"></div>
                </div>
                <span>${cand.confidence_pct}%</span>
              `;
              barsContainer.appendChild(row);
            });
          }
        }
      } catch (e) {
        console.warn('[DatasetHub] Custom inference failed:', e);
      }
    };
    reader.readAsDataURL(file);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.datasetHub = new DatasetHub();
});
