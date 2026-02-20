/* ==========================================================================
   K8s Failure Intelligence Copilot — app.js
   Cinematic single-file architecture:
     Splash, Background3D, CursorGlow, ChatUI, Effects, App controller
   ========================================================================= */

/* ── Splash — Cinematic boot sequence ─────────────────────────────────── */
class Splash {
  constructor() {
    this.el = document.getElementById('splash');
    this.progressBar = document.getElementById('splash-progress-bar');
    this.logLines = document.querySelectorAll('.splash-log-line');
    this.progress = 0;
  }

  async run() {
    if (!this.el) return;

    const totalDuration = 2800;
    const lineCount = this.logLines.length;
    const lineInterval = totalDuration / (lineCount + 1);

    for (let i = 0; i < lineCount; i++) {
      await this._wait(i === 0 ? 500 : lineInterval);
      this.logLines[i].classList.add('visible');
      this.progress = ((i + 1) / lineCount) * 100;
      if (this.progressBar) {
        this.progressBar.style.width = this.progress + '%';
      }
    }

    await this._wait(500);
    return this.exit();
  }

  exit() {
    return new Promise((resolve) => {
      if (!this.el) {
        resolve();
        return;
      }
      this.el.classList.add('exit');
      setTimeout(() => {
        this.el.style.display = 'none';
        resolve();
      }, 800);
    });
  }

  _wait(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}


/* ── Background3D — Three.js cinematic particle scene ─────────────────── */
class Background3D {
  constructor(canvas) {
    this.canvas = canvas;
    this.mouse = { x: 0, y: 0 };
    this.clock = new THREE.Clock();
    this.paused = false;
    this._init();
    this._bindEvents();
    this._animate();
  }

  _init() {
    const w = window.innerWidth;
    const h = window.innerHeight;

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: false,
      alpha: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x030711, 0.008);

    this.camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 300);
    this.camera.position.set(0, 0, 40);

    this.scene.add(new THREE.AmbientLight(0x112244, 0.5));
    this.light1 = new THREE.PointLight(0xff6b35, 0.8, 120);
    this.light1.position.set(20, 15, 15);
    this.scene.add(this.light1);
    this.light2 = new THREE.PointLight(0x00e5ff, 0.6, 120);
    this.light2.position.set(-20, -15, 10);
    this.scene.add(this.light2);
    this.light3 = new THREE.PointLight(0xff00aa, 0.3, 80);
    this.light3.position.set(0, 0, -20);
    this.scene.add(this.light3);

    this._createParticles();
    this._createFloatingShapes();
    this._createNetworkLines();
    this._createDataStreams();
  }

  _createParticles() {
    const count = 1200;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const sizes = new Float32Array(count);

    const accentColor = new THREE.Color(0xff6b35);
    const cyanColor = new THREE.Color(0x00e5ff);
    const whiteColor = new THREE.Color(0xffffff);

    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 100;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 80;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 60;
      sizes[i] = Math.random() * 2.5 + 0.5;

      const r = Math.random();
      let c;
      if (r < 0.3) c = accentColor;
      else if (r < 0.5) c = cyanColor;
      else c = whiteColor;
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    const texCanvas = document.createElement('canvas');
    texCanvas.width = 64;
    texCanvas.height = 64;
    const ctx = texCanvas.getContext('2d');
    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.2, 'rgba(255,255,255,0.6)');
    grad.addColorStop(0.5, 'rgba(255,255,255,0.1)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);
    const texture = new THREE.CanvasTexture(texCanvas);

    const mat = new THREE.PointsMaterial({
      size: 1.8,
      map: texture,
      vertexColors: true,
      transparent: true,
      opacity: 0.7,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    });
    this.particles = new THREE.Points(geo, mat);
    this.scene.add(this.particles);
  }

  _createFloatingShapes() {
    this.shapes = [];
    const makeShape = (geo, color, pos, scale) => {
      const mat = new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.25,
        wireframe: true,
        transparent: true,
        opacity: 0.3,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(...pos);
      mesh.scale.setScalar(scale);
      mesh.userData = {
        rotSpeed: { x: Math.random() * 0.004 + 0.001, y: Math.random() * 0.006 + 0.001 },
        floatOffset: Math.random() * Math.PI * 2,
        floatSpeed: Math.random() * 0.3 + 0.3,
        floatAmp: Math.random() * 0.005 + 0.002,
      };
      this.scene.add(mesh);
      this.shapes.push(mesh);
    };
    makeShape(new THREE.IcosahedronGeometry(1, 1), 0xff6b35, [-20, 10, -12], 3);
    makeShape(new THREE.OctahedronGeometry(1, 0), 0x00e5ff, [22, -8, -18], 2.5);
    makeShape(new THREE.TorusGeometry(1, 0.3, 8, 24), 0xff6b35, [14, 14, -22], 2.5);
    makeShape(new THREE.TetrahedronGeometry(1, 0), 0x00e5ff, [-16, -12, -10], 2);
    makeShape(new THREE.DodecahedronGeometry(1, 0), 0xff00aa, [0, -16, -25], 1.8);
    makeShape(new THREE.TorusKnotGeometry(0.8, 0.3, 48, 8), 0x00e5ff, [-8, 18, -30], 1.5);
  }

  _createNetworkLines() {
    const nodeCount = 60;
    const nodes = [];
    for (let i = 0; i < nodeCount; i++) {
      nodes.push(new THREE.Vector3(
        (Math.random() - 0.5) * 80,
        (Math.random() - 0.5) * 60,
        (Math.random() - 0.5) * 40 - 10,
      ));
    }
    const positions = [];
    for (let i = 0; i < nodeCount; i++) {
      for (let j = i + 1; j < nodeCount; j++) {
        if (nodes[i].distanceTo(nodes[j]) < 20) {
          positions.push(nodes[i].x, nodes[i].y, nodes[i].z);
          positions.push(nodes[j].x, nodes[j].y, nodes[j].z);
        }
      }
    }
    if (positions.length) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      const mat = new THREE.LineBasicMaterial({
        color: 0x4488cc,
        transparent: true,
        opacity: 0.08,
        blending: THREE.AdditiveBlending,
      });
      this.networkLines = new THREE.LineSegments(geo, mat);
      this.scene.add(this.networkLines);
    }
  }

  _createDataStreams() {
    this.dataStreams = [];
    const streamCount = 8;
    for (let s = 0; s < streamCount; s++) {
      const count = 20;
      const positions = new Float32Array(count * 3);
      const baseX = (Math.random() - 0.5) * 60;
      const baseZ = (Math.random() - 0.5) * 30 - 10;
      for (let i = 0; i < count; i++) {
        positions[i * 3] = baseX + (Math.random() - 0.5) * 2;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 60;
        positions[i * 3 + 2] = baseZ;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const mat = new THREE.PointsMaterial({
        size: 0.8,
        color: s % 2 === 0 ? 0x00e5ff : 0xff6b35,
        transparent: true,
        opacity: 0.3,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const points = new THREE.Points(geo, mat);
      points.userData = { speed: Math.random() * 0.15 + 0.05 };
      this.scene.add(points);
      this.dataStreams.push(points);
    }
  }

  _bindEvents() {
    let resizeTimer;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => this._onResize(), 150);
    });

    document.addEventListener('mousemove', (e) => {
      this.mouse.x = (e.clientX / window.innerWidth - 0.5) * 2;
      this.mouse.y = (e.clientY / window.innerHeight - 0.5) * 2;
    });

    document.addEventListener('visibilitychange', () => {
      this.paused = document.hidden;
      if (!this.paused) {
        this.clock.getDelta();
        this._animate();
      }
    });
  }

  _onResize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  _animate() {
    if (this.paused) return;
    requestAnimationFrame(() => this._animate());

    const t = this.clock.getElapsedTime();

    const targetX = Math.sin(t * 0.1) * 3 + this.mouse.x * 2;
    const targetY = Math.cos(t * 0.08) * 2 - this.mouse.y * 1.5;
    this.camera.position.x += (targetX - this.camera.position.x) * 0.02;
    this.camera.position.y += (targetY - this.camera.position.y) * 0.02;
    this.camera.lookAt(0, 0, 0);

    if (this.particles) {
      this.particles.rotation.y = t * 0.02;
      this.particles.rotation.x = Math.sin(t * 0.015) * 0.15;
    }

    this.shapes.forEach((s) => {
      s.rotation.x += s.userData.rotSpeed.x;
      s.rotation.y += s.userData.rotSpeed.y;
      s.position.y += Math.sin(t * s.userData.floatSpeed + s.userData.floatOffset) * s.userData.floatAmp;
      s.position.x += Math.cos(t * s.userData.floatSpeed * 0.7 + s.userData.floatOffset) * s.userData.floatAmp * 0.5;
    });

    this.light1.position.x = Math.sin(t * 0.25) * 25;
    this.light1.position.z = Math.cos(t * 0.25) * 18;
    this.light2.position.x = Math.cos(t * 0.2) * 25;
    this.light2.position.z = Math.sin(t * 0.2) * 18;
    this.light3.position.x = Math.sin(t * 0.15 + 2) * 15;
    this.light3.position.y = Math.cos(t * 0.15 + 2) * 15;

    if (this.networkLines) {
      this.networkLines.material.opacity = 0.05 + Math.sin(t * 0.5) * 0.03;
      this.networkLines.rotation.y = t * 0.003;
    }

    this.dataStreams.forEach((stream) => {
      const pos = stream.geometry.attributes.position;
      for (let i = 0; i < pos.count; i++) {
        pos.array[i * 3 + 1] -= stream.userData.speed;
        if (pos.array[i * 3 + 1] < -30) {
          pos.array[i * 3 + 1] = 30;
        }
      }
      pos.needsUpdate = true;
    });

    this.renderer.render(this.scene, this.camera);
  }

  setTheme(theme) {
    const colors = { dark: 0x030711, neon: 0x000008, light: 0xf0f2f7 };
    const fogColor = colors[theme] || colors.dark;
    this.scene.fog.color.setHex(fogColor);
    this.scene.background = null;
    if (this.particles) {
      this.particles.material.opacity = theme === 'light' ? 0.25 : 0.7;
    }
    if (this.networkLines) {
      this.networkLines.material.opacity = theme === 'light' ? 0.03 : 0.08;
    }
    this.dataStreams.forEach((s) => {
      s.material.opacity = theme === 'light' ? 0.1 : 0.3;
    });
  }
}


/* ── CursorGlow — Mouse-follow ambient glow ───────────────────────────── */
class CursorGlow {
  constructor() {
    this.el = document.getElementById('cursor-glow');
    if (!this.el) return;
    this._bind();
    setTimeout(() => this.el.classList.add('active'), 3000);
  }

  _bind() {
    document.addEventListener('mousemove', (e) => {
      if (!this.el) return;
      this.el.style.left = e.clientX + 'px';
      this.el.style.top = e.clientY + 'px';
    });
  }
}


/* ── ChatUI — Message handling, streaming, markdown rendering ─────────── */
class ChatUI {
  constructor() {
    this.messagesEl = document.getElementById('chat-messages');
    this.questionEl = document.getElementById('question');
    this.sendBtn = document.getElementById('send-btn');
    this.clearBtn = document.getElementById('clear-btn');
    this.sessionEl = document.getElementById('session-id');
    this.modelEl = document.getElementById('model-override');
    this.welcomeEl = document.getElementById('welcome-screen');

    this.isStreaming = false;
    this.autoScroll = true;
    this.messages = [];

    this._bindEvents();
    this._configureMarked();
  }

  _configureMarked() {
    if (typeof marked !== 'undefined') {
      marked.setOptions({
        breaks: true,
        gfm: true,
        highlight: (code, lang) => {
          if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
            try { return hljs.highlight(code, { language: lang }).value; }
            catch (_) { /* ignore */ }
          }
          if (typeof hljs !== 'undefined') {
            try { return hljs.highlightAuto(code).value; }
            catch (_) { /* ignore */ }
          }
          return code;
        },
      });
    }
  }

  _bindEvents() {
    this.sendBtn.addEventListener('click', () => this._handleSend());

    this.questionEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this._handleSend();
      }
    });

    this.questionEl.addEventListener('input', () => {
      this.questionEl.style.height = 'auto';
      this.questionEl.style.height = Math.min(this.questionEl.scrollHeight, 160) + 'px';
    });

    this.messagesEl.addEventListener('scroll', () => {
      const el = this.messagesEl;
      this.autoScroll = (el.scrollHeight - el.scrollTop - el.clientHeight) < 60;
    });

    this.clearBtn.addEventListener('click', () => this._clearMemory());

    document.querySelectorAll('.welcome-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        const query = chip.dataset.query;
        if (query) {
          this.questionEl.value = query;
          this._handleSend();
        }
      });
    });
  }

  async _handleSend() {
    const text = this.questionEl.value.trim();
    if (!text || this.isStreaming) return;

    if (this.welcomeEl) {
      if (typeof gsap !== 'undefined') {
        gsap.to(this.welcomeEl, {
          opacity: 0, y: -20, scale: 0.95,
          duration: 0.4, ease: 'power3.in',
          onComplete: () => {
            this.welcomeEl.style.display = 'none';
            this.welcomeEl = null;
          },
        });
      } else {
        this.welcomeEl.style.display = 'none';
        this.welcomeEl = null;
      }
    }

    this._addMessage('user', text);
    this.questionEl.value = '';
    this.questionEl.style.height = 'auto';

    await this._streamDiagnosis(text);
  }

  _addMessage(role, content, meta = null) {
    const wrapper = document.createElement('div');
    wrapper.classList.add('message', `message--${role}`);

    const bubble = document.createElement('div');
    bubble.classList.add('msg-bubble');

    if (role === 'user') {
      bubble.textContent = content;
    } else {
      bubble.innerHTML = this._renderMarkdown(content);
      this._addCodeCopyButtons(bubble);
    }

    wrapper.appendChild(bubble);

    const time = document.createElement('div');
    time.classList.add('msg-time');
    time.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    if (role === 'ai' && meta) {
      const metaWrapper = this._createMetaBar(meta, content);
      const metaBar = metaWrapper.querySelector('.msg-meta') || metaWrapper;
      if (metaBar.classList.contains('msg-meta')) {
        metaBar.prepend(time);
      } else {
        const mb = metaWrapper.querySelector('.msg-meta');
        if (mb) mb.prepend(time);
      }
      wrapper.appendChild(metaWrapper);
    } else {
      const mb = document.createElement('div');
      mb.classList.add('msg-meta');
      mb.appendChild(time);
      wrapper.appendChild(mb);
    }

    this.messagesEl.appendChild(wrapper);
    this._scrollToBottom();

    if (typeof gsap !== 'undefined') {
      gsap.fromTo(wrapper,
        { opacity: 0, y: 24, scale: 0.94, filter: 'blur(4px)' },
        {
          opacity: 1, y: 0, scale: 1, filter: 'blur(0px)',
          duration: 0.6,
          ease: 'back.out(1.2)',
        }
      );
    }

    return wrapper;
  }

  _createMetaBar(meta, fullText) {
    const bar = document.createElement('div');
    bar.classList.add('msg-meta');

    if (meta.confidence != null) {
      const conf = document.createElement('span');
      conf.classList.add('msg-badge', 'msg-badge--confidence');
      conf.textContent = `${Math.round(meta.confidence * 100)}%`;
      conf.title = 'Confidence';
      bar.appendChild(conf);
    }

    if (meta.model_used) {
      const model = document.createElement('span');
      model.classList.add('msg-badge', 'msg-badge--model');
      model.textContent = meta.model_used;
      model.title = 'Model used';
      bar.appendChild(model);
    }

    if (meta.response_type) {
      const type = document.createElement('span');
      type.classList.add('msg-badge');
      type.textContent = meta.response_type;
      bar.appendChild(type);
    }

    const copyBtn = document.createElement('button');
    copyBtn.classList.add('msg-copy-btn');
    copyBtn.textContent = 'Copy';
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(fullText).then(() => {
        copyBtn.textContent = 'Copied!';
        App.toast('Copied to clipboard', 'success');
        setTimeout(() => { copyBtn.textContent = 'Copy'; }, 2000);
      });
    });
    bar.appendChild(copyBtn);

    const wrapper = document.createElement('div');
    wrapper.appendChild(bar);

    if (meta.sources && meta.sources.length > 0) {
      const details = document.createElement('details');
      details.classList.add('msg-sources');
      const summary = document.createElement('summary');
      summary.textContent = `Sources (${meta.sources.length})`;
      details.appendChild(summary);
      const ul = document.createElement('ul');
      meta.sources.forEach((s) => {
        const li = document.createElement('li');
        li.textContent = s;
        ul.appendChild(li);
      });
      details.appendChild(ul);
      wrapper.appendChild(details);
    }

    return wrapper;
  }

  _addCodeCopyButtons(bubble) {
    bubble.querySelectorAll('pre').forEach((pre) => {
      const btn = document.createElement('button');
      btn.classList.add('code-copy-btn');
      btn.textContent = 'Copy';
      btn.addEventListener('click', () => {
        const code = pre.querySelector('code');
        navigator.clipboard.writeText(code ? code.textContent : pre.textContent).then(() => {
          btn.textContent = 'Copied!';
          setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
        });
      });
      pre.style.position = 'relative';
      pre.appendChild(btn);
    });
  }

  _showTyping() {
    const indicator = document.createElement('div');
    indicator.classList.add('typing-indicator');
    indicator.id = 'typing-indicator';
    indicator.setAttribute('aria-label', 'AI is thinking');
    indicator.textContent = 'Thinking…';
    this.messagesEl.appendChild(indicator);
    this._scrollToBottom();
    return indicator;
  }

  _removeTyping() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
  }

  async _streamDiagnosis(question) {
    this.isStreaming = true;
    this.sendBtn.disabled = true;
    App.setStatus('Thinking…');

    this._showTyping();

    const sessionId = this.sessionEl?.value?.trim() || 'web';
    const forceModel = this.modelEl?.value || null;
    const payload = { question, session_id: sessionId, force_model: forceModel };

    let rawText = '';
    let isJsonStream = false;
    let hasPlaceholder = false;
    let diagnosis = null;
    let aiMessage = null;
    let bubble = null;

    try {
      const res = await fetch('/diagnose/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        return await this._fallbackDiagnose(question, payload);
      }

      this._removeTyping();
      aiMessage = document.createElement('div');
      aiMessage.classList.add('message', 'message--ai');
      bubble = document.createElement('div');
      bubble.classList.add('msg-bubble');
      aiMessage.appendChild(bubble);
      this.messagesEl.appendChild(aiMessage);

      if (typeof gsap !== 'undefined') {
        gsap.fromTo(aiMessage,
          { opacity: 0, y: 20, scale: 0.95, filter: 'blur(3px)' },
          { opacity: 1, y: 0, scale: 1, filter: 'blur(0px)', duration: 0.5, ease: 'back.out(1.1)' }
        );
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue;

          try {
            const event = JSON.parse(jsonStr);

            if (event.error) {
              rawText += `\n\n**Error:** ${event.error}`;
              bubble.innerHTML = this._renderMarkdown(rawText);
              break;
            }

            if (event.token) {
              rawText += event.token;
              const sample = rawText.trimStart();
              if (!isJsonStream) {
                if (sample.startsWith('{') || sample.startsWith('```json') || sample.startsWith('```')) {
                  isJsonStream = true;
                }
              }
              if (isJsonStream) {
                if (!hasPlaceholder) {
                  bubble.innerHTML = this._renderMarkdown('_Generating response..._');
                  hasPlaceholder = true;
                }
              } else {
                bubble.innerHTML = this._renderMarkdown(rawText);
                this._scrollToBottom();
              }
            }

            if (event.done && event.diagnosis) {
              diagnosis = event.diagnosis;
            }
          } catch (_) {
            /* Ignore malformed SSE lines */
          }
        }
      }

      if (diagnosis) {
        const displayText = this._formatDiagnosis(diagnosis, rawText);
        await this._streamMarkdownText(bubble, displayText);
        this._addCodeCopyButtons(bubble);

        const metaWrapper = this._createMetaBar(diagnosis, displayText);
        const timeEl = document.createElement('div');
        timeEl.classList.add('msg-time');
        timeEl.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const metaBar = metaWrapper.querySelector('.msg-meta');
        if (metaBar) metaBar.prepend(timeEl);

        aiMessage.appendChild(metaWrapper);
      } else {
        bubble.innerHTML = this._renderMarkdown(rawText);
        this._addCodeCopyButtons(bubble);
      }

      this._scrollToBottom();
      this._addSuggestions(rawText);
    } catch (err) {
      console.error('Stream error:', err);
      if (!aiMessage) {
        return await this._fallbackDiagnose(question, payload);
      }
      App.toast('Stream error — see console', 'error');
    } finally {
      this.isStreaming = false;
      this.sendBtn.disabled = false;
      App.setStatus('Ready');
      this._removeTyping();
    }
  }

  async _fallbackDiagnose(question, payload) {
    try {
      const res = await fetch('/diagnose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      this._removeTyping();

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Request failed' }));
        this._addMessage('ai', `**Error:** ${err.detail || 'Unknown error'}`);
        App.toast('Diagnosis failed', 'error');
        return;
      }

      const data = await res.json();
      const dx = data.diagnosis;
      const displayText = this._formatDiagnosis(dx);
      this._addMessage('ai', displayText, dx);
      this._addSuggestions(displayText);
      App.toast('Diagnosis complete', 'success');
    } catch (err) {
      this._removeTyping();
      this._addMessage('ai', `**Error:** ${err.message}`);
      App.toast('Request failed', 'error');
    } finally {
      this.isStreaming = false;
      this.sendBtn.disabled = false;
      App.setStatus('Ready');
    }
  }

  _formatDiagnosis(dx, rawText) {
    if (rawText && !rawText.trim().startsWith('{')) {
      return rawText;
    }

    let text = '';
    if (dx.root_cause && dx.root_cause !== 'Unknown') {
      text += `### Root Cause\n${dx.root_cause}\n\n`;
    }
    if (dx.explanation) {
      text += `### Explanation\n${dx.explanation}\n\n`;
    }
    if (dx.recommended_fix && dx.recommended_fix !== 'N/A') {
      text += `### Recommended Fix\n${dx.recommended_fix}\n`;
    }
    return text || dx.root_cause || 'No diagnosis available.';
  }

  _renderMarkdown(text) {
    if (!text) return '';
    try {
      const html = typeof marked !== 'undefined'
        ? marked.parse(text)
        : text.replace(/\n/g, '<br>');
      return typeof DOMPurify !== 'undefined'
        ? DOMPurify.sanitize(html)
        : html;
    } catch (_) {
      return text.replace(/\n/g, '<br>');
    }
  }

  async _streamMarkdownText(targetEl, fullText) {
    if (!targetEl) return;
    const parts = fullText.split(/(\s+)/);
    const chunkSize = 5;
    let index = 0;
    let current = '';

    while (index < parts.length) {
      const next = parts.slice(index, index + chunkSize).join('');
      current += next;
      index += chunkSize;
      targetEl.innerHTML = this._renderMarkdown(current);
      this._scrollToBottom();
      await new Promise((resolve) => setTimeout(resolve, 60));
    }
  }

  _addSuggestions(responseText) {
    const old = document.querySelector('.suggestions');
    if (old) old.remove();

    const lower = (responseText || '').toLowerCase();
    const suggestions = [];

    if (lower.includes('oomkill') || lower.includes('memory'))
      suggestions.push('How do I increase Kubernetes pod memory limits?');
    if (lower.includes('crashloop') || lower.includes('restart'))
      suggestions.push('Show me the Kubernetes pod restart logs');
    if (lower.includes('image') || lower.includes('pull'))
      suggestions.push('How do I fix ImagePullBackOff in Kubernetes?');
    if (lower.includes('schedul'))
      suggestions.push('Why is my Kubernetes pod not being scheduled?');

    suggestions.push('What other Kubernetes issues could cause this error?');

    const pills = suggestions.slice(0, 3);

    const container = document.createElement('div');
    container.classList.add('suggestions');
    container.setAttribute('role', 'group');
    container.setAttribute('aria-label', 'Suggested follow-up questions');

    pills.forEach((text, i) => {
      const pill = document.createElement('button');
      pill.classList.add('suggestion-pill');
      pill.textContent = text;
      pill.addEventListener('click', () => {
        this.questionEl.value = text;
        this._handleSend();
      });
      container.appendChild(pill);

      if (typeof gsap !== 'undefined') {
        gsap.fromTo(pill,
          { opacity: 0, y: 8, scale: 0.9 },
          { opacity: 1, y: 0, scale: 1, duration: 0.4, delay: 0.1 + i * 0.08, ease: 'back.out(1.5)' }
        );
      }
    });

    const inputArea = document.querySelector('.input-area');
    if (inputArea && inputArea.parentNode) {
      inputArea.parentNode.insertBefore(container, inputArea);
    }
  }

  _scrollToBottom() {
    if (!this.autoScroll) return;
    requestAnimationFrame(() => {
      this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    });
  }

  async _clearMemory() {
    const sessionId = this.sessionEl?.value?.trim() || 'web';
    try {
      await fetch('/memory/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });
      App.toast('Memory cleared', 'info');
    } catch (_) {
      App.toast('Failed to clear memory', 'error');
    }
  }
}


/* ── Effects — GSAP cinematic animations ──────────────────────────────── */
class Effects {
  static pageEntrance() {
    if (typeof gsap === 'undefined') return;

    const tl = gsap.timeline({ defaults: { ease: 'power4.out' } });

    tl.fromTo('.header',
      { opacity: 0, y: -40, scale: 0.96 },
      { opacity: 1, y: 0, scale: 1, duration: 0.8 }
    );

    tl.fromTo('.chat-container',
      { opacity: 0, y: 40, scale: 0.92, filter: 'blur(8px)' },
      { opacity: 1, y: 0, scale: 1, filter: 'blur(0px)', duration: 0.9 },
      '-=0.5'
    );

    tl.fromTo('.welcome-orb',
      { opacity: 0, scale: 0, rotation: -180 },
      { opacity: 1, scale: 1, rotation: 0, duration: 0.8, ease: 'back.out(2)' },
      '-=0.4'
    );

    tl.fromTo('.welcome-title',
      { opacity: 0, y: 20, filter: 'blur(6px)' },
      { opacity: 1, y: 0, filter: 'blur(0px)', duration: 0.5 },
      '-=0.3'
    );

    tl.fromTo('.welcome-desc',
      { opacity: 0, y: 15 },
      { opacity: 1, y: 0, duration: 0.4 },
      '-=0.2'
    );

    tl.fromTo('.welcome-chip',
      { opacity: 0, y: 12, scale: 0.85 },
      { opacity: 1, y: 0, scale: 1, duration: 0.4, stagger: 0.08, ease: 'back.out(1.5)' },
      '-=0.2'
    );

    tl.fromTo('.input-area',
      { opacity: 0, y: 20 },
      { opacity: 1, y: 0, duration: 0.5 },
      '-=0.3'
    );
  }
}


/* ── App Controller ───────────────────────────────────────────────────── */
const App = {
  bg: null,
  chat: null,
  splash: null,
  cursorGlow: null,

  async init() {
    const canvas = document.getElementById('bg-canvas');
    if (canvas && typeof THREE !== 'undefined') {
      try {
        this.bg = new Background3D(canvas);
      } catch (err) {
        console.warn('Three.js init failed:', err);
      }
    }

    const app = document.getElementById('app');
    const revealApp = () => {
      if (!app) return;
      app.classList.remove('app-hidden');
      app.classList.add('visible');
    };

    try {
      this.splash = new Splash();
      await this.splash.run();
    } catch (err) {
      console.warn('Splash failed:', err);
    } finally {
      revealApp();
    }

    this.cursorGlow = new CursorGlow();
    this.chat = new ChatUI();

    const savedTheme = localStorage.getItem('k8s-theme') || 'dark';
    this._applyTheme(savedTheme);

    document.getElementById('theme-btn')?.addEventListener('click', () => {
      const themes = ['dark', 'neon', 'light'];
      const current = document.documentElement.dataset.theme || 'dark';
      const next = themes[(themes.indexOf(current) + 1) % themes.length];
      this._applyTheme(next);
      localStorage.setItem('k8s-theme', next);
      this.toast(`Theme: ${next}`, 'info');
    });

    this._checkHealth();
    Effects.pageEntrance();
  },

  _applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    if (this.bg) this.bg.setTheme(theme);

    const btn = document.getElementById('theme-btn');
    if (btn) {
      const icons = { dark: '🌙', neon: '⚡', light: '☀️' };
      btn.textContent = icons[theme] || '🌙';
      btn.title = `Theme: ${theme}`;
    }
  },

  async _checkHealth() {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    try {
      const res = await fetch('/health');
      const data = await res.json();
      if (res.ok && data.status === 'ok') {
        dot?.classList.remove('error');
        if (text) text.textContent = 'Connected';
      } else {
        dot?.classList.add('error');
        if (text) text.textContent = 'Degraded';
      }
    } catch (_) {
      dot?.classList.add('error');
      if (text) text.textContent = 'Offline';
    }
  },

  setStatus(msg) {
    const el = document.getElementById('status-text');
    if (el) el.textContent = msg;
  },

  toast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.classList.add('toast', `toast--${type}`);
    toast.textContent = message;
    toast.setAttribute('role', 'status');
    container.appendChild(toast);

    setTimeout(() => {
      toast.classList.add('leaving');
      setTimeout(() => toast.remove(), 400);
    }, 3500);
  },
};


/* ── Boot ─────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => App.init());