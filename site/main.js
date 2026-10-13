// n0face site: nav state, scroll reveals, and the hero face (MediaPipe's canonical, anonymous
// mesh, exported by scripts/make_site_assets.py). No dependencies, no network beyond face.json.
(() => {
  "use strict";

  const root = document.documentElement;
  root.classList.add("js");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- nav ---------------------------------------------------------------------------------
  const nav = document.querySelector("[data-nav]");
  const onScroll = () => nav.classList.toggle("is-scrolled", scrollY > 12);
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // ---- reveal on scroll (staggered per parent) ----------------------------------------------
  const groups = new Map();
  document.querySelectorAll(".reveal").forEach((el) => {
    const i = groups.get(el.parentElement) || 0;
    groups.set(el.parentElement, i + 1);
    el.style.setProperty("--d", `${Math.min(i, 6) * 0.08}s`);
  });
  if (reduced || !("IntersectionObserver" in window)) {
    document.querySelectorAll(".reveal").forEach((el) => el.classList.add("is-in"));
  } else {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add("is-in");
          io.unobserve(e.target);
        }
      }
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.12 });
    document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
  }

  // ---- GLITCH card: the app's real glitch frames, in short bursts ----------------------------
  const art = document.querySelector("[data-glitch-art]");
  if (art) {
    if (reduced) {
      art.dataset.f = "1";
    } else {
      const burst = () => {
        let k = 0;
        const steps = 3 + (Math.random() * 4 | 0);
        const tick = () => {
          art.dataset.f = k < steps ? String(1 + (Math.random() * 3 | 0)) : "0";
          if (k++ < steps) setTimeout(tick, 50 + Math.random() * 70);
        };
        tick();
        setTimeout(burst, 1400 + Math.random() * 2600);
      };
      setTimeout(burst, 1200);
    }
  }

  document.querySelectorAll(".pro__strip li").forEach((li, i) => li.style.setProperty("--i", i));

  // ---- app screenshot: un-tilts as it scrolls into view ---------------------------------------
  const shot = document.querySelector(".shot");
  if (shot && !reduced) {
    let queued = false;
    const tilt = () => {
      queued = false;
      const r = shot.getBoundingClientRect();
      const p = Math.min(Math.max((innerHeight - r.top) / (innerHeight * 0.8), 0), 1);
      shot.style.setProperty("--tilt", `${(1 - p) * 12}deg`);
      shot.style.setProperty("--shrink", `${0.94 + p * 0.06}`);
    };
    addEventListener("scroll", () => { if (!queued) { queued = true; requestAnimationFrame(tilt); } }, { passive: true });
    tilt();
  }

  // ---- background: grid, cells that light up, pulses running along the lines, dust ------------
  document.querySelectorAll("[data-bg]").forEach((bg) => backdrop(bg));

  function backdrop(cv) {
    const g = cv.getContext("2d");
    const host = cv.parentElement;
    const CELL = 64;
    let W = 0, H = 0, dpr = 1, ox = 0, cols = 0, rows = 0;
    let lines = null, glowLines = null, spot = null;
    let cells = [], pulses = [], dust = [];
    let mx = -1e4, my = -1e4, sx = -1e4, sy = -1e4;

    const gridCanvas = (style) => {
      const c = document.createElement("canvas");
      c.width = W;
      c.height = H;
      const x = c.getContext("2d");
      x.strokeStyle = style;
      x.lineWidth = 1;
      x.beginPath();
      for (let i = 0; i <= cols; i++) { const X = Math.round((ox + i * CELL) * dpr) + 0.5; x.moveTo(X, 0); x.lineTo(X, H); }
      for (let j = 0; j <= rows; j++) { const Y = Math.round(j * CELL * dpr) + 0.5; x.moveTo(0, Y); x.lineTo(W, Y); }
      x.stroke();
      return c;
    };

    const spawnCell = (t) => {
      // biased toward the middle, where the mask keeps things visible
      const c = Math.floor(cols / 2 + (Math.random() + Math.random() - 1) * cols * 0.45);
      const r = Math.floor(Math.random() * Math.random() * rows * 0.9);
      cells.push({ c, r, t0: t, life: 2.5 + Math.random() * 3.5, hot: Math.random() < 0.18 });
    };
    const spawnPulse = (t) => {
      const across = Math.random() < 0.5;
      const k = across ? Math.floor(Math.random() * rows * 0.8) : Math.floor(cols * 0.15 + Math.random() * cols * 0.7);
      const dir = Math.random() < 0.5 ? 1 : -1;
      pulses.push({ across, k, dir, t0: t, speed: 140 + Math.random() * 160, len: 70 + Math.random() * 90 });
    };

    const resize = () => {
      dpr = Math.min(devicePixelRatio || 1, 1.5);
      const w = host.clientWidth, h = host.clientHeight;
      W = cv.width = Math.round(w * dpr);
      H = cv.height = Math.round(h * dpr);
      ox = ((w / 2) % CELL) - CELL; // a line runs through the middle of the page
      cols = Math.ceil(w / CELL) + 2;
      rows = Math.ceil(h / CELL) + 1;
      lines = gridCanvas("rgba(255,255,255,0.05)");
      glowLines = gridCanvas("rgba(124,255,43,0.55)");
      spot = document.createElement("canvas");
      spot.width = W;
      spot.height = H;
      dust = Array.from({ length: Math.round((w * h) / 26000) }, () => ({
        x: Math.random() * w, y: Math.random() * h, r: 0.6 + Math.random() * 1.1,
        v: 4 + Math.random() * 10, p: Math.random() * 6.3,
      }));
      if (reduced) {
        cells = [];
        for (let i = 0; i < 10; i++) spawnCell(-Math.random() * 3);
        draw(1.5);
      }
    };

    host.addEventListener("pointermove", (e) => {
      const r = cv.getBoundingClientRect();
      mx = e.clientX - r.left;
      my = e.clientY - r.top;
    });
    host.addEventListener("pointerleave", () => { mx = my = -1e4; });

    let nextCell = 0, nextPulse = 0.6, last = 0;
    function draw(t) {
      const dt = Math.min(Math.max(t - last, 0), 0.1);
      last = t;
      g.setTransform(1, 0, 0, 1, 0, 0);
      g.clearRect(0, 0, W, H);
      g.drawImage(lines, 0, 0);
      g.setTransform(dpr, 0, 0, dpr, 0, 0);

      if (!reduced) {
        while (t > nextCell) { spawnCell(t); nextCell = t + 0.22 + Math.random() * 0.35; }
        while (t > nextPulse) { spawnPulse(t); nextPulse = t + 0.7 + Math.random() * 1.3; }
      }

      // cells fade in and out
      cells = cells.filter((c) => t - c.t0 < c.life);
      g.lineWidth = 1;
      for (const c of cells) {
        const p = Math.max(0, (t - c.t0) / c.life);
        const a = Math.sin(Math.PI * p) ** 2;
        const x = ox + c.c * CELL, y = c.r * CELL;
        g.fillStyle = `rgba(124,255,43,${(c.hot ? 0.075 : 0.035) * a})`;
        g.fillRect(x + 1, y + 1, CELL - 1, CELL - 1);
        if (c.hot) {
          g.strokeStyle = `rgba(124,255,43,${0.28 * a})`;
          g.strokeRect(x + 1.5, y + 1.5, CELL - 2, CELL - 2);
        }
      }

      // pulses: a bright head with a fading tail, running along a grid line
      const spanX = W / dpr, spanY = H / dpr;
      pulses = pulses.filter((q) => (t - q.t0) * q.speed < (q.across ? spanX : spanY) + q.len);
      g.lineWidth = 1.2;
      for (const q of pulses) {
        const d = (t - q.t0) * q.speed;
        const span = q.across ? spanX : spanY;
        const head = q.dir > 0 ? d : span - d;
        const tail = head - q.dir * q.len;
        const pos = q.across ? q.k * CELL + 0.5 : ox + q.k * CELL + 0.5;
        const [x0, y0, x1, y1] = q.across ? [tail, pos, head, pos] : [pos, tail, pos, head];
        const grad = g.createLinearGradient(x0, y0, x1, y1);
        grad.addColorStop(0, "rgba(124,255,43,0)");
        grad.addColorStop(1, "rgba(182,255,122,0.75)");
        g.strokeStyle = grad;
        g.beginPath();
        g.moveTo(x0, y0);
        g.lineTo(x1, y1);
        g.stroke();
      }

      // dust drifting up
      for (const s of dust) {
        if (!reduced) {
          s.y -= s.v * dt;
          if (s.y < -4) { s.y = spanY + 4; s.x = Math.random() * spanX; }
        }
        const a = 0.18 + 0.32 * (0.5 + 0.5 * Math.sin(t * 1.3 + s.p));
        g.fillStyle = `rgba(200,255,170,${a})`;
        g.beginPath();
        g.arc(s.x + Math.sin(t * 0.4 + s.p) * 6, s.y, s.r, 0, 6.2832);
        g.fill();
      }

      // the grid lights up around the cursor
      if (sx < -1000 && mx > -1000) { sx = mx; sy = my; }
      sx += (mx - sx) * 0.12;
      sy += (my - sy) * 0.12;
      if (mx > -1000) {
        const x = spot.getContext("2d");
        x.globalCompositeOperation = "source-over";
        x.clearRect(0, 0, W, H);
        x.drawImage(glowLines, 0, 0);
        x.globalCompositeOperation = "destination-in";
        const R = 240 * dpr;
        const rg = x.createRadialGradient(sx * dpr, sy * dpr, 0, sx * dpr, sy * dpr, R);
        rg.addColorStop(0, "rgba(0,0,0,0.55)");
        rg.addColorStop(1, "rgba(0,0,0,0)");
        x.fillStyle = rg;
        x.fillRect(0, 0, W, H);
        g.setTransform(1, 0, 0, 1, 0, 0);
        g.drawImage(spot, 0, 0);
      } else {
        sx = sy = -1e4;
      }
    }

    new ResizeObserver(resize).observe(host);
    resize();
    if (reduced) return;
    let running = false, raf = 0, onScreen = true;
    const t0 = performance.now();
    const loop = (now) => { draw((now - t0) / 1000); raf = requestAnimationFrame(loop); };
    const set = (on) => {
      if (on === running) return;
      running = on;
      if (on) raf = requestAnimationFrame(loop);
      else cancelAnimationFrame(raf);
    };
    new IntersectionObserver(([e]) => { onScreen = e.isIntersecting; set(onScreen && !document.hidden); }).observe(cv);
    document.addEventListener("visibilitychange", () => set(onScreen && !document.hidden));
  }

  // ---- hero: the X mask on the canonical face, drawn from the app's own design, with GLITCH --
  const canvas = document.querySelector("[data-face]");
  const accent = document.querySelector(".hero__accent");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  fetch("assets/face.json")
    .then((r) => r.json())
    .then(start)
    .catch(() => {}); // the page works without it

  function start(data) {
    const V = Float32Array.from(data.v);
    const T = Uint16Array.from(data.t);
    const X = data.x; // [boil][eyes, eye drips, seam, stitches, mouth drips][polyline] flat (tri, w1, w2)
    const K = data.k; // mesh units per design texel
    const n = V.length / 3;
    const nt = T.length / 3;
    const rx = new Float32Array(n), ry = new Float32Array(n), rz = new Float32Array(n);
    const px = new Float32Array(n), py = new Float32Array(n);
    const front = new Uint8Array(nt);
    const depth = new Float32Array(nt);
    const fill = new Array(nt);
    const order = new Uint16Array(nt);
    const L = [-0.35, 0.45, 1].map((c, _, a) => c / Math.hypot(...a));
    const GREEN = [124, 255, 43];
    const PAINT = [236, 232, 228];

    let W = 0, H = 0, dpr = 1;
    let buf = null, tint = null;
    const resize = () => {
      dpr = Math.min(devicePixelRatio || 1, 2);
      W = canvas.width = Math.round(canvas.clientWidth * dpr);
      H = canvas.height = Math.round(canvas.clientHeight * dpr);
      buf = document.createElement("canvas");
      tint = document.createElement("canvas");
      buf.width = tint.width = W;
      buf.height = tint.height = H;
      if (reduced) frame(0.4);
    };
    new ResizeObserver(resize).observe(canvas);

    // pointer steers the head a little
    let aimX = 0, aimY = 0, yaw = 0, pitch = 0;
    const hero = canvas.closest(".hero");
    hero.addEventListener("pointermove", (e) => {
      const r = hero.getBoundingClientRect();
      aimX = ((e.clientX - r.left) / r.width - 0.5) * 2;
      aimY = ((e.clientY - r.top) / r.height - 0.5) * 2;
    });
    hero.addEventListener("pointerleave", () => { aimX = aimY = 0; });

    const hash = (a, b) => {
      let h = (a * 374761393 + b * 668265263) | 0;
      h = Math.imul(h ^ (h >>> 13), 1274126177);
      return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
    };

    // Same shape as n0face.glitch.strength: most windows hold one short burst, plus rare hiccups.
    const WINDOW = 2.2;
    const strength = (t) => {
      const w = Math.floor(t / WINDOW);
      if (hash(w, 1) < 0.7) {
        const dur = 0.08 + hash(w, 2) * 0.37;
        const x = t - w * WINDOW - hash(w, 3) * (WINDOW - dur);
        if (x >= 0 && x < dur) return (0.55 + 0.45 * hash(w, 4)) * (1 - 0.5 * x / dur);
      }
      return hash(Math.floor(t * 30), 9) < 0.02 ? 0.3 : 0;
    };

    // a stroke polyline -> screen points, or null when any of it faces away (as the app does)
    const polyline = (f) => {
      const pts = [];
      for (let i = 0; i < f.length; i += 3) {
        const j = f[i];
        if (!front[j]) return null;
        const w1 = f[i + 1], w2 = f[i + 2], w0 = 1 - w1 - w2;
        const a = T[3 * j], b = T[3 * j + 1], c = T[3 * j + 2];
        pts.push(w0 * px[a] + w1 * px[b] + w2 * px[c], w0 * py[a] + w1 * py[b] + w2 * py[c]);
      }
      return pts;
    };
    const strokePath = (pts, dx, dy) => {
      ctx.beginPath();
      ctx.moveTo(pts[0] + dx, pts[1] + dy);
      for (let i = 2; i < pts.length; i += 2) ctx.lineTo(pts[i] + dx, pts[i + 1] + dy);
      if (pts.length === 2) ctx.lineTo(pts[0] + dx + 0.01, pts[1] + dy);
      ctx.stroke();
    };
    // spray-painted strokes: a solid core (optionally glowing) and an offset dry-brush pass
    const paint = (polys, rgb, width, scale, glow = 0) => {
      const tk = Math.max(width * scale, 1);
      ctx.lineWidth = tk;
      ctx.strokeStyle = `rgb(${rgb})`;
      ctx.shadowColor = `rgba(${rgb},0.85)`;
      ctx.shadowBlur = glow * 16 * dpr;
      for (const p of polys) strokePath(p, 0, 0);
      ctx.shadowBlur = 0;
      ctx.lineWidth = Math.max(tk / 2, 1);
      ctx.strokeStyle = `rgb(${rgb.map((c) => c * 0.8 | 0)})`;
      for (const p of polys) strokePath(p, 0.35 * tk, -0.25 * tk);
    };
    const dots = (polys, rgb, r) => {
      ctx.fillStyle = `rgb(${rgb})`;
      for (const p of polys) {
        ctx.beginPath();
        ctx.arc(p[p.length - 2], p[p.length - 1], r, 0, 6.2832);
        ctx.fill();
      }
    };

    let lastHeadline = -10;

    function frame(t) {
      if (!W || !H) return;
      const tYaw = Math.sin(t * 0.45) * 0.42 + aimX * 0.35;
      const tPitch = Math.sin(t * 0.31) * 0.06 - aimY * 0.16 + 0.04;
      yaw += (tYaw - yaw) * (reduced ? 1 : 0.06);
      pitch += (tPitch - pitch) * (reduced ? 1 : 0.06);
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const s = Math.min(H * 0.4, W * 0.4), D = 7, ox = W / 2, oy = H * 0.5;

      let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
      for (let i = 0; i < n; i++) {
        const x = V[3 * i], y = V[3 * i + 1], z = V[3 * i + 2];
        const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
        const y2 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
        rx[i] = x1; ry[i] = y2; rz[i] = z2;
        const k = D / (D - z2);
        px[i] = ox + x1 * s * k;
        py[i] = oy - y2 * s * k;
        if (px[i] < minX) minX = px[i];
        if (px[i] > maxX) maxX = px[i];
        if (py[i] < minY) minY = py[i];
        if (py[i] > maxY) maxY = py[i];
      }

      // lit, rim-glowing surface (like the app's Painter.surface), back to front
      let m = 0;
      for (let j = 0; j < nt; j++) {
        const a = T[3 * j], b = T[3 * j + 1], c = T[3 * j + 2];
        const cross = (px[b] - px[a]) * (py[c] - py[a]) - (py[b] - py[a]) * (px[c] - px[a]);
        front[j] = cross < 0 ? 1 : 0;
        if (!front[j]) continue;
        const ux = rx[b] - rx[a], uy = ry[b] - ry[a], uz = rz[b] - rz[a];
        const vx = rx[c] - rx[a], vy = ry[c] - ry[a], vz = rz[c] - rz[a];
        let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        const len = Math.hypot(nx, ny, nz) || 1;
        nx /= len; ny /= len; nz /= len;
        const shade = 0.35 + 0.65 * Math.max(nx * L[0] + ny * L[1] + nz * L[2], 0);
        const rim = Math.pow(1 - Math.max(Math.abs(nz), 0), 2.2) * 0.6;
        fill[j] = `rgb(${13 * shade + GREEN[0] * rim | 0},${13 * shade + GREEN[1] * rim | 0},${14 * shade + GREEN[2] * rim | 0})`;
        depth[j] = rz[a] + rz[b] + rz[c];
        order[m++] = j;
      }
      const vis = Array.from(order.subarray(0, m)).sort((p, q) => depth[p] - depth[q]);

      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.lineWidth = 0.8 * dpr;
      for (const j of vis) {
        const a = T[3 * j], b = T[3 * j + 1], c = T[3 * j + 2];
        ctx.beginPath();
        ctx.moveTo(px[a], py[a]);
        ctx.lineTo(px[b], py[b]);
        ctx.lineTo(px[c], py[c]);
        ctx.closePath();
        ctx.fillStyle = ctx.strokeStyle = fill[j];
        ctx.fill();
        ctx.stroke(); // hides the hairline seams between triangles
      }

      // the design: X eyes, stitched grin, paint drips (stop-motion "boil" at 6 fps like the app)
      const scale = K * s;
      const groups = X[reduced ? 0 : Math.floor(t * 6) % 3].map((g) => g.map(polyline).filter(Boolean));
      const [eyes, eyeDrips, seam, stitches, mouthDrips] = groups;
      paint(eyes, GREEN, 16, scale, 1);
      paint(seam, PAINT, 8, scale);
      paint(stitches, PAINT, 7, scale);
      paint(eyeDrips, GREEN, 5, scale, 0.6);
      dots(eyeDrips, GREEN, 3.6 * scale);
      paint(mouthDrips, PAINT, 3.5, scale);
      dots(mouthDrips, PAINT, 2.6 * scale);

      // GLITCH
      const k = reduced ? 0 : strength(t);
      if (k > 0) glitch(t, k, minX, minY, maxX, maxY);
      if (k > 0.5 && accent && t - lastHeadline > 7) {
        lastHeadline = t;
        accent.classList.remove("is-glitching");
        void accent.offsetWidth;
        accent.classList.add("is-glitching");
      }
    }

    // Port of n0face.glitch.apply: wave warp + slice shifts, RGB split, datamosh blocks, streaks.
    function glitch(t, k, x0, y0, x1, y1) {
      const b = buf.getContext("2d");
      b.clearRect(0, 0, W, H);
      b.drawImage(canvas, 0, 0);
      const fw = x1 - x0, pad = fw * 0.35;
      const Y0 = Math.max(y0 - 10 * dpr, 0), Y1 = Math.min(y1 + 10 * dpr, H);
      const seed = Math.floor(t * 24);
      const rnd = (i) => hash(seed, i);

      // wave warp and torn slices, row by row
      ctx.clearRect(0, Y0, W, Y1 - Y0);
      const slices = [];
      for (let i = 0; i < 2 + rnd(1) * 4; i++) {
        const sy0 = Y0 + rnd(10 + i) * (Y1 - Y0);
        slices.push([sy0, sy0 + (4 + rnd(20 + i) * 30) * dpr, (rnd(30 + i) - 0.5) * 0.5 * fw * k]);
      }
      const step = 3 * dpr, amp = 6 * dpr * k, freq = 0.02 / dpr;
      for (let y = Y0; y < Y1; y += step) {
        let dx = Math.sin(y * freq + t * 40) * amp;
        for (const [a, c, d] of slices) if (y >= a && y < c) dx += d;
        ctx.drawImage(buf, 0, y, W, step, dx, y, W, step);
      }

      // colour channels split apart
      const tc = tint.getContext("2d");
      const split = (8 + rnd(40) * 10) * dpr * k;
      for (const [rgb, dx] of [["255,40,210", -split], ["40,230,255", split]]) {
        tc.globalCompositeOperation = "source-over";
        tc.clearRect(0, 0, W, H);
        tc.drawImage(canvas, 0, 0);
        tc.globalCompositeOperation = "multiply";
        tc.fillStyle = `rgb(${rgb})`;
        tc.fillRect(0, 0, W, H);
        tc.globalCompositeOperation = "destination-in";
        tc.drawImage(canvas, 0, 0);
        ctx.globalCompositeOperation = "lighter";
        ctx.globalAlpha = 0.55;
        ctx.drawImage(tint, dx, 0);
      }
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = "source-over";

      // datamosh: blocks copied out of place
      for (let i = 0; i < 2 + rnd(50) * 4 * k; i++) {
        const bw = (20 + rnd(60 + i) * 60) * dpr, bh = (8 + rnd(70 + i) * 26) * dpr;
        const sx = x0 + rnd(80 + i) * fw, sy = Y0 + rnd(90 + i) * (Y1 - Y0);
        ctx.drawImage(buf, sx, sy, bw, bh, sx + (rnd(100 + i) - 0.5) * 80 * dpr, sy + (rnd(110 + i) - 0.5) * 20 * dpr, bw, bh);
      }

      // bright streaks past the face
      for (let i = 0; i < 3 + rnd(120) * 6; i++) {
        const y = Y0 + rnd(130 + i) * (Y1 - Y0);
        const sx = x0 - pad + rnd(140 + i) * (fw + 2 * pad);
        ctx.fillStyle = rnd(150 + i) < 0.5 ? "rgba(182,255,122,0.8)" : "rgba(240,240,240,0.7)";
        ctx.fillRect(sx, y, (30 + rnd(160 + i) * 140) * dpr, Math.max(1, dpr));
      }
    }

    resize();
    if (reduced) return;
    let running = false, raf = 0;
    const t0 = performance.now();
    const loop = (now) => {
      frame((now - t0) / 1000);
      raf = requestAnimationFrame(loop);
    };
    const set = (on) => {
      if (on === running) return;
      running = on;
      if (on) raf = requestAnimationFrame(loop);
      else cancelAnimationFrame(raf);
    };
    let onScreen = true;
    new IntersectionObserver(([e]) => { onScreen = e.isIntersecting; set(onScreen && !document.hidden); }).observe(canvas);
    document.addEventListener("visibilitychange", () => set(onScreen && !document.hidden));
    set(true);
  }
})();
