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

  // ---- hero face -----------------------------------------------------------------------------
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
    const n = V.length / 3;
    const nt = T.length / 3;
    const px = new Float32Array(n), py = new Float32Array(n), pz = new Float32Array(n);
    const order = new Uint16Array(nt);
    const depth = new Float32Array(nt);
    const shade = new Float32Array(nt);
    const KEY = [1, 10, 33, 133, 152, 263, 362, 61, 291, 234, 454];

    let W = 0, H = 0, dpr = 1;
    const resize = () => {
      dpr = Math.min(devicePixelRatio || 1, 2);
      W = Math.round(canvas.clientWidth * dpr);
      H = Math.round(canvas.clientHeight * dpr);
      canvas.width = W;
      canvas.height = H;
      if (reduced) frame(4.2);
    };
    new ResizeObserver(resize).observe(canvas);
    resize();

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

    let glitchUntil = 0, nextGlitch = 3.5;

    function frame(t) {
      if (!W || !H) return;
      // pose
      const tYaw = Math.sin(t * 0.45) * 0.42 + aimX * 0.35;
      const tPitch = Math.sin(t * 0.31) * 0.06 - aimY * 0.16 + 0.04;
      yaw += (tYaw - yaw) * 0.06;
      pitch += (tPitch - pitch) * 0.06;
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const s = Math.min(H * 0.4, W * 0.4), D = 7, ox = W / 2, oy = H * 0.5;

      let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
      for (let i = 0; i < n; i++) {
        const x = V[3 * i], y = V[3 * i + 1], z = V[3 * i + 2];
        const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
        const y2 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
        const k = D / (D - z2);
        px[i] = ox + x1 * s * k;
        py[i] = oy - y2 * s * k;
        pz[i] = z2;
        if (px[i] < minX) minX = px[i];
        if (px[i] > maxX) maxX = px[i];
        if (py[i] < minY) minY = py[i];
        if (py[i] > maxY) maxY = py[i];
      }

      // visible triangles, back to front
      let m = 0;
      for (let j = 0; j < nt; j++) {
        const a = T[3 * j], b = T[3 * j + 1], c = T[3 * j + 2];
        const cross = (px[b] - px[a]) * (py[c] - py[a]) - (py[b] - py[a]) * (px[c] - px[a]);
        if (cross >= 0) continue; // screen y is down, so front faces wind negative
        const area = -cross / (s * s);
        shade[j] = Math.min(1, area * 220);
        depth[j] = pz[a] + pz[b] + pz[c];
        order[m++] = j;
      }
      const vis = Array.from(order.subarray(0, m)).sort((p, q) => depth[p] - depth[q]);

      // scan line sweeps across the face and back
      const sweep = 0.5 - 0.5 * Math.cos(t * 0.55);
      const xs = minX - 30 * dpr + (maxX - minX + 60 * dpr) * (0.12 + sweep * 0.76);

      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, W, H);

      // left of the scan: the mesh
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, xs, H);
      ctx.clip();
      ctx.lineJoin = "round";
      ctx.lineWidth = 1.1 * dpr;
      for (const j of vis) {
        const a = T[3 * j], b = T[3 * j + 1], c = T[3 * j + 2];
        ctx.beginPath();
        ctx.moveTo(px[a], py[a]);
        ctx.lineTo(px[b], py[b]);
        ctx.lineTo(px[c], py[c]);
        ctx.closePath();
        const l = shade[j];
        ctx.fillStyle = `rgb(${6 + 10 * l | 0},${12 + 34 * l | 0},${8 + 10 * l | 0})`;
        ctx.fill();
        ctx.strokeStyle = `rgba(124,255,43,${0.3 + 0.65 * l})`;
        ctx.stroke();
      }
      ctx.fillStyle = "#f2fff0";
      for (const i of KEY) {
        if (px[i] > xs) continue;
        const front = pz[i] > -0.2;
        if (!front) continue;
        ctx.beginPath();
        ctx.arc(px[i], py[i], 2.2 * dpr, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      // right of the scan: the face as redacted blocks
      const step = 7 * dpr, bar = 4 * dpr;
      const rows = Math.ceil((maxY - minY) / step) + 1;
      const lo = new Float32Array(rows).fill(1e9), hi = new Float32Array(rows).fill(-1e9);
      for (const j of vis) {
        const a = T[3 * j], b = T[3 * j + 1], c = T[3 * j + 2];
        const y0 = Math.min(py[a], py[b], py[c]), y1 = Math.max(py[a], py[b], py[c]);
        const x0 = Math.min(px[a], px[b], px[c]), x1 = Math.max(px[a], px[b], px[c]);
        for (let r = Math.ceil((y0 - minY) / step); r <= Math.floor((y1 - minY) / step); r++) {
          if (x0 < lo[r]) lo[r] = x0;
          if (x1 > hi[r]) hi[r] = x1;
        }
      }
      ctx.save();
      ctx.beginPath();
      ctx.rect(xs, 0, W - xs, H);
      ctx.clip();
      const tick = Math.floor(t * 6);
      for (let r = 0; r < rows; r++) {
        if (hi[r] < lo[r]) continue;
        const y = minY + r * step;
        const seed = r * 7 + Math.floor(tick / (2 + (r % 3)));
        let x = lo[r], k = 0;
        while (x < hi[r]) {
          const len = (6 + hash(seed, k) * 34) * dpr;
          const gap = (2 + hash(seed, k + 99) * 5) * dpr;
          const g = 150 + hash(seed, k + 7) * 90 | 0;
          ctx.fillStyle = `rgba(${g},${g + 6},${g},${0.5 + hash(seed, k + 3) * 0.45})`;
          ctx.fillRect(x, y, Math.min(len, hi[r] - x), bar);
          x += len + gap;
          k++;
        }
        if (hash(seed, 1234) > 0.72) {
          ctx.fillStyle = "rgba(210,220,210,0.28)";
          ctx.fillRect(hi[r] + 4 * dpr, y + 1 * dpr, (20 + hash(seed, 77) * 90) * dpr, 2 * dpr);
        }
      }
      ctx.restore();

      // the scan line itself
      const top = minY - 26 * dpr, bottom = maxY + 26 * dpr;
      const halo = ctx.createLinearGradient(xs - 40 * dpr, 0, xs + 4 * dpr, 0);
      halo.addColorStop(0, "rgba(124,255,43,0)");
      halo.addColorStop(1, "rgba(124,255,43,0.16)");
      ctx.fillStyle = halo;
      ctx.fillRect(xs - 40 * dpr, top, 44 * dpr, bottom - top);
      const beam = ctx.createLinearGradient(0, top, 0, bottom);
      beam.addColorStop(0, "rgba(220,255,200,0)");
      beam.addColorStop(0.15, "rgba(220,255,200,0.95)");
      beam.addColorStop(0.85, "rgba(220,255,200,0.95)");
      beam.addColorStop(1, "rgba(220,255,200,0)");
      ctx.fillStyle = beam;
      ctx.fillRect(xs - 0.75 * dpr, top, 1.5 * dpr, bottom - top);

      // GLITCH: a short burst every few seconds, like the app's effect
      if (!reduced && t > nextGlitch) {
        glitchUntil = t + 0.32;
        nextGlitch = t + 4.5 + Math.random() * 4;
        if (accent) {
          accent.classList.remove("is-glitching");
          void accent.offsetWidth;
          accent.classList.add("is-glitching");
        }
      }
      if (t < glitchUntil) {
        const slices = 3 + (Math.random() * 4 | 0);
        for (let i = 0; i < slices; i++) {
          const y = minY + Math.random() * (maxY - minY);
          const h = (4 + Math.random() * 26) * dpr;
          const dx = (Math.random() - 0.5) * 60 * dpr;
          ctx.drawImage(canvas, 0, y, W, h, dx, y, W, h);
        }
      }
    }

    if (reduced) {
      frame(4.2);
      return;
    }
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
