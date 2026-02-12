/**
 * Deep Blue — ROI Drawing + Status Poller
 * Zero dependencies · Raspberry Pi optimized
 */
(function () {
    'use strict';

    /* ── Config ─────────────────────────────── */
    var VW = 640, VH = 480;       // camera resolution
    var POLL_MS = 1500;           // status poll interval
    var TOAST_MS = 2800;           // toast display duration

    /* ── DOM refs ───────────────────────────── */
    var stream = document.getElementById('stream');
    var canvas = document.getElementById('roi-canvas');
    var ctx = canvas.getContext('2d');
    var loader = document.getElementById('loader');

    var btnDraw = document.getElementById('btn-draw');
    var btnUndo = document.getElementById('btn-undo');
    var btnSave = document.getElementById('btn-save');
    var btnClear = document.getElementById('btn-clear');
    var hint = document.getElementById('draw-hint');

    var elFps = document.getElementById('status-fps');
    var elRelay = document.getElementById('status-relay');
    var toastBox = document.getElementById('toast-wrap');

    /* ── State ──────────────────────────────── */
    var drawing = false;
    var verts = [];      // [[x,y], …] in 640×480 space
    var visible = true;

    /* ── Toast ──────────────────────────────── */
    function toast(msg, type) {
        var el = document.createElement('div');
        el.className = 'toast';
        el.textContent = msg;
        if (type === 'ok') el.style.borderColor = 'var(--green)';
        if (type === 'err') el.style.borderColor = 'var(--red)';
        toastBox.appendChild(el);
        setTimeout(function () {
            el.style.opacity = '0';
            el.style.transform = 'translateY(6px)';
            setTimeout(function () { el.remove(); }, 300);
        }, TOAST_MS);
    }

    /* ── Canvas sizing ─────────────────────── */
    // Position canvas exactly over the rendered <img>
    function syncCanvas() {
        var sr = stream.getBoundingClientRect();
        var pr = stream.parentElement.getBoundingClientRect();
        if (sr.width < 2 || sr.height < 2) return;

        // Offset relative to parent (.stream-wrap)
        var top = sr.top - pr.top;
        var left = sr.left - pr.left;

        canvas.width = sr.width;
        canvas.height = sr.height;
        canvas.style.width = sr.width + 'px';
        canvas.style.height = sr.height + 'px';
        canvas.style.top = top + 'px';
        canvas.style.left = left + 'px';

        render();
    }

    /* ── Coordinate mapping ────────────────── */
    function clickToVideo(e) {
        var r = canvas.getBoundingClientRect();
        var cx = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
        var cy = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
        var vx = Math.round(cx * VW / r.width);
        var vy = Math.round(cy * VH / r.height);
        return [Math.max(0, Math.min(VW, vx)),
        Math.max(0, Math.min(VH, vy))];
    }

    function v2c(pt) {
        return [pt[0] * canvas.width / VW,
        pt[1] * canvas.height / VH];
    }

    /* ── Render polygon ────────────────────── */
    function render() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!verts.length) return;

        // Polygon fill + stroke
        ctx.beginPath();
        var s = v2c(verts[0]);
        ctx.moveTo(s[0], s[1]);
        for (var i = 1; i < verts.length; i++) {
            var p = v2c(verts[i]);
            ctx.lineTo(p[0], p[1]);
        }
        if (verts.length >= 3) {
            ctx.closePath();
            ctx.fillStyle = 'rgba(0, 229, 255, 0.12)';
            ctx.fill();
        }
        ctx.strokeStyle = '#00e5ff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Vertex dots
        for (var j = 0; j < verts.length; j++) {
            var c = v2c(verts[j]);
            ctx.beginPath();
            ctx.arc(c[0], c[1], 4, 0, 6.283);
            ctx.fillStyle = j === 0 ? '#ffb700' : '#fff';
            ctx.fill();
        }
    }

    /* ── UI helpers ────────────────────────── */
    function syncButtons() {
        btnUndo.disabled = !drawing || verts.length === 0;
        btnSave.disabled = !drawing || verts.length < 3;
    }

    function enterDraw(on) {
        drawing = on;
        if (on) {
            btnDraw.textContent = '✕  Stop';
            btnDraw.classList.add('active');
            canvas.classList.add('drawing');
            hint.style.display = 'block';
            verts = [];
            render();
        } else {
            btnDraw.textContent = '✎  Draw Zone';
            btnDraw.classList.remove('active');
            canvas.classList.remove('drawing');
            hint.style.display = 'none';
        }
        syncButtons();
    }

    /* ── Events ────────────────────────────── */
    btnDraw.addEventListener('click', function () { enterDraw(!drawing); });

    canvas.addEventListener('click', function (e) {
        if (!drawing) return;
        verts.push(clickToVideo(e));
        requestAnimationFrame(render);
        syncButtons();
    });

    btnUndo.addEventListener('click', function () {
        verts.pop();
        requestAnimationFrame(render);
        syncButtons();
    });

    btnClear.addEventListener('click', function () {
        fetch('/api/roi', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ roi: [] })
        })
            .then(function (r) { return r.json(); })
            .then(function () {
                verts = [];
                requestAnimationFrame(render);
                if (drawing) enterDraw(false);
                toast('Zone cleared', 'ok');
            });
    });

    btnSave.addEventListener('click', function () {
        if (verts.length < 3) return;
        btnSave.disabled = true;
        btnSave.textContent = 'Saving…';

        fetch('/api/roi', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ roi: verts })
        })
            .then(function (r) { return r.json(); })
            .then(function () {
                toast('Zone saved', 'ok');
                enterDraw(false);
            })
            .catch(function () {
                toast('Save failed', 'err');
            })
            .then(function () {
                btnSave.textContent = 'Save Zone';
                syncButtons();
            });
    });

    /* ── Status polling ────────────────────── */
    function poll() {
        if (!visible) return;
        fetch('/api/status')
            .then(function (r) { return r.json(); })
            .then(function (s) {
                elFps.textContent = s.fps.toFixed(1);
                if (s.relay === 'ON') {
                    elRelay.textContent = '⚠ ALARM';
                    elRelay.className = 'badge badge-danger';
                } else {
                    elRelay.textContent = 'CLEAR';
                    elRelay.className = 'badge badge-safe';
                }
            })
            .catch(function () { });
    }

    /* ── Init ───────────────────────────────── */
    // Hide loader once first MJPEG frame arrives
    stream.addEventListener('load', function () {
        loader.classList.add('hidden');
        syncCanvas();
    });

    // Keep canvas aligned on resize
    var ro = new ResizeObserver(function () { syncCanvas(); });
    ro.observe(stream);

    // Pause polling when tab hidden
    document.addEventListener('visibilitychange', function () {
        visible = document.visibilityState === 'visible';
    });

    // Kick off
    setInterval(poll, POLL_MS);
    setTimeout(syncCanvas, 800);       // fallback initial sync
})();
