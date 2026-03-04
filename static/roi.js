/**
 * Deep Blue — Multi-Zone Drawing + State Machine Status + Control Settings
 * Zero dependencies · Raspberry Pi optimized
 */
(function () {
    'use strict';

    /* ── Config ─────────────────────────────── */
    var VW = 640, VH = 480;
    var POLL_MS = 1500;
    var TOAST_MS = 2800;
    var PRESET_COLORS = [
        '#00e5ff', '#00ff9d', '#ff3333', '#ffb700',
        '#2563eb', '#e040fb', '#ff6d00', '#76ff03'
    ];

    /* ── DOM refs ───────────────────────────── */
    var stream = document.getElementById('stream');
    var canvas = document.getElementById('roi-canvas');
    var ctx = canvas.getContext('2d');
    var loader = document.getElementById('loader');

    var zoneListEl = document.getElementById('zone-list');
    var zoneListPanel = document.getElementById('zone-list-panel');
    var zoneEditor = document.getElementById('zone-editor');
    var editorTitle = document.getElementById('editor-title');
    var btnAddZone = document.getElementById('btn-add-zone');
    var btnUndo = document.getElementById('btn-undo');
    var btnSaveZone = document.getElementById('btn-save-zone');
    var btnCancel = document.getElementById('btn-cancel-zone');
    var zoneNameInput = document.getElementById('zone-name');
    var colorSwatchesEl = document.getElementById('color-swatches');
    var hint = document.getElementById('draw-hint');
    var zoneTypeBtns = document.querySelectorAll('.zone-type-btn');
    var zoneWarningsEl = document.getElementById('zone-warnings');

    var elFps = document.getElementById('status-fps');
    var elRelay = document.getElementById('status-relay');
    var elState = document.getElementById('status-state');
    var elYolo = document.getElementById('status-yolo');
    var elWatchdog = document.getElementById('status-watchdog');
    var faultBanner = document.getElementById('fault-banner');
    var faultText = document.getElementById('fault-text');
    var toastBox = document.getElementById('toast-wrap');

    // Settings
    var settingsToggle = document.getElementById('settings-toggle');
    var settingsBody = document.getElementById('settings-body');
    var btnSaveSettings = document.getElementById('btn-save-settings');
    var settingsFeedback = document.getElementById('settings-feedback');
    var cfgClearance = document.getElementById('cfg-clearance');
    var cfgPedClear = document.getElementById('cfg-ped-clear');
    var cfgPassage = document.getElementById('cfg-passage');
    var cfgCooldown = document.getElementById('cfg-cooldown');
    var cfgMotionFrames = document.getElementById('cfg-motion-frames');
    var cfgMotionThresh = document.getElementById('cfg-motion-thresh');
    var cfgYoloPoll = document.getElementById('cfg-yolo-poll');
    var cfgWdInterval = document.getElementById('cfg-wd-interval');
    var cfgWdWidth = document.getElementById('cfg-wd-width');

    /* ── State ──────────────────────────────── */
    var allZones = [];
    var editingZoneId = null;
    var drawVerts = [];
    var selectedColor = PRESET_COLORS[0];
    var selectedZoneType = 'human';
    var editorOpen = false;
    var visible = true;
    var canvasDisplayWidth = 0;
    var canvasDisplayHeight = 0;
    var settingsOpen = false;

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

    /* ── Helpers ────────────────────────────── */
    function escapeHtml(s) {
        var d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    function hexToRgba(hex, alpha) {
        var r = parseInt(hex.slice(1, 3), 16);
        var g = parseInt(hex.slice(3, 5), 16);
        var b = parseInt(hex.slice(5, 7), 16);
        return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
    }

    /* ── Canvas sizing ─────────────────────── */
    function getVisibleStreamRect() {
        var sr = stream.getBoundingClientRect();
        if (sr.width < 2 || sr.height < 2) return null;
        var videoW = stream.naturalWidth || VW;
        var videoH = stream.naturalHeight || VH;
        if (videoW < 1 || videoH < 1) {
            videoW = VW;
            videoH = VH;
        }
        var videoAspect = videoW / videoH;
        var boxAspect = sr.width / sr.height;
        var vr = {
            left: sr.left,
            top: sr.top,
            width: sr.width,
            height: sr.height
        };
        if (boxAspect > videoAspect) {
            vr.height = sr.height;
            vr.width = sr.height * videoAspect;
            vr.left = sr.left + (sr.width - vr.width) / 2;
        } else {
            vr.width = sr.width;
            vr.height = sr.width / videoAspect;
            vr.top = sr.top + (sr.height - vr.height) / 2;
        }
        return vr;
    }

    function syncCanvas() {
        var vr = getVisibleStreamRect();
        var pr = stream.parentElement.getBoundingClientRect();
        if (!vr || vr.width < 2 || vr.height < 2) return;
        var top = vr.top - pr.top;
        var left = vr.left - pr.left;
        var cw = Math.max(1, vr.width);
        var ch = Math.max(1, vr.height);
        var dpr = window.devicePixelRatio || 1;
        canvasDisplayWidth = cw;
        canvasDisplayHeight = ch;
        canvas.width = Math.max(1, Math.round(cw * dpr));
        canvas.height = Math.max(1, Math.round(ch * dpr));
        canvas.style.width = cw + 'px';
        canvas.style.height = ch + 'px';
        canvas.style.top = top + 'px';
        canvas.style.left = left + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
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
        return [pt[0] * canvasDisplayWidth / VW,
                pt[1] * canvasDisplayHeight / VH];
    }

    /* ── Draw a polygon on canvas ──────────── */
    function drawPolygon(points, color, alpha, showDots) {
        if (points.length === 0) return;
        ctx.beginPath();
        var s = v2c(points[0]);
        ctx.moveTo(s[0], s[1]);
        for (var i = 1; i < points.length; i++) {
            var p = v2c(points[i]);
            ctx.lineTo(p[0], p[1]);
        }
        if (points.length >= 3) {
            ctx.closePath();
            ctx.fillStyle = hexToRgba(color, alpha);
            ctx.fill();
        }
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.75)';
        ctx.lineWidth = 4;
        ctx.stroke();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.stroke();
        if (showDots) {
            for (var j = 0; j < points.length; j++) {
                var c = v2c(points[j]);
                ctx.beginPath();
                ctx.arc(c[0], c[1], 4, 0, 6.283);
                ctx.fillStyle = j === 0 ? '#ffb700' : '#fff';
                ctx.fill();
            }
        }
    }

    /* ── Render all zones + current drawing ── */
    function render() {
        if (canvasDisplayWidth < 2 || canvasDisplayHeight < 2) return;
        ctx.clearRect(0, 0, canvasDisplayWidth, canvasDisplayHeight);
        if (editorOpen) {
            for (var i = 0; i < allZones.length; i++) {
                var z = allZones[i];
                if (z.id === editingZoneId) continue;
                if (z.points.length < 3) continue;
                drawPolygon(z.points, z.color, 0.22, false);
            }
        }
        if (editorOpen && drawVerts.length > 0) {
            drawPolygon(drawVerts, selectedColor, 0.30, true);
        }
    }

    /* ── Zone list rendering ───────────────── */
    function renderZoneList() {
        zoneListEl.innerHTML = '';
        if (allZones.length === 0) {
            zoneListEl.innerHTML = '<p style="font-size:.72rem;color:var(--text-dim)">No zones defined. Add one to start monitoring.</p>';
            return;
        }
        for (var i = 0; i < allZones.length; i++) {
            (function (zone) {
                var item = document.createElement('div');
                item.className = 'zone-item';
                var isRoad = zone.zone_type === 'vehicle_road';
                var badgeClass = isRoad ? 'zone-type-badge zone-type-road' : 'zone-type-badge zone-type-human';
                var badgeText = isRoad ? 'ROAD' : 'HUMAN';
                var pixelSpan = isRoad
                    ? '<span class="zone-pixel-pct" data-zone-id="' + zone.id + '">--%</span>'
                    : '';
                item.innerHTML =
                    '<span class="zone-swatch" style="background:' + zone.color + '"></span>' +
                    '<span class="zone-name">' + escapeHtml(zone.name) + '</span>' +
                    pixelSpan +
                    '<span class="' + badgeClass + '">' + badgeText + '</span>' +
                    '<button class="zone-btn-edit" title="Edit">&#9998;</button>' +
                    '<button class="zone-btn-del" title="Delete">&times;</button>';
                item.querySelector('.zone-btn-edit').addEventListener('click', function () { editZone(zone.id); });
                item.querySelector('.zone-btn-del').addEventListener('click', function () { deleteZone(zone.id); });
                zoneListEl.appendChild(item);
            })(allZones[i]);
        }
        updateZoneWarnings();
    }

    function updateZonePixelPcts(pcts) {
        if (!pcts) return;
        var spans = zoneListEl.querySelectorAll('.zone-pixel-pct');
        for (var i = 0; i < spans.length; i++) {
            var zid = spans[i].getAttribute('data-zone-id');
            if (zid && pcts[zid] !== undefined) {
                spans[i].textContent = pcts[zid].toFixed(1) + '%';
            } else {
                spans[i].textContent = '--%';
            }
        }
    }

    function updateZoneWarnings() {
        var hasHuman = false;
        var hasRoad = false;
        for (var i = 0; i < allZones.length; i++) {
            if (allZones[i].zone_type === 'human') hasHuman = true;
            if (allZones[i].zone_type === 'vehicle_road') hasRoad = true;
        }
        var warnings = [];
        if (!hasRoad && allZones.length > 0) {
            warnings.push('No vehicle road zone — motion trigger disabled.');
        }
        if (!hasHuman && allZones.length > 0) {
            warnings.push('No human zone — passage grant blocked (unsafe).');
        }
        if (allZones.length === 0) {
            warnings.push('Add at least one human zone and one vehicle road zone.');
        }
        if (warnings.length > 0) {
            zoneWarningsEl.innerHTML = '';
            for (var w = 0; w < warnings.length; w++) {
                var el = document.createElement('div');
                el.className = 'zone-warning';
                el.textContent = warnings[w];
                zoneWarningsEl.appendChild(el);
            }
            zoneWarningsEl.style.display = '';
        } else {
            zoneWarningsEl.style.display = 'none';
        }
    }

    /* ── Color swatches ────────────────────── */
    function initColorSwatches() {
        colorSwatchesEl.innerHTML = '';
        for (var i = 0; i < PRESET_COLORS.length; i++) {
            (function (c) {
                var el = document.createElement('div');
                el.className = 'color-swatch' + (c === selectedColor ? ' selected' : '');
                el.style.background = c;
                el.addEventListener('click', function () {
                    selectedColor = c;
                    var all = colorSwatchesEl.querySelectorAll('.color-swatch');
                    for (var j = 0; j < all.length; j++) all[j].classList.remove('selected');
                    el.classList.add('selected');
                    requestAnimationFrame(render);
                });
                colorSwatchesEl.appendChild(el);
            })(PRESET_COLORS[i]);
        }
    }

    function selectColorSwatch(color) {
        selectedColor = color;
        var all = colorSwatchesEl.querySelectorAll('.color-swatch');
        for (var i = 0; i < all.length; i++) {
            all[i].classList.remove('selected');
            if (all[i].style.background === color || PRESET_COLORS[i] === color) {
                all[i].classList.add('selected');
            }
        }
    }

    /* ── Zone type selector ─────────────────── */
    function selectZoneType(type) {
        selectedZoneType = type;
        for (var i = 0; i < zoneTypeBtns.length; i++) {
            var btn = zoneTypeBtns[i];
            if (btn.getAttribute('data-type') === type) {
                btn.classList.add('selected');
            } else {
                btn.classList.remove('selected');
            }
        }
        if (type === 'vehicle_road') {
            hint.innerHTML = '<strong>EDIT MODE</strong><br>Draw a vehicle road zone.<br>Motion detection (camera-based) will be used.';
        } else {
            hint.innerHTML = '<strong>EDIT MODE</strong><br>Click on the video to place vertices.<br>Minimum 3 points required.';
        }
    }

    for (var ti = 0; ti < zoneTypeBtns.length; ti++) {
        (function (btn) {
            btn.addEventListener('click', function () {
                selectZoneType(btn.getAttribute('data-type'));
            });
        })(zoneTypeBtns[ti]);
    }

    /* ── UI helpers ────────────────────────── */
    function syncButtons() {
        btnUndo.disabled = drawVerts.length === 0;
        btnSaveZone.disabled = drawVerts.length < 3;
    }

    function showEditor(show) {
        editorOpen = show;
        zoneEditor.style.display = show ? '' : 'none';
        btnAddZone.style.display = show ? 'none' : '';
        canvas.classList.toggle('drawing', show);
        hint.style.display = show ? 'block' : 'none';
        if (!show) {
            editingZoneId = null;
            drawVerts = [];
        }
        syncButtons();
        requestAnimationFrame(render);
    }

    /* ── Zone operations ───────────────────── */
    function loadZones() {
        fetch('/api/zones')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                allZones = data.zones || [];
                renderZoneList();
                requestAnimationFrame(render);
            });
    }

    function startAddZone() {
        editingZoneId = null;
        drawVerts = [];
        selectedColor = PRESET_COLORS[allZones.length % PRESET_COLORS.length];
        zoneNameInput.value = 'Zone ' + (allZones.length + 1);
        editorTitle.textContent = 'New Zone';
        selectZoneType('human');
        initColorSwatches();
        selectColorSwatch(selectedColor);
        showEditor(true);
    }

    function editZone(id) {
        var zone = null;
        for (var i = 0; i < allZones.length; i++) {
            if (allZones[i].id === id) { zone = allZones[i]; break; }
        }
        if (!zone) return;
        editingZoneId = id;
        drawVerts = zone.points.slice();
        selectedColor = zone.color;
        zoneNameInput.value = zone.name;
        editorTitle.textContent = 'Edit Zone';
        selectZoneType(zone.zone_type || 'human');
        initColorSwatches();
        selectColorSwatch(selectedColor);
        showEditor(true);
    }

    function deleteZone(id) {
        fetch('/api/zones/' + id, { method: 'DELETE' })
            .then(function (r) { return r.json(); })
            .then(function () {
                allZones = allZones.filter(function (z) { return z.id !== id; });
                renderZoneList();
                requestAnimationFrame(render);
                toast('Zone deleted', 'ok');
            })
            .catch(function () { toast('Delete failed', 'err'); });
    }

    function saveCurrentZone() {
        var name = zoneNameInput.value.trim() || 'Unnamed';
        if (drawVerts.length < 3) return;
        btnSaveZone.disabled = true;
        btnSaveZone.textContent = 'Saving…';
        var done = function () {
            btnSaveZone.textContent = 'Save Zone';
            syncButtons();
        };
        if (editingZoneId) {
            fetch('/api/zones/' + editingZoneId, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, color: selectedColor, zone_type: selectedZoneType, points: drawVerts })
            })
                .then(function (r) { return r.json(); })
                .then(function () {
                    loadZones();
                    showEditor(false);
                    toast('Zone updated', 'ok');
                })
                .catch(function () { toast('Update failed', 'err'); })
                .then(done);
        } else {
            fetch('/api/zones', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, color: selectedColor, zone_type: selectedZoneType, points: drawVerts })
            })
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (data.ok === false) {
                        toast(data.error || 'Failed', 'err');
                    } else {
                        loadZones();
                        showEditor(false);
                        toast('Zone saved', 'ok');
                    }
                })
                .catch(function () { toast('Save failed', 'err'); })
                .then(done);
        }
    }

    /* ── Settings panel ───────────────────── */
    settingsToggle.addEventListener('click', function () {
        settingsOpen = !settingsOpen;
        settingsBody.style.display = settingsOpen ? '' : 'none';
        if (settingsOpen) loadSettings();
    });

    function loadSettings() {
        fetch('/api/control-config')
            .then(function (r) { return r.json(); })
            .then(function (cfg) {
                cfgClearance.value = cfg.clearance_seconds;
                cfgPedClear.value = cfg.pedestrian_clear_consecutive_frames;
                cfgPassage.value = cfg.passage_seconds;
                cfgCooldown.value = cfg.cooldown_seconds;
                cfgMotionFrames.value = cfg.motion_consecutive_frames;
                cfgMotionThresh.value = cfg.motion_pixel_threshold;
                cfgYoloPoll.value = cfg.yolo_poll_interval_ms;
                cfgWdInterval.value = cfg.watchdog_pulse_interval_ms;
                cfgWdWidth.value = cfg.watchdog_pulse_width_ms;
                settingsFeedback.textContent = '';
            })
            .catch(function () {
                settingsFeedback.textContent = 'Failed to load settings.';
                settingsFeedback.className = 'settings-feedback err';
            });
    }

    btnSaveSettings.addEventListener('click', function () {
        var payload = {
            clearance_seconds: parseFloat(cfgClearance.value),
            pedestrian_clear_consecutive_frames: parseInt(cfgPedClear.value, 10),
            passage_seconds: parseFloat(cfgPassage.value),
            cooldown_seconds: parseFloat(cfgCooldown.value),
            motion_consecutive_frames: parseInt(cfgMotionFrames.value, 10),
            motion_pixel_threshold: parseFloat(cfgMotionThresh.value),
            yolo_poll_interval_ms: parseInt(cfgYoloPoll.value, 10),
            watchdog_pulse_interval_ms: parseInt(cfgWdInterval.value, 10),
            watchdog_pulse_width_ms: parseInt(cfgWdWidth.value, 10)
        };
        // Client-side NaN check
        for (var k in payload) {
            if (isNaN(payload[k])) {
                settingsFeedback.textContent = 'Invalid value for ' + k;
                settingsFeedback.className = 'settings-feedback err';
                return;
            }
        }
        btnSaveSettings.disabled = true;
        btnSaveSettings.textContent = 'Saving…';
        fetch('/api/control-config', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.ok) {
                    settingsFeedback.textContent = 'Saved.';
                    settingsFeedback.className = 'settings-feedback ok';
                    toast('Settings saved', 'ok');
                } else {
                    var msg = (data.errors || []).join('; ') || 'Validation failed';
                    settingsFeedback.textContent = msg;
                    settingsFeedback.className = 'settings-feedback err';
                    toast('Settings rejected', 'err');
                }
            })
            .catch(function () {
                settingsFeedback.textContent = 'Network error.';
                settingsFeedback.className = 'settings-feedback err';
            })
            .then(function () {
                btnSaveSettings.disabled = false;
                btnSaveSettings.textContent = 'Save Settings';
            });
    });

    /* ── Events ────────────────────────────── */
    btnAddZone.addEventListener('click', startAddZone);

    canvas.addEventListener('click', function (e) {
        if (!editorOpen) return;
        drawVerts.push(clickToVideo(e));
        requestAnimationFrame(render);
        syncButtons();
    });

    btnUndo.addEventListener('click', function () {
        drawVerts.pop();
        requestAnimationFrame(render);
        syncButtons();
    });

    btnCancel.addEventListener('click', function () {
        showEditor(false);
    });

    btnSaveZone.addEventListener('click', saveCurrentZone);

    /* ── State display helpers ────────────── */
    var STATE_LABELS = {
        'IDLE_SAFE': 'IDLE',
        'ROAD_MOTION_TRIGGERED': 'MOTION',
        'CLEARANCE_WAIT': 'CLEARING',
        'PEDESTRIAN_HOLD': 'PED HOLD',
        'TRAFFIC_PASSAGE_GRANTED': 'PASSAGE',
        'RECOVERY_COOLDOWN': 'COOLDOWN',
        'FAULT_SAFE': 'FAULT'
    };

    var STATE_BADGE_CLASS = {
        'IDLE_SAFE': 'badge badge-dim',
        'ROAD_MOTION_TRIGGERED': 'badge badge-amber',
        'CLEARANCE_WAIT': 'badge badge-amber',
        'PEDESTRIAN_HOLD': 'badge badge-danger',
        'TRAFFIC_PASSAGE_GRANTED': 'badge badge-safe',
        'RECOVERY_COOLDOWN': 'badge badge-dim',
        'FAULT_SAFE': 'badge badge-fault'
    };

    /* ── Status polling ────────────────────── */
    function poll() {
        if (!visible) return;
        fetch('/api/status')
            .then(function (r) { return r.json(); })
            .then(function (s) {
                // FPS
                elFps.textContent = s.fps.toFixed(1);

                // Relay
                if (s.relay === 'ON') {
                    elRelay.textContent = 'PASS';
                    elRelay.className = 'badge badge-safe';
                } else {
                    elRelay.textContent = 'SAFE';
                    elRelay.className = 'badge badge-danger';
                }

                // Control state
                var stateKey = s.control_state || 'IDLE_SAFE';
                elState.textContent = STATE_LABELS[stateKey] || stateKey;
                elState.className = STATE_BADGE_CLASS[stateKey] || 'badge badge-dim';

                // YOLO mode
                if (s.yolo_mode === 'active') {
                    elYolo.textContent = 'ACTIVE';
                    elYolo.className = 'badge badge-safe';
                } else {
                    elYolo.textContent = 'SLEEP';
                    elYolo.className = 'badge badge-dim';
                }

                // Watchdog
                if (s.watchdog_ok) {
                    elWatchdog.textContent = 'OK';
                    elWatchdog.className = 'badge badge-safe';
                } else {
                    elWatchdog.textContent = 'FAIL';
                    elWatchdog.className = 'badge badge-fault';
                }

                // Live pixel %
                updateZonePixelPcts(s.zone_pixel_pcts);

                // Fault banner
                if (s.fault_active && s.fault_reason) {
                    faultText.textContent = 'FAULT: ' + s.fault_reason.replace(/_/g, ' ').toUpperCase();
                    faultBanner.style.display = '';
                } else {
                    faultBanner.style.display = 'none';
                }
            })
            .catch(function () { });
    }

    /* ── Init ───────────────────────────────── */
    stream.addEventListener('load', function () {
        loader.classList.add('hidden');
        syncCanvas();
    });

    var ro = new ResizeObserver(function () { syncCanvas(); });
    ro.observe(stream);
    if (stream.parentElement) ro.observe(stream.parentElement);
    window.addEventListener('resize', syncCanvas);

    document.addEventListener('visibilitychange', function () {
        visible = document.visibilityState === 'visible';
    });

    loadZones();
    setInterval(poll, POLL_MS);
    setTimeout(syncCanvas, 800);
})();
