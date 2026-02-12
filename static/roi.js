// ROI Polygon Drawing for Deep Blue Dashboard
(function() {
    const stream = document.getElementById("stream");
    const canvas = document.getElementById("roi-canvas");
    const ctx = canvas.getContext("2d");
    const container = document.getElementById("stream-container");

    const btnDraw = document.getElementById("btn-draw");
    const btnUndo = document.getElementById("btn-undo");
    const btnSave = document.getElementById("btn-save");
    const btnClear = document.getElementById("btn-clear");
    const hint = document.getElementById("draw-hint");

    const statusFps = document.getElementById("status-fps");
    const statusRelay = document.getElementById("status-relay");

    let drawing = false;
    let vertices = [];
    // Actual video resolution for coordinate mapping
    const VIDEO_W = 640;
    const VIDEO_H = 480;

    // Resize canvas to match displayed image
    function resizeCanvas() {
        canvas.width = stream.clientWidth;
        canvas.height = stream.clientHeight;
        drawVertices();
    }

    stream.addEventListener("load", resizeCanvas);
    window.addEventListener("resize", resizeCanvas);
    // Initial resize after stream starts
    setTimeout(resizeCanvas, 1000);

    // Convert click position to video coordinates (640x480)
    function toVideoCoords(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = VIDEO_W / canvas.width;
        const scaleY = VIDEO_H / canvas.height;
        return [
            Math.round((e.clientX - rect.left) * scaleX),
            Math.round((e.clientY - rect.top) * scaleY)
        ];
    }

    // Convert video coordinates to canvas coordinates for drawing
    function toCanvasCoords(pt) {
        return [
            pt[0] * canvas.width / VIDEO_W,
            pt[1] * canvas.height / VIDEO_H
        ];
    }

    function drawVertices() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (vertices.length === 0) return;

        // Draw filled polygon
        if (vertices.length >= 3) {
            ctx.beginPath();
            let cp = toCanvasCoords(vertices[0]);
            ctx.moveTo(cp[0], cp[1]);
            for (let i = 1; i < vertices.length; i++) {
                cp = toCanvasCoords(vertices[i]);
                ctx.lineTo(cp[0], cp[1]);
            }
            ctx.closePath();
            ctx.fillStyle = "rgba(0, 200, 0, 0.15)";
            ctx.fill();
        }

        // Draw lines
        ctx.beginPath();
        ctx.strokeStyle = "#00ff00";
        ctx.lineWidth = 2;
        let cp = toCanvasCoords(vertices[0]);
        ctx.moveTo(cp[0], cp[1]);
        for (let i = 1; i < vertices.length; i++) {
            cp = toCanvasCoords(vertices[i]);
            ctx.lineTo(cp[0], cp[1]);
        }
        if (vertices.length >= 3) ctx.closePath();
        ctx.stroke();

        // Draw vertex circles
        vertices.forEach(function(v, i) {
            const c = toCanvasCoords(v);
            ctx.beginPath();
            ctx.arc(c[0], c[1], 5, 0, Math.PI * 2);
            ctx.fillStyle = i === 0 ? "#ff9800" : "#00ff00";
            ctx.fill();
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 1;
            ctx.stroke();
        });
    }

    // Toggle draw mode
    btnDraw.addEventListener("click", function() {
        drawing = !drawing;
        if (drawing) {
            canvas.style.display = "block";
            btnDraw.textContent = "Stop Drawing";
            btnDraw.classList.add("active");
            hint.style.display = "block";
            vertices = [];
            drawVertices();
        } else {
            canvas.style.display = "none";
            btnDraw.textContent = "Draw Zone";
            btnDraw.classList.remove("active");
            hint.style.display = "none";
        }
        updateButtons();
    });

    canvas.addEventListener("click", function(e) {
        if (!drawing) return;
        const pt = toVideoCoords(e);
        vertices.push(pt);
        drawVertices();
        updateButtons();
    });

    btnUndo.addEventListener("click", function() {
        vertices.pop();
        drawVertices();
        updateButtons();
    });

    btnSave.addEventListener("click", function() {
        if (vertices.length < 3) return;
        fetch("/api/roi", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({roi: vertices})
        }).then(function(r) { return r.json(); }).then(function() {
            drawing = false;
            canvas.style.display = "none";
            btnDraw.textContent = "Draw Zone";
            btnDraw.classList.remove("active");
            hint.style.display = "none";
            vertices = [];
            updateButtons();
        });
    });

    btnClear.addEventListener("click", function() {
        fetch("/api/roi", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({roi: []})
        }).then(function(r) { return r.json(); }).then(function() {
            vertices = [];
            drawVertices();
            updateButtons();
        });
    });

    function updateButtons() {
        btnUndo.disabled = !drawing || vertices.length === 0;
        btnSave.disabled = !drawing || vertices.length < 3;
    }

    // Load existing ROI on page load
    fetch("/api/roi").then(function(r) { return r.json(); }).then(function(data) {
        if (data.roi && data.roi.length >= 3) {
            // Show saved ROI is rendered server-side on the video
        }
    });

    // Poll status
    function pollStatus() {
        fetch("/api/status").then(function(r) { return r.json(); }).then(function(s) {
            statusFps.textContent = s.fps + " FPS";
            if (s.relay === "ON") {
                statusRelay.textContent = "ALARM ON";
                statusRelay.className = "stat relay-on";
            } else {
                statusRelay.textContent = "CLEAR";
                statusRelay.className = "stat relay-off";
            }
        }).catch(function() {});
    }
    setInterval(pollStatus, 1000);
})();
