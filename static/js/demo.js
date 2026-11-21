/*
 * demo.js - phần tương tác "kéo đường thẳng" trong trang giải thích.
 *
 * Mỗi chấm là một căn nhà (diện tích, giá). Đường thẳng ŷ = w0 + w1·x.
 * J(w) là trung bình bình phương sai số. Nút "Đặt w tối ưu" dùng công thức
 * bình phương tối thiểu cho một biến:  w1 = cov(x, y) / var(x),  w0 = mean(y) - w1·mean(x)
 * (đây là nghiệm mà Linear Regression tìm ra, chỉ khác là mô hình thật có 856 hệ số).
 */
(function () {
  "use strict";

  var holder = document.getElementById("demo-data");
  var svg = document.getElementById("demo-chart");
  if (!holder || !svg) return;

  var NS = "http://www.w3.org/2000/svg";
  var pts = JSON.parse(holder.textContent).points;
  var n = pts.length;

  // ---------- phép tính ----------
  var mx = 0, my = 0, i;
  for (i = 0; i < n; i++) { mx += pts[i][0]; my += pts[i][1]; }
  mx /= n; my /= n;
  var sxy = 0, sxx = 0;
  for (i = 0; i < n; i++) {
    sxy += (pts[i][0] - mx) * (pts[i][1] - my);
    sxx += (pts[i][0] - mx) * (pts[i][0] - mx);
  }
  var best = { w1: sxy / sxx };
  best.w0 = my - best.w1 * mx;

  function mse(w0, w1) {
    var s = 0, e, k;
    for (k = 0; k < n; k++) { e = pts[k][1] - (w0 + w1 * pts[k][0]); s += e * e; }
    return s / n;
  }
  var bestJ = mse(best.w0, best.w1);

  function fmt(v, d) {
    return v.toLocaleString("vi-VN", { minimumFractionDigits: d, maximumFractionDigits: d });
  }

  // ---------- khung vẽ ----------
  var W = 640, H = 380, M = { l: 54, r: 16, t: 14, b: 46 };
  var xs = pts.map(function (p) { return p[0]; });
  var ys = pts.map(function (p) { return p[1]; });
  var xmin = Math.floor(Math.min.apply(null, xs) / 10) * 10 - 10;
  var xmax = Math.ceil(Math.max.apply(null, xs) / 10) * 10 + 10;
  var ymin = 0;
  var ymax = Math.ceil(Math.max.apply(null, ys)) + 1;

  function X(v) { return M.l + (v - xmin) / (xmax - xmin) * (W - M.l - M.r); }
  function Y(v) { return H - M.b - (v - ymin) / (ymax - ymin) * (H - M.t - M.b); }

  function el(tag, attrs, parent) {
    var node = document.createElementNS(NS, tag);
    for (var k in attrs) { if (Object.prototype.hasOwnProperty.call(attrs, k)) node.setAttribute(k, attrs[k]); }
    (parent || svg).appendChild(node);
    return node;
  }

  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  var clip = el("clipPath", { id: "demo-clip" });
  el("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b }, clip);

  // lưới và nhãn trục
  var t, step;
  for (t = Math.ceil(xmin / 10) * 10; t <= xmax; t += 10) {
    el("line", { x1: X(t), x2: X(t), y1: M.t, y2: H - M.b, "class": "grid" });
    var tx = el("text", { x: X(t), y: H - M.b + 16, "text-anchor": "middle", "class": "tick" });
    tx.textContent = t;
  }
  step = ymax > 10 ? 2 : 1;
  for (t = 0; t <= ymax; t += step) {
    el("line", { x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t), "class": "grid" });
    var ty = el("text", { x: M.l - 8, y: Y(t) + 4, "text-anchor": "end", "class": "tick" });
    ty.textContent = t;
  }
  el("line", { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, "class": "ax" });
  el("line", { x1: M.l, x2: M.l, y1: M.t, y2: H - M.b, "class": "ax" });
  var xt = el("text", { x: (M.l + W - M.r) / 2, y: H - 8, "text-anchor": "middle", "class": "axis-title" });
  xt.textContent = "Diện tích x (m²)";
  var yt = el("text", { x: 14, y: (M.t + H - M.b) / 2, "text-anchor": "middle", "class": "axis-title",
                        transform: "rotate(-90 14 " + ((M.t + H - M.b) / 2) + ")" });
  yt.textContent = "Giá y (tỷ đồng)";

  // các phần thay đổi khi kéo: sai số, đường tối ưu, đường của người dùng, rồi mới tới chấm
  var gRes = el("g", { "clip-path": "url(#demo-clip)" });
  var optLine = el("line", { "class": "fit-opt", "clip-path": "url(#demo-clip)", visibility: "hidden" });
  var fitLine = el("line", { "class": "fit", "clip-path": "url(#demo-clip)" });
  var resLines = [], k;
  for (k = 0; k < n; k++) {
    resLines.push(el("line", { "class": "res" }, gRes));
  }
  for (k = 0; k < n; k++) {
    el("circle", { cx: X(pts[k][0]), cy: Y(pts[k][1]), r: 4.2, "class": "dot" });
  }

  // ---------- điều khiển ----------
  var s1 = document.getElementById("demo-w1");
  var s0 = document.getElementById("demo-w0");
  var o1 = document.getElementById("demo-w1-out");
  var o0 = document.getElementById("demo-w0-out");
  var outJ = document.getElementById("demo-j");
  var outRoot = document.getElementById("demo-root");
  var outPred = document.getElementById("demo-pred");
  var outBest = document.getElementById("demo-jbest");
  var msg = document.getElementById("demo-msg");
  var chkRes = document.getElementById("demo-show-res");
  var chkOpt = document.getElementById("demo-show-opt");

  s1.min = 0; s1.max = Math.max(0.15, best.w1 * 2.2).toFixed(3); s1.step = "any";
  s0.min = 0; s0.max = 10; s0.step = "any";
  outBest.textContent = fmt(bestJ, 2);

  var START = { w0: 2, w1: 0.01 };

  function draw() {
    var w0 = parseFloat(s0.value), w1 = parseFloat(s1.value);
    var j = mse(w0, w1);
    var k2, yhat;

    fitLine.setAttribute("x1", X(xmin)); fitLine.setAttribute("y1", Y(w0 + w1 * xmin));
    fitLine.setAttribute("x2", X(xmax)); fitLine.setAttribute("y2", Y(w0 + w1 * xmax));
    optLine.setAttribute("x1", X(xmin)); optLine.setAttribute("y1", Y(best.w0 + best.w1 * xmin));
    optLine.setAttribute("x2", X(xmax)); optLine.setAttribute("y2", Y(best.w0 + best.w1 * xmax));
    optLine.setAttribute("visibility", chkOpt.checked ? "visible" : "hidden");

    for (k2 = 0; k2 < n; k2++) {
      yhat = w0 + w1 * pts[k2][0];
      resLines[k2].setAttribute("x1", X(pts[k2][0])); resLines[k2].setAttribute("x2", X(pts[k2][0]));
      resLines[k2].setAttribute("y1", Y(pts[k2][1])); resLines[k2].setAttribute("y2", Y(yhat));
    }
    gRes.setAttribute("visibility", chkRes.checked ? "visible" : "hidden");

    o1.textContent = fmt(w1, 4);
    o0.textContent = fmt(w0, 2);
    outJ.textContent = fmt(j, 2);
    outRoot.textContent = fmt(Math.sqrt(j), 2);
    outPred.textContent = fmt(w0 + w1 * 60, 2);

    var gap = (j - bestJ) / bestJ * 100;
    if (gap < 0.5) {
      msg.textContent = "Đây là đường thẳng làm J(w) nhỏ nhất. Linear Regression sẽ tìm ra đúng bộ w này.";
    } else {
      msg.textContent = "J(w) đang lớn hơn mức nhỏ nhất khoảng " + fmt(gap, 0) + "%. Thử kéo tiếp để J(w) giảm xuống.";
    }
  }

  function setW(w0, w1) { s0.value = w0; s1.value = w1; draw(); }

  s0.addEventListener("input", draw);
  s1.addEventListener("input", draw);
  chkRes.addEventListener("change", draw);
  chkOpt.addEventListener("change", draw);
  document.getElementById("demo-best").addEventListener("click", function () { setW(best.w0, best.w1); });
  document.getElementById("demo-reset").addEventListener("click", function () { setW(START.w0, START.w1); });

  setW(START.w0, START.w1);
})();
