/*
 * house.js - vẽ cảnh phố nhà ống bằng SVG.
 * Dùng chung cho trang nhập (ngôi nhà đổi theo số tầng, mặt tiền) và trang kết quả.
 *
 *   House.drawScene(svgElement, {
 *     width: 560, height: 360,         // kích thước viewBox
 *     items: [ {floors, width, palette}, {live: true}, ... ],  // nhà từ trái sang phải
 *     live: {floors: 3, frontage: 4.5},  // thông tin căn nhà của người dùng
 *     label: "Nhà của bạn: 3 tầng",      // chữ phía trên căn nhà (không bắt buộc)
 *     animate: true                      // hiệu ứng xây từng tầng
 *   });
 */
(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var FLOOR_H = 30;
  var ROOF_H = 16;
  var GAP = 9;
  var GLASS = "#BFE0E3";

  var PALETTES = [
    { wall: "#F2B134", trim: "#7D5200", roof: "#A83F2A" }, // vàng nghệ
    { wall: "#E8708F", trim: "#7A1F3A", roof: "#5B2A3C" }, // hồng
    { wall: "#2D8689", trim: "#0A3F41", roof: "#A83F2A" }, // xanh ngói
    { wall: "#F1EEE4", trim: "#6C766F", roof: "#7A8A8D" }, // trắng
    { wall: "#8DBFA7", trim: "#2F6B58", roof: "#A83F2A" }  // xanh lá
  ];

  function el(tag, attrs, parent) {
    var node = document.createElementNS(NS, tag);
    for (var k in attrs) {
      if (Object.prototype.hasOwnProperty.call(attrs, k)) node.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function clamp(v, lo, hi) {
    return Math.min(Math.max(v, lo), hi);
  }

  // Mặt tiền càng rộng thì nhà càng bè (3 m -> hẹp, 10 m -> rộng)
  function liveWidth(frontage) {
    return isFinite(frontage) && frontage > 0 ? clamp(40 + frontage * 6, 54, 104) : 68;
  }

  function anim(delayMs, on) {
    return on ? "animation:rise .45s ease-out both;animation-delay:" + delayMs + "ms" : "";
  }

  function drawHouse(parent, o) {
    var floors = o.floors, w = o.width, ground = o.ground, p = PALETTES[o.palette % PALETTES.length];
    var g = el("g", { transform: "translate(" + o.x + " 0)" }, parent);
    var step = 70;

    for (var i = 0; i < floors; i++) {
      var y = ground - (i + 1) * FLOOR_H;
      var f = el("g", { style: anim(i * step, o.animate) }, g);
      el("rect", { x: 0, y: y, width: w, height: FLOOR_H, fill: p.wall }, f);
      el("rect", { x: -2, y: y - 1, width: w + 4, height: 3, fill: p.trim }, f);

      if (i === 0) {
        // tầng trệt: cửa cuốn và ô cửa nhỏ
        var dx = w * 0.1, dw = w * 0.5;
        el("rect", { x: dx, y: y + 6, width: dw, height: FLOOR_H - 6, fill: p.trim, opacity: 0.92 }, f);
        for (var k = 0; k < 4; k++) {
          el("line", { x1: dx, x2: dx + dw, y1: y + 11 + k * 5, y2: y + 11 + k * 5,
                       stroke: p.wall, "stroke-width": 1, opacity: 0.45 }, f);
        }
        el("rect", { x: w * 0.7, y: y + 8, width: w * 0.2, height: 12, fill: GLASS }, f);
      } else {
        // các tầng trên: hai cửa sổ và ban công có lan can
        var ww = w * 0.28;
        [w * 0.14, w * 0.58].forEach(function (wx) {
          el("rect", { x: wx, y: y + 5, width: ww, height: 14, fill: GLASS }, f);
          el("line", { x1: wx + ww / 2, x2: wx + ww / 2, y1: y + 5, y2: y + 19, stroke: p.trim, "stroke-width": 1 }, f);
        });
        el("line", { x1: -3, x2: w + 3, y1: y + FLOOR_H - 9, y2: y + FLOOR_H - 9,
                     stroke: p.trim, "stroke-width": 5, "stroke-dasharray": "1.6 4.4" }, f);
        el("rect", { x: -3, y: y + FLOOR_H - 6, width: w + 6, height: 3, fill: p.trim }, f);
      }
    }

    var top = ground - floors * FLOOR_H;
    var roof = el("g", { style: anim(floors * step, o.animate) }, g);
    el("rect", { x: -4, y: top - 5, width: w + 8, height: 5, fill: p.trim }, roof);
    el("polygon", {
      points: (-6) + "," + (top - 5) + " " + (w + 6) + "," + (top - 5) + " " +
              (w - 8) + "," + (top - 5 - ROOF_H) + " 8," + (top - 5 - ROOF_H),
      fill: p.roof
    }, roof);
    return top - 5 - ROOF_H;
  }

  function drawBackdrop(svg, W, ground) {
    // mặt trời
    el("circle", { cx: W * 0.84, cy: 82, r: 38, fill: "#F2B134", opacity: 0.9 }, svg);
    // hai cột điện và dây điện võng xuống
    [18, W - 18].forEach(function (px) {
      el("rect", { x: px - 2.5, y: 104, width: 5, height: ground - 104, class: "scene-line-fill" }, svg);
      el("rect", { x: px - 14, y: 110, width: 28, height: 4, class: "scene-line-fill" }, svg);
    });
    [0, 9, 18].forEach(function (dy) {
      el("path", { d: "M18 " + (116 + dy) + " Q " + (W / 2) + " " + (170 + dy) + " " + (W - 18) + " " + (116 + dy),
                   fill: "none", "stroke-width": 1.4, class: "scene-line" }, svg);
    });
  }

  function drawScene(svg, cfg) {
    var W = cfg.width, H = cfg.height, ground = H - 4;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    drawBackdrop(svg, W, ground);

    var live = cfg.live || {};
    var liveFloors = clamp(Math.round(live.floors) || 3, 1, 10);
    var lw = liveWidth(live.frontage);

    var specs = cfg.items.map(function (it) {
      return it.live
        ? { live: true, floors: liveFloors, width: lw, palette: 0 }
        : { floors: it.floors, width: it.width, palette: it.palette };
    });
    var total = specs.reduce(function (s, it) { return s + it.width; }, 0) + GAP * (specs.length - 1);
    var x = (W - total) / 2;

    specs.forEach(function (s) {
      var top = drawHouse(svg, { x: x, ground: ground, floors: s.floors, width: s.width,
                                 palette: s.palette, animate: cfg.animate && s.live });
      if (s.live && cfg.label) {
        var t = el("text", { x: x + s.width / 2, y: Math.max(top - 10, 16), "text-anchor": "middle", class: "scene-label" }, svg);
        t.textContent = cfg.label;
      }
      x += s.width + GAP;
    });
  }

  window.House = { drawScene: drawScene, PALETTES: PALETTES };
})();
