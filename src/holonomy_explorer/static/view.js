import {LABELS, placeLabels} from './layout.js';

// Drawing constants are screen-space choices; mathematical data comes from Python.
export const VIEW = Object.freeze({
  padding: 38, nodeRadius: 6, graphNodeRadius: 5, hitRadius: 15, labelOffset: 12,
  fontSize: 11, lineWidth: 1.5, edgeWidth: 0.9, gridWidth: 0.7, flowMs: 900,
  chartWidth: 540, chartHeight: 200, chartPadding: 38,
  barcodeRow: 7, barcodePadding: 24, chartInset: 14, diagramHeight: 180, figureExtent: 1.3,
  debounceMs: 180, transitionMs: 500, triangleOpacity: 0.035,
});
export const $ = (id) => document.getElementById(id);
export const fmt = (value) => value == null ? '∞ / unavailable' :
  typeof value === 'number' ? (Number.isInteger(value) ? String(value) : Number(value.toPrecision(5)).toString()) : String(value);
export const color = (name) => getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
export function node(tag, text, attributes = {}) {
  const element = document.createElement(tag);
  if (text != null) element.textContent = text;
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
}
export function table(headers, rows) {
  const table = node('table');
  const head = node('thead'), body = node('tbody'), row = node('tr');
  headers.forEach(value => row.append(node('th', value, {scope: 'col'})));
  head.append(row);
  rows.forEach(values => { const row = node('tr'); values.forEach(value => row.append(node('td', fmt(value)))); body.append(row); });
  table.append(head, body);
  const scroll = node('div', null, {class: 'table-scroll'}); scroll.append(table); return scroll;
}
export function metrics(entries) {
  const result = node('div', null, {class: 'metrics'});
  entries.forEach(([label, value]) => { const item = node('div', null, {class: 'metric'}); item.append(node('strong', fmt(value)), node('span', label)); result.append(item); });
  return result;
}
export function parameters(values) {
  const result = node('dl', null, {class: 'parameters'});
  Object.entries(values).forEach(([key, value]) => {
    if (Array.isArray(value) || value && typeof value === 'object') return;
    result.append(node('dt', key.replaceAll('_', ' ')), node('dd', fmt(value)));
  }); return result;
}
export function section(title, child) {
  const result = node('section'); result.append(node('h2', title), child); return result;
}
export class CanvasView {
  constructor(canvas) { this.canvas = canvas; this.ctx = canvas.getContext('2d'); this.map = p => p; this.inverse = p => p; }
  begin(points, extent = null) {
    const {width, height} = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(width * dpr); this.canvas.height = Math.round(height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.ctx.clearRect(0, 0, width, height);
    let xmin = -1, xmax = 1, ymin = -1, ymax = 1;
    if (extent) [xmin, xmax, ymin, ymax] = extent;
    else if (points.length) {
      xmin = Math.min(...points.map(p => p[0])); xmax = Math.max(...points.map(p => p[0]));
      ymin = Math.min(...points.map(p => p[1])); ymax = Math.max(...points.map(p => p[1]));
    }
    // Spectral-only mode preserves coincident points. Relaxation, when selected,
    // happens separately and never changes the coordinates returned by Python.
    const span = Math.max(xmax - xmin, ymax - ymin) || 1;
    const scale = (Math.min(width, height) - 2 * VIEW.padding) / span;
    const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
    this.map = ([x,y]) => [width / 2 + (x-cx)*scale, height / 2 - (y-cy)*scale];
    this.inverse = ([x,y]) => [(x-width/2)/scale+cx, -(y-height/2)/scale+cy];
    this.ctx.font = `${VIEW.fontSize}px system-ui, sans-serif`;
    this.width=width; this.height=height;
  }
  graphLabels(points, priority) {
    const positions = points.map(point=>this.map(point)), ctx=this.ctx;
    const labels = placeLabels(positions,text=>ctx.measureText(text).width,this.width,this.height,VIEW.graphNodeRadius*1.5,priority);
    for (const label of labels) {
      const [x,y]=positions[label.index];
      ctx.beginPath(); ctx.moveTo(x,y);
      ctx.lineTo(Math.max(label.x,Math.min(x,label.x+label.width)),Math.max(label.y,Math.min(y,label.y+label.height)));
      ctx.strokeStyle=color('edge'); ctx.lineWidth=VIEW.gridWidth; ctx.stroke();
      ctx.fillStyle=color('surface'); ctx.fillRect(label.x,label.y,label.width,label.height);
      ctx.fillStyle=color('ink'); ctx.textBaseline='top';
      ctx.fillText(label.text,label.x+LABELS.padding,label.y+LABELS.padding);
    }
    ctx.textBaseline='alphabetic';
  }
  path(points, stroke = color('edge'), width = VIEW.lineWidth, alpha = 1, fill = false) {
    const ctx = this.ctx; ctx.beginPath(); let begun = false;
    for (const point of points) {
      if (!point || point.some(v => v == null || !Number.isFinite(v))) { begun = false; continue; }
      const [x,y] = this.map(point);
      if (begun) ctx.lineTo(x,y); else ctx.moveTo(x,y);
      begun = true;
    }
    ctx.globalAlpha = alpha; ctx.strokeStyle = stroke; ctx.lineWidth = width;
    if (fill) { ctx.closePath(); ctx.fillStyle = stroke; ctx.fill(); } else ctx.stroke();
    ctx.globalAlpha = 1;
  }
  dot(point, fill = color('node'), label = null, hollow = false, radius = VIEW.nodeRadius) {
    if (!point || point.some(v => v == null || !Number.isFinite(v))) return;
    const [x,y] = this.map(point), ctx = this.ctx;
    ctx.beginPath(); ctx.arc(x,y,radius,0,2*Math.PI);
    ctx.fillStyle = hollow ? color('surface') : fill; ctx.fill();
    ctx.strokeStyle = fill; ctx.lineWidth = VIEW.lineWidth; ctx.stroke();
    if (label != null) { ctx.fillStyle = color('muted'); ctx.fillText(String(label), x+VIEW.labelOffset, y-VIEW.labelOffset/2); }
  }
  circle(radius) {
    const [cx,cy] = this.map([0,0]), [ex] = this.map([radius,0]);
    this.ctx.beginPath(); this.ctx.arc(cx,cy,Math.abs(ex-cx),0,2*Math.PI);
    this.ctx.strokeStyle = color('muted'); this.ctx.setLineDash([VIEW.gridWidth*4,VIEW.gridWidth*4]); this.ctx.stroke(); this.ctx.setLineDash([]);
  }
  click(event) { const rect = this.canvas.getBoundingClientRect(); return [event.clientX-rect.left,event.clientY-rect.top]; }
  nearest(event, points) {
    const [x,y] = this.click(event); let best = null, distance = VIEW.hitRadius;
    points.forEach((point,index) => { const [px,py] = this.map(point), d = Math.hypot(px-x,py-y); if (d < distance) { best = index; distance = d; } });
    return best;
  }
}
// Color interpolation is presentation only. Values and their ranges come from the API.
export function blend(a,b,t) {
  const rgb = value => value.slice(1).match(/.{2}/g).map(v => parseInt(v,16));
  const left = rgb(a), right = rgb(b);
  return `rgb(${left.map((v,i) => Math.round(v+(right[i]-v)*Math.max(0,Math.min(1,t)))).join(',')})`;
}
export function diverging(value, range) {
  const span = Math.max(Math.abs(range[0]),Math.abs(range[1])) || 1;
  return blend(color('neutral'), color(value < 0 ? 'negative' : 'positive'), Math.abs(value)/span);
}
export function curvatureScale(range) {
  if(range[0]===0 && range[1]===0) return {range,kind:'zero',note:'zero curvature'};
  if (range[0]<0 && range[1]>0) {
    const extent=Math.max(Math.abs(range[0]),Math.abs(range[1]));
    return {range:[-extent,extent],kind:'diverging',note:'diverging, centered at 0'};
  }
  return {range,kind:range[1]<=0?'negative':'positive',note:'sequential'};
}
export function curvatureColor(value, scale) {
  if (value==null) return color('muted');
  if (value===0) return color('neutral');
  if (scale.kind==='diverging') return diverging(value,scale.range);
  const [low,high]=scale.range;
  const fraction=high===low?1:scale.kind==='negative'?(high-value)/(high-low):(value-low)/(high-low);
  return blend(color('neutral'),color(scale.kind),fraction);
}
export function sequential(value, range) {
  if (value == null) return color('muted');
  return blend(color('surface'),color('accent'), (value-range[0])/(range[1]-range[0] || 1));
}
export function community(index) {
  // Separate components need categorical hues; the rest of the UI uses one accent.
  const goldenAngle = 180 * (3-Math.sqrt(5));
  return `hsl(${index*goldenAngle%360} 34% 53%)`;
}
