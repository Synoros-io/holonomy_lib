import {VIEW, fmt, node, color} from './view.js';

const SVG_NAMESPACE = 'http://www.w3.org/2000/svg'; // W3C SVG namespace, not a network request.
const CHART = Object.freeze({tickCount: 4, pointRadius: 3, arrowSize: 4, labelBaseline: 14});
function svgNode(tag, attributes = {}, text = null) {
  const element = document.createElementNS(SVG_NAMESPACE, tag);
  for (const [key,value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text != null) element.textContent = text;
  return element;
}
function chart(title, height = VIEW.chartHeight) {
  const wrapper = node('div', null, {class:'chart'});
  const svg = svgNode('svg', {viewBox:`0 0 ${VIEW.chartWidth} ${height}`, role:'img', 'aria-label':title});
  wrapper.append(node('h2',title),svg); return [wrapper,svg];
}
function line(svg,x1,y1,x2,y2,cls = 'axis') { svg.append(svgNode('line',{x1,y1,x2,y2,class:cls})); }
function text(svg,x,y,value,anchor = 'middle') { svg.append(svgNode('text',{x,y,'text-anchor':anchor},value)); }
function axis(svg, xmax, ymax, height = VIEW.chartHeight) {
  const p = VIEW.chartPadding, w = VIEW.chartWidth;
  const x = value => p+value/xmax*(w-2*p), y = value => height-p-value/ymax*(height-2*p);
  line(svg,p,p,p,height-p); line(svg,p,height-p,w-p,height-p);
  for (let i=0;i<=CHART.tickCount;i++) {
    const xv = xmax*i/CHART.tickCount, yv = ymax*i/CHART.tickCount;
    text(svg,x(xv),height-p+CHART.labelBaseline,fmt(xv)); text(svg,p-6,y(yv)+3,fmt(yv),'end');
  }
  return [x,y];
}
export function spectrumChart(data) {
  const [wrapper,svg] = chart(`Normalized Laplacian spectrum · λ₂ = ${fmt(data.spectral_gap)}`, VIEW.chartHeight/2);
  const p = VIEW.chartPadding, width = VIEW.chartWidth-2*p, y = VIEW.chartHeight/4;
  const max = Math.max(2,...data.spectrum);
  line(svg,p,y,p+width,y);
  data.spectrum.forEach(value => {
    const x = p+value/max*width;
    line(svg,x,y-10,x,y+10,'plot-line');
  });
  [0,1,2].forEach(value => text(svg,p+value/max*width,y+28,String(value)));
  return wrapper;
}
export function topologyCharts(data, epsilon) {
  const result = document.createDocumentFragment();
  const flattened = data.bars.flatMap((bars,dim) => bars.map(bar => ({bar,dim})));
  const height = Math.max(VIEW.chartHeight,Math.min(VIEW.maxBarcodeHeight,flattened.length*VIEW.barcodeRow+2*VIEW.chartPadding));
  const [barcode,svg] = chart('Persistence barcode · [birth, death) · → survives to infinity',height);
  const max = data.events.at(-1) || 1, p = VIEW.chartPadding;
  const x = value => p+value/max*(VIEW.chartWidth-2*p);
  const row = (height-2*p)/Math.max(1,flattened.length);
  flattened.forEach(({bar:[birth,death],dim},index) => {
    const y = p+(index+0.5)*row;
    const active = birth <= epsilon && (death == null || epsilon < death);
    line(svg,x(birth),y,x(death ?? max),y,`bar ${dim ? 'h1' : ''} ${active ? 'active' : ''}`);
    if (death == null) {
      const tip = x(max), a = CHART.arrowSize;
      svg.append(svgNode('path',{d:`M${tip-a},${y-a} L${tip},${y} L${tip-a},${y+a}`,fill:'none',stroke:color(dim ? 'positive':'accent')}));
    }
  });
  line(svg,x(epsilon),p,x(epsilon),height-p,'plot-line');
  [0,max/2,max].forEach(value => text(svg,x(value),height-p+CHART.labelBaseline,fmt(value)));
  const [diagram,plot] = chart('Persistence diagram · x: birth ε · y: death ε · top row: ∞');
  const [px,py] = axis(plot,max,max);
  line(plot,px(0),py(0),px(max),py(max));
  flattened.forEach(({bar:[birth,death],dim}) => {
    const y = death == null ? p/2 : py(death);
    plot.append(svgNode('circle',{cx:px(birth),cy:y,r:CHART.pointRadius,class:dim?'h1-point':'point'}));
  });
  text(plot,VIEW.chartWidth-p,p/2-5,'∞');
  result.append(barcode,diagram); return result;
}
function logChart(rows, perItem) {
  const [wrapper,svg] = chart(`${perItem ? 'Time per item' : 'Time per batch'} (ms) · log axes · min / median / max`);
  const p = VIEW.chartPadding, width = VIEW.chartWidth, height = VIEW.chartHeight;
  const lower = row => perItem ? row.per_item_min_ms : row.min_ms;
  const middle = row => perItem ? row.per_item_ms : row.median_ms;
  const upper = row => perItem ? row.per_item_max_ms : row.max_ms;
  const xs = [...new Set(rows.map(row => row.batch))].sort((a,b)=>a-b);
  let ymin = Math.log10(Math.min(...rows.map(lower))), ymax = Math.log10(Math.max(...rows.map(upper)));
  if (ymin === ymax) { ymin -= 0.5; ymax += 0.5; }
  const xmin = Math.log10(xs[0]), xmax = Math.log10(xs.at(-1));
  const x = value => p+(Math.log10(value)-xmin)/(xmax-xmin || 1)*(width-2*p);
  const y = value => height-p-(Math.log10(value)-ymin)/(ymax-ymin)*(height-2*p);
  line(svg,p,p,p,height-p); line(svg,p,height-p,width-p,height-p);
  xs.forEach(value => text(svg,x(value),height-p+CHART.labelBaseline,String(value)));
  for(let i=0;i<=CHART.tickCount;i++) {
    const value = 10**(ymin+(ymax-ymin)*i/CHART.tickCount);
    text(svg,p-5,y(value)+3,Number(value.toPrecision(2)).toString(),'end');
  }
  for (const device of new Set(rows.map(row => row.device))) {
    const series = rows.filter(row=>row.device===device), cls = device==='cpu'?'plot-line':'plot-line secondary';
    svg.append(svgNode('polyline',{points:series.map(row=>`${x(row.batch)},${y(middle(row))}`).join(' '),class:cls}));
    series.forEach(row => {
      const cx = x(row.batch); line(svg,cx,y(lower(row)),cx,y(upper(row)),cls);
      svg.append(svgNode('circle',{cx,cy:y(middle(row)),r:CHART.pointRadius,fill:color(device==='cpu'?'accent':'positive')}));
    });
  }
  text(svg,width/2,height-4,'Batch size B'); return wrapper;
}
export function batchCharts(data) {
  const fragment = document.createDocumentFragment();
  fragment.append(logChart(data.rows,false),logChart(data.rows,true)); return fragment;
}
