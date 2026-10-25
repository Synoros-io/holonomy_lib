import {VIEW, $, fmt, node, table, metrics, parameters, section, CanvasView, color, diverging, sequential, community} from './view.js';
import {spectrumChart, topologyCharts, batchCharts} from './charts.js';

const SCENES = {
  graphs: {title:'Graphs', description:'Inspect local curvature, diffusion and connectivity on a real graph. Positions are the two non-trivial Laplacian eigenvectors.', view:'Spectral embedding', hint:'Click a node to choose the source'},
  curvature: {title:'Curvature', description:'Move continuously between hyperbolic, flat and spherical charts. All geodesics and distances come from the κ-stereographic manifold.', view:'κ-stereographic chart', hint:'Click to add points; click points to select a pair'},
  topology: {title:'Topology', description:'A point cloud, its Vietoris–Rips filtration, and the persistence of components and loops. ε is the maximum edge length, in coordinate units.', view:'Vietoris–Rips complex', hint:'Click to add a point'},
  batching: {title:'Batching', description:'Measure the library at different batch sizes, with a warm-up before each shape. Persistence reduction is sequential per batch member and uses CPU bookkeeping, so it is expected to scale differently.', view:'Synchronized wall-time measurements', hint:'CPU and available GPU'},
};
let config, active = 'graphs', revision = 0, pending = false, busy = false, debounce, playTimer;
const state = Object.fromEntries(Object.keys(SCENES).map(key => [key,{params:{},data:null,points:[],selected:[],event:0,epsilon:0,frame:0,transition:1}]));
const view = new CanvasView($('canvas'));

async function api(name, request) {
  const response = await fetch(`/api/${name}`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(request)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}
function error(message = '') { $('error').textContent = message; $('error').hidden = !message; }
function control(label, input) { const wrapper = node('label'); wrapper.append(node('span',label),input); return wrapper; }
function select(key,label,options,value) {
  const input = node('select',null,{name:key});
  options.forEach(option => { const [value,text] = Array.isArray(option)?option:[option,option]; input.append(node('option',text,{value})); });
  input.value = value; return control(label,input);
}
function numeric(key,label,value,min,max,step = 1) {
  return control(label,node('input',null,{name:key,type:'number',value,min,max,step}));
}
function range(key,label,value,min,max,step) {
  const wrapper = control(label,node('input',null,{name:key,type:'range',value,min,max,step}));
  const text = wrapper.firstChild; text.className='range-label'; text.append(node('output',fmt(value))); return wrapper;
}
function group(...children) { const result = node('div',null,{class:'control-grid'}); result.append(...children); return result; }
function help(text) { return node('p',text,{class:'help'}); }
function action(label,callback) { const button = node('button',label,{type:'button'}); button.addEventListener('click',callback); return button; }
function coordinatesControls() {
  const row = group(numeric('_x','x',0,-Number.MAX_VALUE,Number.MAX_VALUE,'any'),numeric('_y','y',0,-Number.MAX_VALUE,Number.MAX_VALUE,'any'));
  const buttons = node('div',null,{class:'button-row'});
  buttons.append(action('Add point',() => {
    const form = $('controls');
    addPoint([Number(form.elements.namedItem('_x').value),Number(form.elements.namedItem('_y').value)]);
  }),action('Clear',() => {
    const current=state[active]; current.points=[]; current.selected=[];
    if(active==='topology') current.params.cloud='custom';
    if(active==='curvature') Object.assign(current.params,{connections:'all',pair_a:0,pair_b:1});
    buildControls(); schedule();
  }));
  return [row,buttons];
}
function buildControls() {
  const form=$('controls'), current=state[active], p=current.params, defaults=config.defaults, limits=config.limits;
  form.replaceChildren();
  const append=(...items)=>form.append(...items);
  if(active==='graphs') {
    append(select('graph','Graph',[
      ['karate',"Zachary’s karate club"],['sbm','Stochastic block model'],['barbell','Barbell'],
      ['cliques','Ring of cliques'],['geometric','Random geometric'],['custom','Paste edge list'],
    ],p.graph));
    if(['sbm','geometric','custom'].includes(p.graph)) append(numeric('nodes','Nodes',p.nodes,1,limits.nodes));
    if(p.graph==='sbm') append(numeric('blocks','Blocks',p.blocks,1,p.nodes),group(numeric('p_in','p within',p.p_in,0,1,'any'),numeric('p_out','p between',p.p_out,0,1,'any')));
    if(['barbell','cliques'].includes(p.graph)) append(numeric('clique_size','Nodes per clique',p.clique_size,2,Math.floor(limits.nodes/2)));
    if(p.graph==='cliques') append(numeric('cliques','Cliques',p.cliques,2,Math.floor(limits.nodes/2)));
    if(p.graph==='geometric') append(numeric('radius','Connection radius',p.radius,0,Math.SQRT2,'any'));
    if(p.graph==='custom') { const input=node('textarea',p.edges || '',{name:'edges',placeholder:'0 1\n1 2\n2 0 0.5','aria-label':'Edge list'}); append(control('i j [weight], one edge per line',input),help('Zero-based IDs. Missing IDs are isolated vertices. Self-loops are dropped.')); }
    if(['sbm','geometric'].includes(p.graph)) append(numeric('seed','Seed',p.seed,0,config.max_seed));
    append(select('overlay','Overlay',[
      ['none','Plain graph'],['forman','Forman curvature'],['ollivier','Ollivier–Ricci curvature'],['flow','Ricci flow + surgery'],
      ['heat','Heat diffusion'],['resistance','Effective resistance'],['spectrum','Laplacian spectrum'],
    ],p.overlay));
    if(['heat','resistance'].includes(p.overlay)) append(numeric('source','Source node',p.source,0,(current.data?.nodes || limits.nodes)-1));
    if(p.overlay==='heat') append(range('_log_time','Diffusion time log₁₀ t',Math.log10(p.time),Math.log10(config.heat_time_range[0]),Math.log10(config.heat_time_range[1]),'any'),help(`t = ${fmt(p.time)} · logarithmic slider`));
    if(p.overlay==='flow') append(
      group(numeric('steps','Flow steps',p.steps,1,limits.flow_steps),numeric('surgery_period','Surgery period',p.surgery_period,1,p.steps)),
      numeric('surgery_threshold','Surgery threshold × initial mean',p.surgery_threshold,Number.EPSILON,config.max_surgery_threshold,'any'),
      numeric('dt','Step size dt',p.dt,Number.EPSILON,1,'any'),
      help(`At most ${limits.flow_frames} frames. Each frame runs from the original adjacency.`),
    );
  } else if(active==='curvature') {
    append(range('kappa','Curvature κ',p.kappa,-2,2,'any'),help('κ < 0: open ball. κ > 0: conservative hemisphere. Hollow points are excluded.'),...coordinatesControls());
    if(current.points.length>1) {
      const options=current.points.map((_,i)=>[String(i),`Point ${i}`]);
      append(select('connections','Geodesics',[['all','All pairs'],['selected','Selected pair']],p.connections));
      if(p.connections==='selected') append(group(select('pair_a','From',options,String(p.pair_a)),select('pair_b','To',options,String(p.pair_b))));
    }
    append(help(`${current.points.length} / ${limits.curvature_points} points. Distances use the library’s conformal normalization.`));
  } else if(active==='topology') {
    append(select('cloud','Point cloud',[
      ['circle','Noisy circle'],['two_circles','Two circles'],['figure_eight','Figure eight'],['annulus','Annulus'],['clusters','Gaussian clusters'],['custom','Draw / edit points'],
    ],p.cloud));
    if(p.cloud!=='custom') append(group(numeric('points_count','Points',p.points_count,1,limits.points),numeric('seed','Seed',p.seed,0,config.max_seed)),numeric('noise','Noise σ (coordinate units)',p.noise,0,1,'any'));
    append(...coordinatesControls());
    if(current.data) append(range('_epsilon','Radius ε',current.epsilon,0,current.data.events.at(-1)||1,'any'),help('Bars and counts are precomputed once. Moving ε only changes the view.'));
    append(help(`At most ${limits.points} points; H₀ = components, H₁ = loops.`));
  } else {
    append(select('primitive','Primitive',[['heat','Heat kernel on random graphs'],['distance','Stereographic pairwise distances'],['persistence','Persistence diagrams (H₀, H₁)']],p.primitive));
    const persistence=p.primitive==='persistence';
    append(numeric('size',persistence?'Points per cloud':'Nodes / points per item',p.size,1,persistence?limits.persistence_size:limits.nodes),
      group(numeric('repeats','Measured repeats',p.repeats,1,limits.repeats),numeric('seed','Seed',p.seed,0,config.max_seed)),
      help(`Batch sizes: ${defaults.batches.filter(b=>b<=(persistence?limits.persistence_batch:limits.batch)).join(', ')}. One warm-up per shape. GPU is compared when available.`));
  }
  const run=node('button',active==='batching'?'Run sweep':'Compute',{type:'submit',class:'primary',id:'run'});
  append(run,node('span','',{class:'busy-note',id:'busy-note'})); setBusy(busy);
}
function readControls() {
  const current=state[active];
  for(const input of $('controls').elements) {
    if(!input.name || input.name.startsWith('_')) continue;
    current.params[input.name]=(input.type==='number'||input.type==='range')?Number(input.value):input.value;
  }
}
function requestBody(scene) {
  const current=state[scene], request={...current.params,device:$('device').value,dtype:$('dtype').value};
  if(scene==='curvature') {
    request.points=current.points;
    if(request.connections==='selected' && current.points.length>1) request.pairs=[[Number(request.pair_a),Number(request.pair_b)]];
    delete request.connections; delete request.pair_a; delete request.pair_b;
  }
  if(scene==='topology' && request.cloud==='custom') request.points=current.points;
  return request;
}
function setBusy(value) {
  if($('run')) { $('run').disabled=value; $('run').textContent=value?'Computing…':active==='batching'?'Run sweep':'Compute'; }
  if($('busy-note')) $('busy-note').textContent=value?'Running the library on the selected device':'';
  $('status-state').textContent=value?'Computing…':'Ready'; $('controls').setAttribute('aria-busy',String(value));
}
function schedule() { revision++; pending=true; void drain(); }
async function drain() {
  // Serialize compute requests, and never replace newer controls with a stale
  // response. Aborting a fetch would not cancel the underlying torch operation.
  if(busy) return;
  busy=true; setBusy(true);
  while(pending) {
    pending=false;
    const scene=active, version=revision;
    try {
      error();
      const response=await api(scene,requestBody(scene));
      if(version!==revision) continue;
      const current=state[scene]; current.data=response;
      if(scene==='topology') { current.points=response.points; current.event=0; current.epsilon=0; }
      if(scene==='graphs') { current.frame=0; current.transition=1; }
      buildControls(); render();
      $('status-device').textContent=`${response.device} · ${response.dtype}`;
      $('status-time').textContent=`Library calls: ${fmt(response.elapsed_ms)} ms`;
    } catch(reason) { if(version===revision) { error(reason.message); $('empty').textContent='Computation failed. Adjust the controls and try again.'; } }
  }
  busy=false; setBusy(false);
}
function activate(scene) {
  clearTimeout(debounce);
  clearInterval(playTimer); playTimer=null;
  active=scene;
  document.querySelectorAll('[data-scene]').forEach(button=>{
    if(button.dataset.scene===scene) button.setAttribute('aria-current','page'); else button.removeAttribute('aria-current');
  });
  const meta=SCENES[scene]; $('title').textContent=meta.title; $('description').textContent=meta.description;
  $('view-label').textContent=meta.view; $('hint').textContent=meta.hint;
  buildControls(); render(); schedule();
}
function legendText(text) { $('legend').replaceChildren(node('span',text)); }
function legendRange(label,range,sequentialColor=false) {
  $('legend').replaceChildren(node('span',label),node('span',fmt(range[0])),node('span',null,{class:`gradient ${sequentialColor?'sequential':''}`}),node('span',fmt(range[1])));
}
function drawGraph(data,current) {
  view.begin(data.coordinates);
  const frame=data.frames?.[current.frame], previous=data.frames?.[Math.max(0,current.frame-1)];
  data.edges.forEach(([a,b],index)=>{
    let alpha=1, stroke=color('edge');
    if(data.edge_values) stroke=data.edge_values[index]==null?color('muted'):diverging(data.edge_values[index],data.range);
    if(frame) {
      const was=previous[a][b]>0, now=frame[a][b]>0;
      alpha=now?1:was?1-current.transition:0;
    }
    view.path([data.coordinates[a],data.coordinates[b]],stroke,VIEW.lineWidth,alpha);
  });
  const finite=data.node_values?.filter(Number.isFinite) || [], range=finite.length?[Math.min(...finite),Math.max(...finite)]:[0,0];
  data.coordinates.forEach((point,index)=>{
    const fill=data.node_values?sequential(data.node_values[index],range):frame && current.frame===data.frames.length-1?community(data.final_labels[index]):color('node');
    view.dot(point,fill,index,data.node_values && data.node_values[index]==null,index===data.source && ['heat','resistance'].includes(data.overlay)?VIEW.nodeRadius*1.5:VIEW.nodeRadius);
  });
  if(data.edge_values) legendRange('Edge curvature (dimensionless)',data.range);
  else if(data.node_values) { legendRange(data.value_label,range,true); if(data.node_values.includes(null)) $('legend').append(node('span','Hollow = ∞ / unavailable')); }
  else if(frame) legendText(`Step ${data.frame_steps[current.frame]} · final colors identify connected components`);
  else legendText('Unweighted node markers · edge weights in the details panel');
}
function graphDetails(data,current) {
  clearInterval(playTimer); playTimer=null;
  const details=$('details'); details.replaceChildren(metrics([['nodes',data.nodes],['edges',data.edges.length]]));
  details.append(section('Connectivity',node('p',`Component sizes: ${data.component_sizes.join(', ')}. λ₂ = ${fmt(data.spectral_gap)}.`)));
  if(data.frames) {
    details.append(section('Final flow components',node('p',data.final_sizes.join(', '))),node('p',data.flow_method));
    if(data.faction_note) details.append(section('Observed karate split',node('p',`Wrong-side nodes: ${data.faction_wrong_side??'not a two-way partition'}. ${data.faction_note}`)));
  }
  if(data.node_values) details.append(section('Values by node',table(['Node','Value'],data.node_values.map((value,index)=>[index,value]))));
  else details.append(section('Values by edge',table(['Edge','Weight',...(data.edge_values?['κ']:[])],data.edges.map(([a,b],index)=>[`${a} ↔ ${b}`,data.weights[index],...(data.edge_values?[data.edge_values[index]]:[])]))));
  if(data.reference_check) details.append(section('Reference check',node('p',`Triangle edge / bridge expected: ${data.reference_check.expected.map(fmt).join(', ')}. Observed: ${data.reference_check.observed.map(fmt).join(', ')}.`)));
  details.append(section('Parameters',parameters(data.parameters)));
  if(data.parameters.citation) details.append(node('p',data.parameters.citation));
  $('charts').replaceChildren(spectrumChart(data));
  const playback=$('playback'); playback.hidden=!data.frames; playback.replaceChildren();
  if(data.frames) {
    const play=action(playTimer?'Pause':'Play',()=>{
      if(playTimer) { clearInterval(playTimer); playTimer=null; play.textContent='Play'; return; }
      play.textContent='Pause';
      if(current.frame===data.frames.length-1) { current.frame=0; current.transition=1; }
      playTimer=setInterval(()=>{
        if(active!=='graphs'||current.frame>=data.frames.length-1) { clearInterval(playTimer); playTimer=null; play.textContent='Play'; return; }
        current.frame++; slider.value=current.frame; animateFlow(data,current);
      },VIEW.flowMs);
    });
    const slider=node('input',null,{type:'range',min:0,max:data.frames.length-1,value:current.frame,step:1,'aria-label':'Flow frame'});
    slider.addEventListener('input',()=>{current.frame=Number(slider.value);current.transition=1;drawGraph(data,current);});
    playback.append(play,slider,node('span',`${data.frames.length} frames`));
  }
}
function animateFlow(data,current) {
  const start=performance.now(); current.transition=0;
  const frame=()=>{
    if(active!=='graphs'||state.graphs.data!==data) return;
    current.transition=Math.min(1,(performance.now()-start)/VIEW.transitionMs); drawGraph(data,current);
    if(current.transition<1) requestAnimationFrame(frame);
  }; requestAnimationFrame(frame);
}
function drawCurvature(data,current) {
  const extent=Math.max(VIEW.figureExtent,...data.points.flat().map(Math.abs));
  view.begin(data.points,[-extent,extent,-extent,extent]);
  data.grid.forEach(curve=>view.path(curve,color('grid'),VIEW.gridWidth));
  if(data.boundary_radius) view.circle(data.boundary_radius);
  data.curves.forEach(curve=>view.path(curve,color('accent')));
  data.points.forEach((point,index)=>view.dot(point,current.selected.includes(index)?color('positive'):color('node'),index,!data.valid[index]));
  if(data.mean) view.dot(data.mean.point,color('negative'),'μ',!data.mean.converged);
  legendText(`${data.chart_note}${data.boundary_radius?` · boundary radius ${fmt(data.boundary_radius)}`:''}`);
}
function curvatureDetails(data) {
  const details=$('details'); details.replaceChildren(metrics([['valid points',data.included_indices.length],['κ',data.parameters.kappa]]));
  if(data.valid.some(value=>!value)) details.append(node('p','Hollow points lie outside the current domain and are excluded.'));
  details.append(section('Pairwise geodesic distances',table(['Point',...data.included_indices],data.distances.map((row,index)=>[data.included_indices[index],...row]))));
  details.append(section('Coordinates',table(['Point','x','y'],data.points.map((point,index)=>[index,...point]))));
  if(data.mean) details.append(section('Fréchet estimate',node('p',`μ = (${data.mean.point.map(fmt).join(', ')}). Riemannian gradient residual: ${fmt(data.mean.residual)}. ${data.mean.converged?'Within dtype-based tolerance.':'Not converged.'}`)));
  if(data.parameters.kappa>0) details.append(node('p','Mean is omitted for positive curvature.'));
  details.append(section('Parameters',parameters(data.parameters)));
  $('charts').replaceChildren(); $('playback').hidden=true;
}
function drawTopology(data,current) {
  view.begin(data.points,data.points.length?null:[-VIEW.figureExtent,VIEW.figureExtent,-VIEW.figureExtent,VIEW.figureExtent]);
  const epsilon=current.epsilon;
  data.triangles.forEach((triangle,index)=>{ if(data.triangle_radii[index]<=epsilon) view.path(triangle.map(i=>data.points[i]),color('triangle'),VIEW.gridWidth,VIEW.triangleOpacity,true); });
  data.edges.forEach(([a,b],index)=>{ if(data.edge_radii[index]<=epsilon) view.path([data.points[a],data.points[b]],color('edge')); });
  data.points.forEach((point,index)=>view.dot(point,color('node'),index));
  legendText(`ε = ${fmt(epsilon)} · H₀: green · H₁: blue · filled triangles at diameter ≤ ε`);
}
function topologyDetails(data,current) {
  const index=current.event, betti=data.betti[index], epsilon=current.epsilon;
  const details=$('details'); details.replaceChildren(metrics([['β₀ components',betti[0]],['β₁ loops',betti[1]]]));
  details.append(section('Filtration at ε',parameters({epsilon,points:data.points.length,edges:data.edge_counts[index],triangles:data.triangle_counts[index]})),
    node('p','Intervals include birth and exclude death. Infinite bars end in an arrow. Zero-length bars never contribute to Betti counts.'),
    section('Square reference',node('p',data.reference_check.passed?'Passed: one H₁ bar [1, √2).':'Failed: current library emits extra bars. Treat H₁ counts as provisional.')),
    section('Parameters',parameters(data.parameters)));
  $('charts').replaceChildren(topologyCharts(data,epsilon)); $('playback').hidden=true;
}
function batchingDetails(data) {
  $('charts').replaceChildren(batchCharts(data));
  $('details').replaceChildren(section('Measurement',node('p',data.timing_note)),section('Parameters',parameters(data.parameters)),
    section('Devices',node('p',[...new Set(data.rows.map(row=>row.device))].join(', '))),
    node('p','One warm-up for each device and batch shape. Median is plotted with min–max whiskers. Timings include any host work inside the primitive.'),
    node('p','Changing dtype changes the computation; compare rows within a single sweep.'));
  $('data-table').replaceChildren(node('h2','All measurements (milliseconds)'),table(
    ['Device','B','Min / batch','Median / batch','Max / batch','Min / item','Median / item','Max / item','All repeats'],
    data.rows.map(row=>[row.device,row.batch,row.min_ms,row.median_ms,row.max_ms,row.per_item_min_ms,row.per_item_ms,row.per_item_max_ms,row.samples_ms.map(fmt).join(', ')])));
  legendText('CPU: green · GPU: blue · one primitive call processes the full batch'); $('playback').hidden=true;
}
function render() {
  const current=state[active], data=current.data;
  document.querySelector('.visual-panel').classList.toggle('batch-mode',active==='batching');
  $('data-table').hidden=active!=='batching'||!data;
  $('warnings').replaceChildren(); $('empty').hidden=Boolean(data);
  if(!data) { view.begin([]); $('charts').replaceChildren(); $('details').replaceChildren(); $('legend').replaceChildren(); $('playback').hidden=true; $('empty').textContent='Waiting for the library…'; return; }
  data.warnings.forEach(message=>$('warnings').append(node('p',message)));
  if(active==='graphs') { drawGraph(data,current); graphDetails(data,current); }
  else if(active==='curvature') { drawCurvature(data,current); curvatureDetails(data); }
  else if(active==='topology') { drawTopology(data,current); topologyDetails(data,current); }
  else batchingDetails(data);
  const callDetails=node('details');
  callDetails.append(node('summary','Library call timings (ms)'),table(['Function','ms','Arguments'],data.calls.map(call=>[call.function.split('.').slice(-2).join('.'),call.elapsed_ms,Object.entries(call.parameters).map(([key,value])=>`${key}=${fmt(value)}`).join(', ')])));
  $('details').append(callDetails);
}
function addPoint(point) {
  const current=state[active], cap=active==='curvature'?config.limits.curvature_points:config.limits.points;
  if(point.some(value=>!Number.isFinite(value))) { error('Point coordinates must be finite.'); return; }
  if(current.points.length>=cap) { error(`The interactive limit is ${cap} points. Clear the cloud to start again.`); return; }
  current.points=[...current.points,point];
  if(active==='topology') current.params.cloud='custom';
  buildControls(); schedule();
}
$('canvas').addEventListener('click',event=>{
  const current=state[active], data=current.data; if(!data) return;
  if(active==='graphs') {
    const index=view.nearest(event,data.coordinates); if(index==null) return;
    current.params.source=index; if(!['heat','resistance'].includes(current.params.overlay)) current.params.overlay='heat';
    buildControls(); schedule();
  } else if(active==='curvature') {
    const index=view.nearest(event,data.points);
    if(index==null) addPoint(view.inverse(view.click(event)));
    else {
      current.selected=[...current.selected.filter(value=>value!==index),index].slice(-2);
      if(current.selected.length===2) Object.assign(current.params,{connections:'selected',pair_a:current.selected[0],pair_b:current.selected[1]});
      buildControls(); schedule();
    }
  } else if(active==='topology') addPoint(view.inverse(view.click(event)));
});
$('controls').addEventListener('submit',event=>{event.preventDefault();if($('controls').reportValidity()){readControls();schedule();}});
$('controls').addEventListener('input',event=>{
  const input=event.target; if(input.type!=='range') return;
  input.closest('label')?.querySelector('output')?.replaceChildren(fmt(Number(input.value)));
  if(input.name==='_epsilon') {
    const current=state.topology, value=Number(input.value);
    current.epsilon=value;
    // Counts are server-computed at filtration events. Between events only the
    // cursor moves; the browser does not run another persistence computation.
    const next=current.data.events.findIndex(radius=>radius>value); current.event=next<0?current.data.events.length-1:Math.max(0,next-1);
    render(); return;
  }
  if(input.name==='_log_time') state.graphs.params.time=10**Number(input.value); else readControls();
  clearTimeout(debounce); debounce=setTimeout(schedule,VIEW.debounceMs);
});
$('controls').addEventListener('change',event=>{
  const input=event.target; if(!input.name||input.name.startsWith('_')||input.type==='range') return;
  readControls();
  if(active==='graphs' && ['graph','nodes','clique_size','cliques','edges'].includes(input.name)) state.graphs.params.source=0;
  if(active==='batching' && input.name==='primitive' && state.batching.params.primitive==='persistence') state.batching.params.size=Math.min(state.batching.params.size,config.limits.persistence_size);
  if(['graph','overlay','cloud','connections','primitive'].includes(input.name)) buildControls();
  if($('controls').reportValidity()) schedule();
});
document.querySelectorAll('[data-scene]').forEach(button=>button.addEventListener('click',()=>activate(button.dataset.scene)));
[$('device'),$('dtype')].forEach(input=>input.addEventListener('change',schedule));
new ResizeObserver(()=>{if(config) render();}).observe($('canvas'));
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{if(config) render();});

try {
  config=await api('config',{});
  if(config.cuda_available) $('device').append(node('option','CUDA / ROCm',{value:config.device.startsWith('cuda')?config.device:'cuda'}));
  $('device').value=config.device; $('dtype').value=config.dtype;
  const d=config.defaults;
  state.graphs.params={...d,graph:'karate',overlay:'forman',source:0,time:1,edges:''};
  state.curvature.params={kappa:-1,connections:'all',pair_a:0,pair_b:1};
  state.curvature.points=config.curvature_points;
  state.topology.params={cloud:'circle',points_count:d.points_count,noise:d.noise,seed:d.seed};
  state.batching.params={primitive:'heat',size:d.size,repeats:d.repeats,seed:d.seed};
  activate('graphs');
} catch(reason) { error(reason.message); $('status-state').textContent='Connection failed'; }
