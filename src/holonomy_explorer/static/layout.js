// Presentation only. Fruchterman and Reingold (1991), Graph Drawing by
// Force-directed Placement: repulsion k²/d, attraction d²/k, linear cooling.
// The unit-area layout starts at the spectral coordinates. The small index-based
// offset separates coincident eigenvector coordinates without a random seed.
export const RELAXATION = Object.freeze({
  iterations: 180, initialTemperature: 0.08, separationFraction: 0.01,
});

export function relaxLayout(coordinates, edges) {
  const count = coordinates.length;
  if (count < 2) return coordinates.map(point => [...point]);
  const center = [0, 1].map(axis => coordinates.reduce((sum, point) => sum+point[axis], 0)/count);
  const span = Math.max(...[0, 1].map(axis =>
    Math.max(...coordinates.map(point => point[axis]))-Math.min(...coordinates.map(point => point[axis])))) || 1;
  const ideal = Math.sqrt(1/count), separation = ideal*RELAXATION.separationFraction;
  const goldenAngle = Math.PI*(3-Math.sqrt(5));
  const points = coordinates.map((point,index) => [
    (point[0]-center[0])/span + separation*Math.cos(index*goldenAngle),
    (point[1]-center[1])/span + separation*Math.sin(index*goldenAngle),
  ]);
  for (let iteration=0; iteration<RELAXATION.iterations; iteration++) {
    const forces = points.map(() => [0, 0]);
    for (let left=0; left<count; left++) {
      for (let right=left+1; right<count; right++) {
        let dx = points[left][0]-points[right][0], dy = points[left][1]-points[right][1];
        if (dx===0 && dy===0) { dx=separation*Math.cos(right*goldenAngle); dy=separation*Math.sin(right*goldenAngle); }
        const distanceSquared = Math.max(dx*dx+dy*dy, separation*separation);
        const strength = ideal*ideal/distanceSquared;
        forces[left][0] += dx*strength; forces[left][1] += dy*strength;
        forces[right][0] -= dx*strength; forces[right][1] -= dy*strength;
      }
    }
    for (const [left,right] of edges) {
      const dx = points[left][0]-points[right][0], dy = points[left][1]-points[right][1];
      const strength = Math.hypot(dx,dy)/ideal;
      forces[left][0] -= dx*strength; forces[left][1] -= dy*strength;
      forces[right][0] += dx*strength; forces[right][1] += dy*strength;
    }
    const temperature = RELAXATION.initialTemperature*(1-iteration/RELAXATION.iterations);
    points.forEach((point,index) => {
      const [dx,dy] = forces[index], length = Math.hypot(dx,dy);
      const step = length ? Math.min(length,temperature)/length : 0;
      // Keep the unit area used by k fixed. An outlying leaf should not stretch
      // the viewport and compress the denser part of the graph into a small fan.
      point[0] = Math.max(-0.5,Math.min(0.5,point[0]+dx*step));
      point[1] = Math.max(-0.5,Math.min(0.5,point[1]+dy*step));
    });
  }
  return points;
}

// Screen-space budgets for labels, not parameters of the graph computation.
export const LABELS = Object.freeze({padding: 2, gap: 4, height: 12, rings: 3});
const intersects = (a,b) => a.x<b.x+b.width && a.x+a.width>b.x && a.y<b.y+b.height && a.y+a.height>b.y;

export function placeLabels(points, measure, width, height, radius, priority) {
  const required = [...new Set(priority.filter(index => index!=null && points[index]))];
  const order = [...required,...points.map((_,index)=>index).filter(index=>!required.includes(index))];
  const occupied = [], placed = [];
  const nodes = points.map(([x,y])=>({x:x-radius,y:y-radius,width:2*radius,height:2*radius}));
  for (const index of order) {
    const [x,y] = points[index], text = String(index);
    const boxWidth = measure(text)+2*LABELS.padding, boxHeight = LABELS.height+2*LABELS.padding;
    const candidates = [];
    for (let ring=1; ring<=LABELS.rings; ring++) {
      const offset = radius+LABELS.gap+(ring-1)*boxHeight;
      candidates.push(
        [x+offset,y-boxHeight/2], [x-offset-boxWidth,y-boxHeight/2],
        [x-boxWidth/2,y-offset-boxHeight], [x-boxWidth/2,y+offset],
      );
    }
    // Required labels can use the top margin with a leader line when a dense
    // spectral cluster has no local opening. Reserve them before other labels.
    if (required.includes(index)) {
      for (let left=LABELS.gap; left+boxWidth<width; left+=boxWidth+LABELS.gap)
        candidates.push([left,LABELS.gap]);
    }
    const candidate = candidates.map(([x,y])=>({x,y,width:boxWidth,height:boxHeight})).find(box=>
      box.x>=0 && box.y>=0 && box.x+box.width<=width && box.y+box.height<=height &&
      !occupied.some(other=>intersects(box,other)) && !nodes.some(other=>intersects(box,other)));
    if (!candidate) continue;
    occupied.push(candidate); placed.push({index,text,...candidate});
  }
  return placed;
}
