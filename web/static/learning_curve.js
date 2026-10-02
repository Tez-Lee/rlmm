/* Phase A is separate from the preserved v0/v1 benchmark sections. */
(async () => {
  const section=document.getElementById('learning-curve');
  if(!section)return;
  let data;
  try {
    let response=await fetch('/api/learning_curve');
    if(!response.ok)response=await fetch('learning_curve.json');
    if(!response.ok)throw Error('No exported learning curve yet');
    data=await response.json();
  } catch(error) {
    document.getElementById('curve-status').textContent=error.message;
    return;
  }
  const fmt=s=>s.std===null?`${s.mean.toFixed(3)} (n=${s.n})`:`${s.mean.toFixed(3)} ± ${s.std.toFixed(3)} (n=${s.n})`;
  document.getElementById('curve-status').textContent=data.complete?'Phase A complete: all 36 checkpoints evaluated.':`Phase A in progress: ${data.results.length}/36 checkpoints evaluated. Missing values remain pending.`;
  function render(){
    const budget=Number(document.getElementById('curve-budget').value);
    const mode=document.getElementById('curve-mode').value;
    const log=document.getElementById('curve-log').checked;
    const summaries=data.summaries.filter(s=>s.validation_bytes===budget && s.routing_mode===(s.model==='prg_v1'?mode:'sampled'));
    const tbody=document.getElementById('curve-table');tbody.replaceChildren();
    for(const s of summaries){
      const row=document.createElement('tr');
      const values=[s.model,s.training_tokens.toLocaleString(),fmt(s.loss),fmt(s.ppl),s.static_bytes.toLocaleString(),s.peak_bytes.toLocaleString(),s.active_edges_per_token?fmt(s.active_edges_per_token):'N/A',s.model==='prg_v1'?'ON':'N/A','static'];
      for(const value of values){const td=document.createElement('td');td.textContent=value;row.append(td);}tbody.append(row);
    }
    const colors=['#5ca4ff','#4fd1c5','#fa6c92'];
    const names=['transformer_core','transformer_total','prg_v1'];
    const series=names.map((name,i)=>({name,color:colors[i],points:summaries.filter(s=>s.model===name && s.loss.n===3).map(s=>({x:s.training_tokens,loss:s.loss.mean,ppl:s.ppl.mean}))}));
    draw('curve-loss',series,'loss',log);draw('curve-ppl',series,'ppl',log);
    const gaps=data.paired_gaps.filter(g=>g.validation_bytes===budget && g.routing_mode===mode && g.seeds.length===3);
    const paired=names.slice(0,2).map((name,i)=>({name:'PRG-v1 / '+name,color:colors[i],points:gaps.filter(g=>g.baseline===name).map(g=>({x:g.training_tokens,gap:g.loss_gap.mean,ratio:g.ppl_ratio.mean}))}));
    draw('curve-gap',paired,'gap',log);draw('curve-ratio',paired,'ratio',log);
  }
  function draw(id,series,metric,log){
    const canvas=document.getElementById(id),ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height;
    ctx.clearRect(0,0,W,H);ctx.font='12px system-ui';ctx.fillStyle='#abc0d4';
    const all=series.flatMap(s=>s.points);
    if(!all.length){ctx.fillText('Waiting for three-seed checkpoint groups',30,35);return;}
    const f=x=>log?Math.log10(x):x;
    const xs=all.map(p=>f(p.x)),ys=all.map(p=>p[metric]);
    const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
    const X=x=>65+(f(x)-xmin)/(xmax-xmin||1)*(W-90),Y=y=>H-65-(y-ymin)/(ymax-ymin||1)*(H-95);
    for(let i=0;i<=4;i++){const y=ymin+(ymax-ymin)*i/4;ctx.fillText(y.toFixed(2),5,Y(y));ctx.strokeStyle='#405369';ctx.beginPath();ctx.moveTo(60,Y(y));ctx.lineTo(W-20,Y(y));ctx.stroke();}
    for(const x of [...new Set(all.map(p=>p.x))])ctx.fillText((x/1000)+'k',X(x)-15,H-45);
    series.forEach((s,i)=>{ctx.strokeStyle=s.color;ctx.fillStyle=s.color;ctx.beginPath();s.points.forEach((p,j)=>j?ctx.lineTo(X(p.x),Y(p[metric])):ctx.moveTo(X(p.x),Y(p[metric])));ctx.stroke();for(const p of s.points){ctx.beginPath();ctx.arc(X(p.x),Y(p[metric]),3,0,Math.PI*2);ctx.fill();}ctx.fillText(s.name,10+i*180,H-10);});
    ctx.fillStyle='#abc0d4';ctx.fillText(log?'Training bytes (log scale)':'Training bytes (linear scale)',180,H-27);
  }
  for(const id of ['curve-budget','curve-mode','curve-log'])document.getElementById(id).addEventListener('change',render);
  render();
})();
