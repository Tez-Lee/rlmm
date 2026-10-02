/* Mutable topology snapshots and exact saved traversal replay. No external CDN. */
(async()=>{
  const $=id=>document.getElementById(id);let data,row,live=null,timer;
  if(!$('v2-status'))return;
  try{let r=await fetch('/api/benchmarks_v2');if(!r.ok)r=await fetch('v2_results.json');if(!r.ok)throw Error('No v2 export yet');data=await r.json();}
  catch(e){$('v2-status').textContent=e.message;return;}
  $('v2-status').textContent=data.complete?'Three-seed main study complete.':`Main study in progress: ${48-data.missing_checkpoints.length}/48 v2 checkpoints evaluated.`;
  const fmt=s=>s.std===null?`${s.mean.toFixed(3)} (n=${s.n})`:`${s.mean.toFixed(3)} ± ${s.std.toFixed(3)}`;
  const cell=(tr,value)=>{const td=document.createElement('td');td.textContent=value;tr.append(td);};
  function summary(){
    const body=$('v2-bench');body.replaceChildren();
    const mode=$('v2-route-mode').value,budget=Number($('v2-validation').value);
    for(const s of data.summaries.filter(s=>s.validation_bytes===budget&&s.routing_mode===(s.model.startsWith('prg')?mode:'sampled'))){
      const tr=document.createElement('tr');[s.model, s.training_tokens.toLocaleString(),fmt(s.loss),fmt(s.ppl),s.static_bytes.toLocaleString(),s.peak_bytes.toLocaleString(),fmt(s.training_metadata_bytes),s.active_edges_per_token?fmt(s.active_edges_per_token):'N/A'].forEach(v=>cell(tr,v));body.append(tr);
    }
    for(const metric of ['loss','ppl'])chart(metric,budget,mode);
  }
  function chart(metric,budget,mode){
    const ctx=$('v2-'+metric+'-chart').getContext('2d'),W=570,H=280;ctx.clearRect(0,0,W,H);ctx.font='11px system-ui';
    const names=['transformer_core','transformer_total','prg_v1','prg_v2_adaptive'],colors=['#5ca4ff','#4fd1c5','#fa6c92','#d7a0ff'];
    const groups=names.map(n=>data.summaries.filter(s=>s.model===n&&s.validation_bytes===budget&&s.routing_mode===(n.startsWith('prg')?mode:'sampled')&&s[metric].n===3));
    const all=groups.flat();if(!all.length)return;
    const f=x=>$('v2-log').checked?Math.log10(x):x,xs=all.map(s=>f(s.training_tokens)),ys=all.map(s=>s[metric].mean);
    const minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);
    const X=x=>60+(f(x)-minx)/(maxx-minx||1)*490,Y=y=>215-(y-miny)/(maxy-miny||1)*180;
    ctx.fillStyle='#abc0d4';for(let i=0;i<5;i++){const y=miny+(maxy-miny)*i/4;ctx.fillText(y.toFixed(2),5,Y(y));}
    for(const x of [...new Set(all.map(s=>s.training_tokens))])ctx.fillText(x/1000+'k',X(x)-15,235);
    groups.forEach((group,i)=>{ctx.strokeStyle=colors[i];ctx.fillStyle=colors[i];ctx.beginPath();group.forEach((s,j)=>j?ctx.lineTo(X(s.training_tokens),Y(s[metric].mean)):ctx.moveTo(X(s.training_tokens),Y(s[metric].mean)));ctx.stroke();group.forEach(s=>{ctx.beginPath();ctx.arc(X(s.training_tokens),Y(s[metric].mean),3,0,Math.PI*2);ctx.fill();});ctx.fillText(names[i],10+(i%2)*280,255+Math.floor(i/2)*15);});
  }
  function choose(){
    live=null;
    row=data.results.find(r=>r.model===$('v2-model').value&&r.seed===Number($('v2-seed').value)&&r.training_tokens===Number($('v2-milestone').value));
    if(!row){$('v2-topology-stats').textContent='Selected checkpoint is pending.';$('v2-graph').replaceChildren();return;}
    const t=row.topology;
    $('v2-recurrence').checked=(row.config?.recurrence ?? row.model!=='prg_v2_no_recurrence');
    $('v2-accumulation').checked=(row.config?.accumulation ?? row.model!=='prg_v2_no_accumulation');
    $('v2-fatigue').checked=(row.config?.fatigue ?? true);
    $('v2-topology-stats').textContent=`Rewires ${t.counters.accepted}; reverts ${t.counters.reverted}; exploration ${t.counters.exploration}; exploitation ${t.counters.exploitation}; entropy ${t.topology_entropy.toFixed(3)}; Gini ${t.indegree_gini.toFixed(3)}; largest hub ${t.largest_hub}; isolated ${t.isolated_regions}; unique destinations ${t.unique_destination_regions}; edges ever ${t.unique_edges_ever}; mean current age ${t.mean_current_gateway_age.toFixed(1)} updates. Hotness/credit are proxies, not causal utility.`;
    const tb=$('v2-components');tb.replaceChildren();
    t.hotness.forEach((hot,r)=>{const tr=document.createElement('tr');[r,hot.toFixed(3),t.components.visit_activity[r].toFixed(4),t.components.usefulness[r].toFixed(3),t.components.message_activity[r].toFixed(3),t.components.accumulator_activity[r].toFixed(3),t.components.in_degree[r],t.gateway_age[r].join('/')].forEach(v=>cell(tr,v));tb.append(tr);});
    const hb=$('v2-rewires');hb.replaceChildren();
    for(const h of t.history.slice(-150).reverse()){const tr=document.createElement('tr');[h.step,h.source,h.slot,`${h.old} → ${h.new}`,h.kind,h.hot_score?.toFixed(3)??'—',h.age,h.reason].forEach(v=>cell(tr,v));hb.append(tr);}
    $('v2-token').max=Math.max(0,row.dynamics.trajectory.length-1);$('v2-token').value=0;$('v2-cycle').value=0;draw();
  }
  function draw(){
    if(!row)return;
    const timeline=live?.timeline?.map(t=>t.steps)??row.dynamics.trajectory;
    const token=Number($('v2-token').value),steps=timeline[token]??[];
    $('v2-cycle').max=Math.max(0,steps.length-1);
    const cycle=Math.min(Number($('v2-cycle').value),steps.length-1),step=steps[cycle],t=row.topology;
    const table=live?.gateway_table??t.gateway_table,R=table.length;
    let prev=data.results.filter(r=>r.model===row.model&&r.seed===row.seed&&r.training_tokens<row.training_tokens).sort((a,b)=>b.training_tokens-a.training_tokens)[0]?.topology.gateway_table;
    if(!prev)prev=Array.from({length:R},(_,r)=>Array.from({length:R-1},(_,j)=>(r+j+1)%R).sort((a,b)=>((a-r+R)*17)%R-((b-r+R)*17)%R).slice(0,table[r].length));
    const edges=new Set(table.flatMap((ds,s)=>ds.map(d=>s+','+d))),old=new Set(prev?.flatMap((ds,s)=>ds.map(d=>s+','+d))??[]);
    const svg=$('v2-graph');svg.replaceChildren();
    const ns='http://www.w3.org/2000/svg',create=(name,attrs,text)=>{const el=document.createElementNS(ns,name);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(text!==undefined)el.textContent=text;svg.append(el);return el;};
    const defs=create('defs',{});const marker=document.createElementNS(ns,'marker');Object.entries({id:'v2-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'}).forEach(([k,v])=>marker.setAttribute(k,v));const arrow=document.createElementNS(ns,'path');arrow.setAttribute('d','M 0 0 L 10 5 L 0 10 z');arrow.setAttribute('fill','#efcf69');marker.append(arrow);defs.append(marker);
    const pos=Array.from({length:R},(_,r)=>[450+190*Math.cos(2*Math.PI*r/R),210+175*Math.sin(2*Math.PI*r/R)]);
    const line=(s,d,color,dash,width=1)=>create('line',{x1:pos[s][0],y1:pos[s][1],x2:pos[d][0],y2:pos[d][1],stroke:color,'stroke-width':width,'stroke-dasharray':dash??'',opacity:width===1?.35:1});
    if(prev&&$('v2-diff').checked)for(const e of old)if(!edges.has(e)){const [s,d]=e.split(',').map(Number);line(s,d,'#f87575','4 3');}
    for(const e of edges){const [s,d]=e.split(',').map(Number);line(s,d,prev&&!old.has(e)?'#4ed9a0':'#55758c');}
    create('circle',{cx:820,cy:210,r:18,fill:'#364759',stroke:'#efcf69'});create('text',{x:792,y:245,fill:'#efcf69','font-size':12},'OUTPUT');
    for(const event of step?.events??[]){
      let el;if(event.destination>=0&&event.destination!==event.region)el=line(event.region,event.destination,'#efcf69',null,3);
      else if(event.destination<0)el=create('line',{x1:pos[event.region][0],y1:pos[event.region][1],x2:802,y2:210,stroke:'#efcf69','stroke-width':3});
      if(el)el.setAttribute('marker-end','url(#v2-arrow)');
    }
    const active=new Set(step?.active??[]),min=Math.min(...t.hotness),max=Math.max(...t.hotness);
    pos.forEach(([x,y],r)=>{const h=(t.hotness[r]-min)/(max-min||1),node=create('circle',{cx:x,cy:y,r:active.has(r)?12:8,'data-region':r,fill:`hsl(${220-h*200} 65% 55%)`,stroke:active.has(r)?'#fff':'#182333','stroke-width':active.has(r)?3:1});const title=document.createElementNS(ns,'title');title.textContent=`R${r}; hot=${t.hotness[r].toFixed(3)}; indegree=${t.components.in_degree[r]}; fatigue=${(step?.fatigue[r]??0).toFixed(3)}; accumulator=${(step?.accumulator_magnitude[r]??0).toFixed(3)}`;node.append(title);create('text',{x:x+12,y:y+4,fill:'#ddd','font-size':11},'R'+r);});
    $('v2-step-label').textContent=`${row.model} seed${row.seed}, ${row.training_tokens.toLocaleString()} training bytes; token ${token+1}, cycle ${cycle+1}. Yellow=actual traversal; green=added gateway; red dashed=removed gateway. Hover nodes for fatigue/accumulator.`;
    const details=$('v2-trajectory');details.replaceChildren();
    steps.forEach((s,i)=>{const block=document.createElement('details');block.open=i===cycle;const title=document.createElement('summary');title.textContent=`Cycle ${i+1}: `+s.events.map(e=>`R${e.region} → ${e.destination<0?'OUTPUT':'R'+e.destination}`).join(' | ');block.append(title);
      for(const e of s.events){const p=document.createElement('p');p.textContent=`Walker${e.walker}; previous R${e.previous_region}; action${e.action}; accumulator ${e.accumulator_before.toFixed(3)} → ${e.accumulator_magnitude.toFixed(3)}; fatigue ${e.fatigue.toFixed(3)}; output contribution ${e.contribution_magnitude.toFixed(3)}; probabilities [${e.route_probability.map(p=>p.toFixed(3)).join(', ')}]`;block.append(p);}details.append(block);});
  }
  $('v2-generate').onclick=async()=>{
    $('v2-live-status').textContent='Generating…';
    try{const response=await fetch('/api/v2/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:$('v2-prompt').value,seed:Number($('v2-generation-seed').value),checkpoint_seed:Number($('v2-seed').value),topology_model:$('v2-model').value,training_tokens:Number($('v2-milestone').value),max_tokens:Number($('v2-max-tokens').value),temperature:Number($('v2-temperature').value),max_cycles:Number($('v2-max-cycles').value),forced_region:$('v2-forced-region').value===''?null:Number($('v2-forced-region').value),recurrence_override:$('v2-local-override').value===''?null:Number($('v2-local-override').value),fatigue_strength:Number($('v2-fatigue-strength').value),recovery:Number($('v2-recovery').value),stochastic:$('v2-stochastic').checked,recurrence:$('v2-recurrence').checked,accumulation:$('v2-accumulation').checked,fatigue:$('v2-fatigue').checked,repeats:Number($('v2-repeat').value)})});if(!response.ok){let detail;try{detail=await response.json();}catch{}throw Error(detail?.detail??'Live generation needs the local backend and saved checkpoints.');}const result=await response.json();const runs=result.runs;const first=runs[0];$('v2-core-output').textContent=first.transformer_core.text;$('v2-peak-output').textContent=first.transformer_total?.text??'Peak model requires the updated local backend.';$('v2-static-output').textContent=first.prg_v1.text;$('v2-adaptive-output').textContent=first[$('v2-model').value].text;live=first[$('v2-model').value];$('v2-token').value=0;$('v2-token').max=live.timeline.length-1;$('v2-cycle').value=0;draw();$('v2-live-status').textContent=`${runs.length} runs; ${new Set(runs.map(r=>r[$('v2-model').value].completion)).size} distinct outputs; live backend. Byte decoding may show replacement characters.`;const visits=Array(32).fill(0);for(const run of runs)for(const token of run[$('v2-model').value].timeline)for(const step of token.steps)for(const event of step.events)visits[event.region]++;$('v2-repeat-data').textContent='Region visit counts: '+JSON.stringify(visits)+'\n'+JSON.stringify(runs.map(r=>({output:r[$('v2-model').value].completion,paths:r[$('v2-model').value].timeline.map(t=>t.steps.flatMap(s=>s.events.map(e=>e.region)))})),null,2);}
    catch(e){$('v2-live-status').textContent=e.message+' Saved replay remains available from the repository bundle.';}
  };
  for(const id of ['v2-validation','v2-route-mode','v2-log'])$(id).onchange=summary;
  for(const id of ['v2-model','v2-seed','v2-milestone'])$(id).onchange=choose;
  for(const id of ['v2-token','v2-cycle','v2-diff'])$(id).oninput=draw;
  $('v2-animate').onclick=()=>{if(timer){clearInterval(timer);timer=null;return;}let i=0;const milestones=[12800,51200,204800,819200];timer=setInterval(()=>{$('v2-milestone').value=milestones[i++%milestones.length];choose();},1500);};
  summary();choose();
})();
