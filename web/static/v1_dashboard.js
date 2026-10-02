let v1Current=null;
const v1$=id=>document.getElementById(id);

function average(values){return values.reduce((a,b)=>a+b,0)/Math.max(1,values.length)}

async function loadV1Benchmark(){
  try{
    let response=await fetch(api+'/api/benchmarks_v1').catch(()=>null);
    if(!response||!response.ok)response=await fetch('v1_results.json');
    if(!response.ok)return;
    let data=await response.json(),groups={};
    data.results.forEach(row=>(groups[row.model]??=[]).push(row));
    let tbody=v1$('v1_bench');tbody.replaceChildren();
    Object.entries(groups).forEach(([name,rows])=>{
      let sample=rows[0],d=sample.dynamics||{};
      let cell=[name,average(rows.map(r=>r.validation.sampled.ppl)).toFixed(1),
        name==='prg_v1'?`${average(rows.map(r=>r.validation.expected.ppl)).toFixed(1)} / ${average(rows.map(r=>r.validation.argmax.ppl)).toFixed(1)}`:'—',
        (sample.memory.core_static_packed_bytes/1024).toFixed(1),
        (sample.memory.peak_theoretical_inference_bytes/1024).toFixed(1),
        sample.memory.structural_low_bit_parameters,
        d.active_edge_ratio_per_cycle===undefined?'—':(d.active_edge_ratio_per_cycle*100).toFixed(3)+'%',
        average(rows.map(r=>r.train_tokens_per_second)).toFixed(0),
        average(rows.map(r=>r.inference_tokens_per_second)).toFixed(0)];
      let tr=tbody.insertRow();cell.forEach(v=>{let td=tr.insertCell();td.textContent=v});
    });
    let ablations=v1$('v1_ablation');ablations.replaceChildren();
    Object.entries(data.ablations||{}).forEach(([mode,rows])=>{
      let tr=ablations.insertRow();[`${mode} OFF`,average(rows.map(row=>row.sampled_ppl)).toFixed(1),rows.map(row=>row.seed).join(', ')].forEach(v=>{let td=tr.insertCell();td.textContent=v})
    });
    let targets=['prg_v1','transformer_core','transformer_total'].map(name=>groups[name]?.[0]?.memory);
    let keys=['embedding_output_head','global_router','shared_router_network','region_specific_parameters',
      'local_edges','region_gateway_addresses','transition_probabilities','other_numerical',
      'other_static_metadata','total_static_packed_bytes','node_state','fatigue_state',
      'accumulator','event_state','other_runtime_state','peak_theoretical_inference_bytes'];
    let memory=v1$('v1_memory');memory.replaceChildren();
    keys.forEach(key=>{let row=memory.insertRow();[key,...targets.map(m=>m?.static?.[key]??m?.dynamic?.[key]??m?.[key]??'—')].forEach(v=>{let td=row.insertCell();td.textContent=v})});
    let v1=groups.prg_v1?.[0],m=v1?.dynamics||{};
    v1$('v1_summary').textContent=v1?`${data.validation_tokens} held-out tokens per mode · ${m.total_local_edges?.toLocaleString()} stored slots, ${m.nonzero_ternary_local_edges?.toLocaleString()} nonzero · ${m.traversed_local_edges_per_token?.toFixed(0)} edge slots/token (${m.effective_nonzero_edges_per_token?.toFixed(0)} nonzero) · ${m.active_nodes_per_token?.toFixed(0)} unique active nodes/token · ${m.actual_traversal_count_per_token?.toFixed(2)} visits/token · Monte Carlo expected ${m.monte_carlo_expected_traversal_count_per_token?.toFixed(2)} visits/token · accumulator ${m.mean_accumulator_magnitude?.toFixed(3)} · visit/accumulator correlation ${m.visits_accumulator_correlation?.toFixed(3)??'undefined'}`:'';
    let visits=m.region_utilization||[],canvas=v1$('v1_region_hist'),g=canvas.getContext('2d');
    g.fillStyle='#0d1826';g.fillRect(0,0,canvas.width,canvas.height);
    visits.forEach((count,i)=>{let bar=canvas.width/Math.max(1,visits.length),height=count/Math.max(1,...visits)*(canvas.height-30);g.fillStyle='#2dc9af';g.fillRect(i*bar+1,canvas.height-20-height,Math.max(1,bar-2),height)});
    g.fillStyle='#abc0d4';g.fillText(`R0 → R${Math.max(0,visits.length-1)}`,8,canvas.height-5);
  }catch(e){v1$('v1_summary').textContent='PRG-v1 results are not exported yet: '+e.message}
}

function v1Action(action,gatewayCount){
  if(action===0)return 'LOCAL';
  if(action<=gatewayCount)return `NEIGHBOR_${action-1}`;
  if(action===gatewayCount+1)return 'RETURN';
  return 'OUTPUT';
}

function showV1Run(run){
  if(!run)return;
  v1Current=run;
  v1$('v1_token').max=Math.max(0,run.timeline.length-1);
  v1$('v1_token').value=0;v1$('v1_cycle').value=0;
  let timeline=v1$('v1_timeline');timeline.replaceChildren();
  run.timeline.forEach((token,index)=>{
    let details=document.createElement('details');
    let summary=document.createElement('summary');
    let events=token.steps.flatMap(s=>s.events),gatewayCount=events[0]?.route_probability.length-3||0;
    let recurrent=events.filter(e=>e.action===0||e.action===gatewayCount+1).length;
    summary.textContent=`Token ${index+1}: ${token.token} · initial ${token.steps[0]?.active.map(r=>'R'+r).join(', ')||'none'} · ${events.length} visits, ${recurrent} recurrent/return actions · ${token.steps.map(s=>s.events.map(e=>`R${e.region}→${e.destination<0?'OUT':'R'+e.destination}`).join(', ')).join(' | ')}`;
    details.append(summary);
    token.steps.forEach(step=>step.events.forEach(event=>{
      let p=document.createElement('p');
      let action=v1Action(event.action,event.route_probability.length-3);
      p.textContent=`cycle ${step.cycle} · walker ${event.walker} · R${event.region} → ${event.destination<0?'OUTPUT':'R'+event.destination} (${action}) · previous R${event.previous_region} · P(base) [${event.base_probability.map(v=>v.toFixed(2)).join(', ')}] · P(effective) [${event.route_probability.map(v=>v.toFixed(2)).join(', ')}] · accumulator ${event.accumulator_before.toFixed(3)} → ${event.accumulator_magnitude.toFixed(3)} · contribution ${event.contribution_magnitude.toFixed(3)} · fatigue ${event.fatigue.toFixed(3)}`;
      details.append(p);
    }));
    timeline.append(details);
  });
  drawV1Graph();
}
window.showV1Run=showV1Run;

function drawV1Graph(){
  if(!v1Current)return;
  let tokenIndex=+v1$('v1_token').value,steps=v1Current.timeline[tokenIndex].steps;
  if(!steps.length)return;
  v1$('v1_cycle').max=Math.max(0,steps.length-1);
  let cycle=Math.min(+v1$('v1_cycle').value,steps.length-1),step=steps[cycle];
  let svg=v1$('v1_graph');svg.replaceChildren();
  let count=step.fatigue.length,points=Array.from({length:count},(_,i)=>[
    400+150*Math.cos(i*2*Math.PI/count),180+140*Math.sin(i*2*Math.PI/count)]);
  const svgNode=(name,attrs)=>{let node=document.createElementNS('http://www.w3.org/2000/svg',name);
    Object.entries(attrs).forEach(([key,value])=>node.setAttribute(key,value));svg.append(node);return node};
  let active=new Set(step.events.map(e=>e.region));
  step.events.forEach(event=>{
    if(event.destination<0||event.destination===event.region)return;
    let src=points[event.region],dst=points[event.destination];
    svgNode('line',{x1:src[0],y1:src[1],x2:dst[0],y2:dst[1],stroke:'#ffc857','stroke-width':4,opacity:.75});
  });
  points.forEach((p,i)=>{
    svgNode('circle',{cx:p[0],cy:p[1],r:count>32?10:15,fill:active.has(i)?'#2dc9af':'#384b60'});
    let label=svgNode('text',{x:p[0],y:p[1]+4,fill:'#fff','text-anchor':'middle','font-size':count>32?8:11});
    label.textContent=i;
  });
  let routes=step.events.map(event=>`R${event.region} → ${event.destination<0?'OUTPUT':'R'+event.destination}`);
  v1$('v1_step_label').textContent=`Token ${tokenIndex+1} (${v1Current.timeline[tokenIndex].token}), cycle ${cycle+1}: ${routes.join(' · ')}`;
}

v1$('v1_token').oninput=()=>{v1$('v1_cycle').value=0;drawV1Graph()};
v1$('v1_cycle').oninput=drawV1Graph;
loadV1Benchmark();
fetch('v1_sample.json').then(r=>r.ok?r.json():null).then(data=>{
  if(data?.prg_v1){showV1Run(data.prg_v1);v1$('prg_v1').textContent=data.prg_v1.text}
}).catch(()=>{});
