// Optional DOM/integration check: npm install --prefix .deps/jsdom jsdom
const fs=require('node:fs'),assert=require('node:assert/strict');
const {JSDOM}=require('../.deps/jsdom/node_modules/jsdom');
const url=process.env.PRGLM_TEST_URL||'http://127.0.0.1:8870';
const dom=new JSDOM(fs.readFileSync('web/static/index.html','utf8'),{url,runScripts:'outside-only'});
const w=dom.window,errors=[];
w.addEventListener('error',e=>errors.push(e.message));
w.fetch=(path,options)=>fetch(new URL(path,url),options);
w.HTMLCanvasElement.prototype.getContext=()=>new Proxy({},{get:()=>()=>{},set:()=>true});
const wait=async(predicate)=>{for(let n=0;n<100;n++){if(predicate())return;await new Promise(r=>setTimeout(r,100));}throw Error('Timed out');};
(async()=>{
  w.eval(fs.readFileSync('web/static/learning_curve.js','utf8'));
  w.eval(fs.readFileSync('web/static/v2_dashboard.js','utf8'));
  await wait(()=>w.document.querySelector('#v2-bench tr'));
  assert.match(w.document.getElementById('curve-status').textContent,/complete/);
  const select=w.document.getElementById('v2-milestone');select.value='12800';select.dispatchEvent(new w.Event('change'));
  assert.equal(w.document.querySelectorAll('#v2-graph circle').length,32);
  assert.equal(w.document.querySelectorAll('#v2-components tr').length,32);
  assert(w.document.querySelectorAll('#v2-rewires tr').length>0);
  w.document.getElementById('v2-validation').value='16384';w.document.getElementById('v2-validation').dispatchEvent(new w.Event('change'));
  w.document.getElementById('v2-max-tokens').value=2;
  w.document.getElementById('v2-generate').click();
  await wait(()=>w.document.getElementById('v2-live-status').textContent.includes('live backend'));
  assert(w.document.getElementById('v2-adaptive-output').textContent.length>0);
  assert.equal(errors.length,0,errors.join('\n'));
  console.log('Dashboard DOM + live API passed; canvas drawing is mocked, not a browser screenshot.');
  w.close();
})().catch(e=>{console.error(e);w.close();process.exitCode=1;});
