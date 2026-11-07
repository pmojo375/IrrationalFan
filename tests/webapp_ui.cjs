// DOM regression checks; CI supplies jsdom. No production JavaScript dependency.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {JSDOM,VirtualConsole}=require('jsdom');
const html=fs.readFileSync(process.argv[2]||require('node:path').join(__dirname,'../webapp/index.html'),'utf8');
const game={game_id:'123',home_id:1,away_id:2,home_team:'Michigan State',away_team:'Minnesota',date:'2026-11-07',kickoff:'2026-11-07T17:00:00Z',week:10};
const key='irrationalfan-search-v1:/irrationalfan/';
let job=null,posts=0,completed=false,dropPost=false,pollFailures=0;
const errors=[];
const response=(body,status=200)=>({ok:status<400,status,text:async()=>JSON.stringify(body)});
function launch(saved){
 const vc=new VirtualConsole();vc.on('jsdomError',e=>errors.push(e));
 return new JSDOM(html,{url:'https://local.test/irrationalfan/',runScripts:'dangerously',virtualConsole:vc,beforeParse(w){
  w.matchMedia=()=>({matches:true});w.HTMLElement.prototype.scrollIntoView=()=>{};w.confirm=()=>true;
  if(saved)w.localStorage.setItem(key,JSON.stringify(saved));
  const timeout=w.setTimeout.bind(w);w.setTimeout=(fn,ms)=>timeout(fn,ms===1500||ms===4000?20:ms);
  w.fetch=async(url,opts={})=>{
   if(url==='api/features')return response({features:[{id:'rest_days',label:'Rest days',group:'previous',perspective:'team',advanced:false},{id:'opp_wins',label:'Opponent pregame wins',group:'records',perspective:'opponent',advanced:false},{id:'month',label:'Month',group:'schedule',perspective:'team',advanced:false}],groups:{previous:'Previous games',records:'Records',schedule:'Schedule'}});
   if(url.startsWith('api/games'))return response({games:[game]});
   if(url==='api/jobs'&&opts.method==='POST'){
    posts++;const settings=JSON.parse(opts.body);
    if(!job||job.status==='failed')job={id:'job-'+posts,status:'running',game,team:'Michigan State',opponent:'Minnesota',settings,created_at:new Date().toISOString(),started_at:new Date().toISOString()};
    if(dropPost){dropPost=false;throw Error('lost response');}
    return response(job,202);
   }
   if(url==='api/jobs')return response({jobs:job?[{...job,result:undefined}]:[],limit:32});
   if(url.startsWith('api/jobs/')){
    if(pollFailures-->0)throw Error('lost connection');
    if(!job)return response({detail:'Discovery no longer available. Start a new search.'},404);
    if(completed){job.status='complete';job.result={game,team:job.team,opponent:job.opponent,settings:job.settings,count:1,discoveries:[{record:'12-0',description:'A ridiculous pattern',n_games:12,win_pct:1,conditions:['Week 10'],scope:'team',ranking_score:10,matching_games:[]}],report:'Saved report'};}
    return response(job);
   }
   if(url==='api/cache'&&opts.method==='DELETE'){job=null;return response({cleared:1});}
   throw Error('Unexpected request '+url);
  };
 }});
}
async function waitFor(check){for(let n=0;n<300;n++){if(check())return;await new Promise(r=>setTimeout(r,10));}throw Error('Timed out waiting for UI');}
(async()=>{
 let dom=launch();let w=dom.window;const $=id=>w.document.getElementById(id);
 await waitFor(()=>$('run').disabled===false);
 $('discovery-mode').value='custom';$('discovery-mode').onchange();$('weirdness').value='80';$('weirdness').oninput();
 $('stat-rest_days').value='require';$('stat-rest_days').onchange();
 $('stat-month').value='exclude';$('stat-month').onchange();
 $('only-stats').checked=true;$('only-stats').onchange();
 $('run').click();await waitFor(()=>w.localStorage.getItem(key)?.includes('job-1'));
 assert.deepEqual(job.settings.allowed_stats,['rest_days']);assert.deepEqual(job.settings.required_stats,['rest_days']);assert.ok(job.settings.excluded_stats.includes('month'));
 assert.equal(posts,1);assert.equal($('job-progress').hidden,false);assert.equal($('run').disabled,true);
 const saved=JSON.parse(w.localStorage.getItem(key));dom.window.close();
 // Reload reconnects by ID without another POST, even after a transient poll failure.
 pollFailures=1;dom=launch(saved);w=dom.window;
 await waitFor(()=>$('job-detail').textContent.includes('elapsed'));
 assert.equal(posts,1);assert.equal($('stat-rest_days').value,'require');assert.equal($('stat-month').value,'exclude');assert.equal($('only-stats').checked,true);assert.equal($('stat-opp_wins').value,'allow');completed=true;
 await waitFor(()=>$('results').hidden===false);
 assert.equal($('results-title').textContent,'The case for Michigan State');
 assert.equal($('run').disabled,false);assert.equal(w.localStorage.getItem(key),null);
 await waitFor(()=>$('cached-reports').querySelector('button'));
 $('weirdness').value='5';$('cached-reports').querySelector('button').click();
 await waitFor(()=>$('weirdness').value==='80');
 assert.equal($('discovery-mode').value,'custom');assert.equal($('selected-game').textContent,'Minnesota at Michigan State');
 dom.window.close();
 // Lost submission response is retried safely and its saved result appears automatically.
 job=null;completed=false;dropPost=true;const before=posts;dom=launch();w=dom.window;
 await waitFor(()=>$('run').disabled===false);$('run').click();
 await waitFor(()=>posts===before+2);assert.equal(job.id,'job-'+(before+1));completed=true;
 await waitFor(()=>$('results').hidden===false);assert.equal($('error').textContent,'');dom.window.close();
 // Removed IDs stop retrying and give the visitor an actionable message.
 job=null;dom=launch({id:'removed'});w=dom.window;
 await waitFor(()=>$('error').textContent.includes('no longer available'));
 assert.equal($('run').disabled,false);assert.equal(w.localStorage.getItem(key),null);dom.window.close();
 assert.deepEqual(errors,[]);console.log('PASS: UI start, reload, connection recovery, automatic results, restored settings, lost POST and removed-job recovery');
})().catch(e=>{console.error(e);process.exit(1)});
