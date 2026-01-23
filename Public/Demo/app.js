// Farm-logic Daily Demo (MVP)
// Starts from zero and loads today's puzzle via api/today.json.

const $ = (sel) => document.querySelector(sel);

const state = {
  today: null,
  puzzle: null,
  difficulty: 'easy',
  selectedCardId: null,
  flipSelected: false,
  firstCell: null,   // {r,c}
  placements: new Map(), // cardId -> {aOnFirst:boolean, c1:{r,c}, c2:{r,c}}
  cellToPlacement: new Map(), // "r,c" -> cardId
  regionOfCell: new Map(), // "r,c" -> regionId
  regionDefs: [],
};

// Animal legs mapping used for rules.
const LEGS = {
  Chicken: 2,
  Cow: 4,
  Horse: 4,
  Dog: 4,
  Cat: 4,
  Bee: 6,
  Spider: 8,
  Snail: 0,
};

function cellKey(r, c){ return `${r},${c}`; }

function toast(msg, kind='info'){
  const el = $('#toast');
  el.textContent = msg;
  el.classList.add('show');
  el.style.borderColor = kind==='ok' ? 'rgba(52,211,153,.45)' : kind==='bad' ? 'rgba(251,113,133,.55)' : 'rgba(255,255,255,.14)';
  setTimeout(() => el.classList.remove('show'), 2200);
}

function apiUrl(path){
  // Works for GitHub Pages under a repo subpath, because it is relative to current URL.
  return new URL(path.replace(/^\//,''), window.location.href).toString();
}

async function fetchJson(path){
  const res = await fetch(apiUrl(path), { cache: 'no-store' });
  if(!res.ok) throw new Error(`HTTP ${res.status} for ${path}`);
  return await res.json();
}

function normalizeRule(rule){
  // rule is like {type:'legs', op:'=', value:12} or {type:'uniqueSpecies'} etc.
  if(!rule || !rule.type) return { type: 'none' };
  return rule;
}

function ruleToText(rule){
  rule = normalizeRule(rule);
  if(rule.type === 'legs') return `Sum of legs in region ${rule.op} ${rule.value}`;
  if(rule.type === 'animals') return `Number of animals in region ${rule.op} ${rule.value}`;
  if(rule.type === 'uniqueSpecies') return `All non-empty species in region must be unique`;
  if(rule.type === 'onlySpecies') return `All cells in region must be ${rule.species} (no empty)`;
  return `No rule`;
}

function isActiveCell(activeSet, r, c){
  return activeSet.has(cellKey(r,c));
}

function areAdjacent(a, b){
  const dr = Math.abs(a.r - b.r);
  const dc = Math.abs(a.c - b.c);
  return (dr + dc) === 1;
}

function getAnimalAt(r, c){
  const k = cellKey(r,c);
  const cardId = state.cellToPlacement.get(k);
  if(!cardId) return null;
  const pl = state.placements.get(cardId);
  const card = state.puzzle.cards.find(x => x.id === cardId);
  if(!pl || !card) return null;

  const onFirst = (pl.c1.r === r && pl.c1.c === c);
  const aOnFirst = pl.aOnFirst;
  const a = card.a ?? null;
  const b = card.b ?? null;

  if(onFirst){
    return aOnFirst ? a : b;
  } else {
    return aOnFirst ? b : a;
  }
}

function render(){
  if(!state.puzzle){
    $('#board').innerHTML = '<div class="meta">Load a puzzle to start.</div>';
    $('#cards').innerHTML = '';
    $('#rules').innerHTML = '';
    $('#placed').innerHTML = '';
    $('#placedCount').textContent = '0 / 0';
    return;
  }

  renderBoard();
  renderCards();
  renderRules();
  renderPlaced();
}

function renderBoard(){
  const b = $('#board');
  const {rows, cols, activeCellsCoords} = state.puzzle.grid;

  b.style.gridTemplateColumns = `repeat(${cols}, 58px)`;

  const activeSet = new Set(activeCellsCoords.map(([r,c]) => cellKey(r,c)));

  const html = [];
  for(let r=0;r<rows;r++){
    for(let c=0;c<cols;c++){
      const active = isActiveCell(activeSet, r, c);
      const k = cellKey(r,c);
      const animal = active ? getAnimalAt(r,c) : null;
      const cls = ['cell', active ? 'active' : 'inactive'];
      if(state.firstCell && state.firstCell.r===r && state.firstCell.c===c) cls.push('selectedA');
      html.push(`
        <div class="${cls.join(' ')}" data-r="${r}" data-c="${c}" ${active?'tabindex="0"':''}>
          <div class="small">${r},${c}</div>
          <div class="animal">${animal ? animal : ''}</div>
        </div>
      `);
    }
  }
  b.innerHTML = html.join('');

  b.querySelectorAll('.cell.active').forEach(cell => {
    cell.addEventListener('click', () => onCellClick(cell));
    cell.addEventListener('keydown', (e) => {
      if(e.key === 'Enter' || e.key === ' '){
        e.preventDefault();
        onCellClick(cell);
      }
    });
  });
}

function renderCards(){
  const wrap = $('#cards');
  const cards = state.puzzle.cards;
  const html = [];

  for(const c of cards){
    const used = state.placements.has(c.id);
    const selected = state.selectedCardId === c.id;
    html.push(`
      <div class="card ${used?'used':''} ${selected?'selected':''}" data-id="${c.id}">
        <div class="domino">
          <div class="half">${c.a ?? ''}</div>
          <div class="half">${c.b ?? ''}</div>
        </div>
        <div class="fence">
          <span>${c.id}</span>
          <span>fence: ${c.fence ?? '-'}</span>
        </div>
      </div>
    `);
  }

  wrap.innerHTML = html.join('');
  wrap.querySelectorAll('.card').forEach(el => {
    el.addEventListener('click', () => {
      const id = el.dataset.id;
      if(state.placements.has(id)){
        toast('Card already placed. Remove it from the Placed list.', 'info');
        return;
      }
      state.selectedCardId = id;
      state.flipSelected = false;
      state.firstCell = null;
      $('#selInfo').textContent = `Selected ${id} (flip: off)`;
      render();
    });
  });
}

function renderRules(results=null){
  const wrap = $('#rules');
  const regions = state.regionDefs;
  const items = [];

  for(const reg of regions){
    const rule = normalizeRule(reg.rule);
    const res = results ? results.get(reg.id) : null;
    const cls = ['rule'];
    if(res){ cls.push(res.ok ? 'good' : 'bad'); }

    items.push(`
      <div class="${cls.join(' ')}" data-rid="${reg.id}">
        <div class="head">
          <div class="id">${reg.id}</div>
          <span class="badge">${reg.cells.length} cells</span>
        </div>
        <div class="txt">${ruleToText(rule)}</div>
        ${res ? `<div class="txt">${res.ok ? '✅ OK' : '❌ ' + res.msg}</div>` : ''}
      </div>
    `);
  }

  wrap.innerHTML = items.join('');
}

function renderPlaced(){
  const wrap = $('#placed');
  const total = state.puzzle.cards.length;
  const placed = state.placements.size;
  $('#placedCount').textContent = `${placed} / ${total}`;

  const items = [];
  for(const [cardId, pl] of state.placements.entries()){
    items.push(`
      <div class="placedItem">
        <div>
          <div><b>${cardId}</b> <span class="badge">${pl.c1.r},${pl.c1.c} ↔ ${pl.c2.r},${pl.c2.c}</span></div>
          <div class="meta">aOnFirst: ${pl.aOnFirst ? 'true' : 'false'}</div>
        </div>
        <button class="btn danger" data-remove="${cardId}">Remove</button>
      </div>
    `);
  }

  wrap.innerHTML = items.join('') || '<div class="meta">No cards placed yet.</div>';
  wrap.querySelectorAll('button[data-remove]').forEach(btn => {
    btn.addEventListener('click', () => removePlacement(btn.dataset.remove));
  });
}

function removePlacement(cardId){
  const pl = state.placements.get(cardId);
  if(!pl) return;

  state.placements.delete(cardId);
  state.cellToPlacement.delete(cellKey(pl.c1.r, pl.c1.c));
  state.cellToPlacement.delete(cellKey(pl.c2.r, pl.c2.c));
  toast(`Removed ${cardId}`, 'info');
  render();
}

function onCellClick(cellEl){
  if(!state.selectedCardId){
    toast('Select a card first.', 'info');
    return;
  }

  const r = parseInt(cellEl.dataset.r, 10);
  const c = parseInt(cellEl.dataset.c, 10);
  const k = cellKey(r,c);

  if(state.cellToPlacement.has(k)){
    toast('Cell already occupied. Remove the card first.', 'bad');
    return;
  }

  if(!state.firstCell){
    state.firstCell = {r,c};
    toast('Now click an adjacent cell.', 'info');
    render();
    return;
  }

  const second = {r,c};
  if(!areAdjacent(state.firstCell, second)){
    toast('Cells must be adjacent (up/down/left/right).', 'bad');
    return;
  }

  // Also check second is free
  if(state.cellToPlacement.has(cellKey(second.r, second.c))){
    toast('Second cell is already occupied.', 'bad');
    return;
  }

  const cardId = state.selectedCardId;
  const aOnFirst = !state.flipSelected; // if flipSelected, swap

  state.placements.set(cardId, { aOnFirst, c1: state.firstCell, c2: second });
  state.cellToPlacement.set(cellKey(state.firstCell.r, state.firstCell.c), cardId);
  state.cellToPlacement.set(cellKey(second.r, second.c), cardId);

  state.selectedCardId = null;
  state.firstCell = null;
  state.flipSelected = false;
  $('#selInfo').textContent = 'No card selected';

  toast(`Placed ${cardId}`, 'ok');
  render();
}

function clearHighlights(){
  document.querySelectorAll('.cell').forEach(el => {
    el.classList.remove('fail','pass');
  });
}

function highlightRegions(results){
  // highlight region cells based on pass/fail
  const activeCells = new Set(state.puzzle.grid.activeCellsCoords.map(([r,c]) => cellKey(r,c)));

  // Build mapping cell -> ok?
  const cellOk = new Map();
  for(const reg of state.regionDefs){
    const res = results.get(reg.id);
    if(!res) continue;
    for(const [r,c] of reg.cells){
      const k = cellKey(r,c);
      if(activeCells.has(k)) cellOk.set(k, res.ok);
    }
  }

  document.querySelectorAll('.cell.active').forEach(el => {
    const r = parseInt(el.dataset.r,10);
    const c = parseInt(el.dataset.c,10);
    const k = cellKey(r,c);
    if(cellOk.has(k)) el.classList.add(cellOk.get(k) ? 'pass' : 'fail');
  });
}

function compare(op, left, right){
  if(op === '=') return left === right;
  if(op === '<') return left < right;
  if(op === '>') return left > right;
  return false;
}

function evaluateRegion(reg){
  const rule = normalizeRule(reg.rule);
  let legsSum = 0;
  let animalsCount = 0;
  const seen = new Set();

  for(const [r,c] of reg.cells){
    const a = getAnimalAt(r,c);
    if(a){
      animalsCount += 1;
      legsSum += (LEGS[a] ?? 0);
    }
    if(rule.type === 'uniqueSpecies' && a){
      if(seen.has(a)){
        return { ok:false, msg:`Duplicate species: ${a}` };
      }
      seen.add(a);
    }
    if(rule.type === 'onlySpecies'){
      if(!a) return { ok:false, msg:`Empty cell not allowed (needs ${rule.species})` };
      if(a !== rule.species) return { ok:false, msg:`Found ${a}, expected ${rule.species}` };
    }
  }

  if(rule.type === 'legs'){
    const ok = compare(rule.op, legsSum, rule.value);
    return ok ? {ok:true, msg:''} : {ok:false, msg:`Legs sum is ${legsSum}`};
  }
  if(rule.type === 'animals'){
    const ok = compare(rule.op, animalsCount, rule.value);
    return ok ? {ok:true, msg:''} : {ok:false, msg:`Animals count is ${animalsCount}`};
  }
  if(rule.type === 'uniqueSpecies'){
    return {ok:true, msg:''};
  }
  if(rule.type === 'onlySpecies'){
    return {ok:true, msg:''};
  }

  return {ok:true, msg:''};
}

function allCardsPlaced(){
  return state.puzzle && state.placements.size === state.puzzle.cards.length;
}

function checkWin(results){
  if(!allCardsPlaced()) return false;
  for(const res of results.values()) if(!res.ok) return false;
  return true;
}

function onCheck(){
  if(!state.puzzle){
    toast('Load a puzzle first.', 'bad');
    return;
  }

  clearHighlights();

  // If not all cards placed, we still validate but show warning
  if(!allCardsPlaced()){
    toast('Not all cards placed yet.', 'info');
  }

  const results = new Map();
  for(const reg of state.regionDefs){
    results.set(reg.id, evaluateRegion(reg));
  }

  renderRules(results);
  highlightRegions(results);

  const ok = checkWin(results);
  if(ok){
    toast('🎉 Correct! All regions satisfied.', 'ok');
  } else {
    const bad = [...results.entries()].filter(([_,r]) => !r.ok).length;
    toast(bad ? `❌ ${bad} region(s) failing.` : '✅ No failing regions (place all cards).', bad ? 'bad' : 'ok');
  }
}

function onReset(){
  if(!state.puzzle) return;
  state.selectedCardId = null;
  state.flipSelected = false;
  state.firstCell = null;
  state.placements.clear();
  state.cellToPlacement.clear();
  $('#selInfo').textContent = 'No card selected';
  clearHighlights();
  renderRules(null);
  toast('Reset board.', 'info');
  render();
}

function onFlip(){
  if(!state.selectedCardId){
    toast('Select a card first.', 'info');
    return;
  }
  state.flipSelected = !state.flipSelected;
  $('#selInfo').textContent = `Selected ${state.selectedCardId} (flip: ${state.flipSelected ? 'on' : 'off'})`;
  toast(`Flip: ${state.flipSelected ? 'on' : 'off'}`, 'info');
}

async function loadToday(){
  state.difficulty = $('#difficulty').value;

  toast('Loading today...', 'info');
  const today = await fetchJson('api/today.json');
  state.today = today;

  const path = today.paths?.[state.difficulty];
  if(!path){
    throw new Error(`No path for difficulty ${state.difficulty}`);
  }

  const puzzle = await fetchJson(path);
  state.puzzle = puzzle;

  // Regions
  state.regionDefs = puzzle.regions || [];

  // Meta line
  const date = today.dateUtc || puzzle.dateUtc || '—';
  $('#metaLine').textContent = `Date (UTC): ${date} · Difficulty: ${state.difficulty}`;

  // Reset play state
  state.selectedCardId = null;
  state.flipSelected = false;
  state.firstCell = null;
  state.placements.clear();
  state.cellToPlacement.clear();
  $('#selInfo').textContent = 'No card selected';

  toast(`Loaded ${state.difficulty} for ${date}`, 'ok');
  render();
}

function wireUI(){
  $('#btnLoad').addEventListener('click', async () => {
    try{
      await loadToday();
    } catch(err){
      console.error(err);
      toast(`Load failed: ${err.message}`, 'bad');
    }
  });

  $('#btnCheck').addEventListener('click', () => {
    try{ onCheck(); } catch(err){
      console.error(err);
      toast(`Check failed: ${err.message}`, 'bad');
    }
  });

  $('#btnReset').addEventListener('click', () => onReset());
  $('#btnFlip').addEventListener('click', () => onFlip());

  $('#difficulty').addEventListener('change', () => {
    state.difficulty = $('#difficulty').value;
  });

  // Keyboard shortcuts
  window.addEventListener('keydown', (e) => {
    if(e.key.toLowerCase() === 'f') onFlip();
    if(e.key.toLowerCase() === 'r') onReset();
  });
}

wireUI();
render();
