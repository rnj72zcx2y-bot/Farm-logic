// Farm-logic Daily Demo (MVP+)
// Improvements:
// - Drag & Drop cards (drag to first cell, then tap adjacent highlighted cell)
// - Undo button (reverts last placement)
// - Remove on board (tap occupied cell)
// - Highlight valid second cells after first cell selection

const $ = (sel) => document.querySelector(sel);

const state = {
  today: null,
  puzzle: null,
  difficulty: 'easy',
  selectedCardId: null,
  flipSelected: false,
  firstCell: null,      // {r,c}
  placements: new Map(), // cardId -> {aOnFirst:boolean, c1:{r,c}, c2:{r,c}}
  cellToPlacement: new Map(), // "r,c" -> cardId
  regionDefs: [],
  moveStack: [],        // array of cardIds placed in order
  drag: {
    active: false,
    cardId: null,
    ghostEl: null,
    lastOverCellKey: null,
  },
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
  // Project pages root: https://<user>.github.io/<repo>/
  const seg = window.location.pathname.split('/').filter(Boolean);
  const repoRoot = (seg.length >= 1) ? `/${seg[0]}/` : '/';
  const base = window.location.origin + repoRoot;
  return new URL(path.replace(/^\/+/, ''), base).toString();
}

async function fetchJson(path){
  const res = await fetch(apiUrl(path), { cache: 'no-store' });
  if(!res.ok) throw new Error(`HTTP ${res.status} for ${path}`);
  return await res.json();
}

function normalizeRule(rule){
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

function clearHighlights(){
  document.querySelectorAll('.cell').forEach(el => {
    el.classList.remove('fail','pass','validSecond','dragOver');
  });
}

function highlightValidSeconds(){
  document.querySelectorAll('.cell').forEach(el => el.classList.remove('validSecond'));
  if(!state.puzzle || !state.firstCell) return;
  const {rows, cols, activeCellsCoords} = state.puzzle.grid;
  const activeSet = new Set(activeCellsCoords.map(([r,c]) => cellKey(r,c)));

  const nbs = [
    {r: state.firstCell.r+1, c: state.firstCell.c},
    {r: state.firstCell.r-1, c: state.firstCell.c},
    {r: state.firstCell.r, c: state.firstCell.c+1},
    {r: state.firstCell.r, c: state.firstCell.c-1},
  ];
  for(const nb of nbs){
    if(nb.r<0||nb.c<0||nb.r>=rows||nb.c>=cols) continue;
    const k = cellKey(nb.r, nb.c);
    if(!activeSet.has(k)) continue;
    if(state.cellToPlacement.has(k)) continue;
    const el = document.querySelector(`.cell.active[data-r="${nb.r}"][data-c="${nb.c}"]`);
    if(el) el.classList.add('validSecond');
  }
}

function render(){
  if(!state.puzzle){
    $('#board').innerHTML = '<div class="meta">Load a puzzle to start.</div>';
    $('#cards').innerHTML = '';
    $('#rules').innerHTML = '';
    $('#placed').innerHTML = '';
    $('#placedCount').textContent = '0 / 0';
    $('#btnUndo').disabled = true;
    return;
  }

  renderBoard();
  renderCards();
  renderRules();
  renderPlaced();
  $('#btnUndo').disabled = state.moveStack.length === 0;
  highlightValidSeconds();
}

function renderBoard(){
  const b = $('#board');
  const {rows, cols, activeCellsCoords} = state.puzzle.grid;
  b.style.gridTemplateColumns = `repeat(${cols}, 58px)`;

  const activeSet = new Set(activeCellsCoords.map(([r,c]) => cellKey(r,c)));

  const html = [];
  for(let r=0;r<rows;r++){
    for(let c=0;c<cols;c++){
      const active = activeSet.has(cellKey(r,c));
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
      if(e.key === 'Enter' || e.key === ' '){ e.preventDefault(); onCellClick(cell); }
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
      <div class="card ${used?'used':''} ${selected?'selected':''}" data-id="${c.id}" draggable="false">
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
    const id = el.dataset.id;

    el.addEventListener('click', () => {
      if(state.placements.has(id)){
        toast('Card already placed. Tap an occupied cell to remove.', 'info');
        return;
      }
      state.selectedCardId = id;
      state.flipSelected = false;
      state.firstCell = null;
      $('#selInfo').textContent = `Selected ${id} (flip: off)`;
      render();
    });

    // Pointer-based drag for iPad/Safari
    el.addEventListener('pointerdown', (e) => {
      if(state.placements.has(id)) return;
      startDrag(e, el, id);
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
    btn.addEventListener('click', () => removePlacement(btn.dataset.remove, true));
  });
}

function removePlacement(cardId, pushUndo=false){
  const pl = state.placements.get(cardId);
  if(!pl) return;

  state.placements.delete(cardId);
  state.cellToPlacement.delete(cellKey(pl.c1.r, pl.c1.c));
  state.cellToPlacement.delete(cellKey(pl.c2.r, pl.c2.c));

  // also remove from moveStack if present (keep order)
  state.moveStack = state.moveStack.filter(x => x !== cardId);

  if(pushUndo){ toast(`Removed ${cardId}`, 'info'); }
  state.firstCell = null;
  render();
}

function placeCardOnCells(cardId, c1, c2){
  const aOnFirst = !state.flipSelected;
  state.placements.set(cardId, { aOnFirst, c1, c2 });
  state.cellToPlacement.set(cellKey(c1.r, c1.c), cardId);
  state.cellToPlacement.set(cellKey(c2.r, c2.c), cardId);
  state.moveStack.push(cardId);

  state.selectedCardId = null;
  state.firstCell = null;
  state.flipSelected = false;
  $('#selInfo').textContent = 'No card selected';

  toast(`Placed ${cardId}`, 'ok');
  render();
}

function onCellClick(cellEl){
  const r = parseInt(cellEl.dataset.r, 10);
  const c = parseInt(cellEl.dataset.c, 10);
  const k = cellKey(r,c);

  // Remove on-board if occupied
  if(state.cellToPlacement.has(k)){
    const cardId = state.cellToPlacement.get(k);
    removePlacement(cardId, true);
    return;
  }

  if(!state.selectedCardId){
    toast('Select or drag a card first.', 'info');
    return;
  }

  if(!state.firstCell){
    state.firstCell = {r,c};
    toast('Now tap an adjacent highlighted cell.', 'info');
    render();
    return;
  }

  const second = {r,c};
  if(!areAdjacent(state.firstCell, second)){
    toast('Cells must be adjacent (up/down/left/right).', 'bad');
    return;
  }

  if(state.cellToPlacement.has(cellKey(second.r, second.c))){
    toast('Second cell is already occupied.', 'bad');
    return;
  }

  placeCardOnCells(state.selectedCardId, state.firstCell, second);
}

function onUndo(){
  const last = state.moveStack.pop();
  if(!last){ toast('Nothing to undo.', 'info'); return; }
  const pl = state.placements.get(last);
  if(!pl){ render(); return; }
  state.placements.delete(last);
  state.cellToPlacement.delete(cellKey(pl.c1.r, pl.c1.c));
  state.cellToPlacement.delete(cellKey(pl.c2.r, pl.c2.c));
  toast(`Undid ${last}`, 'info');
  state.firstCell = null;
  render();
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
      if(seen.has(a)) return { ok:false, msg:`Duplicate species: ${a}` };
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

function highlightRegions(results){
  const activeCells = new Set(state.puzzle.grid.activeCellsCoords.map(([r,c]) => cellKey(r,c)));
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

function onCheck(){
  if(!state.puzzle){ toast('Load a puzzle first.', 'bad'); return; }

  clearHighlights();
  highlightValidSeconds();

  if(!allCardsPlaced()) toast('Not all cards placed yet.', 'info');

  const results = new Map();
  for(const reg of state.regionDefs){
    results.set(reg.id, evaluateRegion(reg));
  }

  renderRules(results);
  highlightRegions(results);

  const ok = checkWin(results);
  if(ok){ toast('🎉 Correct! All regions satisfied.', 'ok'); }
  else {
    const bad = [...results.values()].filter(r => !r.ok).length;
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
  state.moveStack = [];
  $('#selInfo').textContent = 'No card selected';
  clearHighlights();
  renderRules(null);
  toast('Reset board.', 'info');
  render();
}

function onFlip(){
  if(!state.selectedCardId){ toast('Select a card first.', 'info'); return; }
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
  if(!path) throw new Error(`No path for difficulty ${state.difficulty}`);

  const puzzle = await fetchJson(path);
  state.puzzle = puzzle;

  state.regionDefs = puzzle.regions || [];

  const date = today.dateUtc || puzzle.dateUtc || '—';
  $('#metaLine').textContent = `Date (UTC): ${date} · Difficulty: ${state.difficulty}`;

  state.selectedCardId = null;
  state.flipSelected = false;
  state.firstCell = null;
  state.placements.clear();
  state.cellToPlacement.clear();
  state.moveStack = [];
  $('#selInfo').textContent = 'No card selected';

  toast(`Loaded ${state.difficulty} for ${date}`, 'ok');
  render();
}

// ---------------- Drag & Drop (Pointer) ----------------
function startDrag(e, cardEl, cardId){
  // only left click / primary touch
  state.drag.active = true;
  state.drag.cardId = cardId;
  state.selectedCardId = cardId;
  state.flipSelected = false;
  state.firstCell = null;

  $('#selInfo').textContent = `Selected ${cardId} (dragging…)`;

  // create ghost
  const ghost = cardEl.cloneNode(true);
  ghost.classList.add('ghost');
  ghost.style.left = `${e.clientX}px`;
  ghost.style.top = `${e.clientY}px`;
  document.body.appendChild(ghost);
  state.drag.ghostEl = ghost;

  cardEl.setPointerCapture?.(e.pointerId);

  window.addEventListener('pointermove', onDragMove, {passive:false});
  window.addEventListener('pointerup', onDragEnd, {passive:false, once:true});
  window.addEventListener('pointercancel', onDragEnd, {passive:false, once:true});
}

function cellFromPoint(x, y){
  const el = document.elementFromPoint(x, y);
  if(!el) return null;
  const cell = el.closest?.('.cell.active');
  return cell || null;
}

function onDragMove(e){
  if(!state.drag.active) return;
  e.preventDefault();
  const g = state.drag.ghostEl;
  if(g){
    g.style.left = `${e.clientX}px`;
    g.style.top = `${e.clientY}px`;
  }

  const cell = cellFromPoint(e.clientX, e.clientY);
  document.querySelectorAll('.cell.dragOver').forEach(el => el.classList.remove('dragOver'));
  if(cell){
    const r = parseInt(cell.dataset.r,10);
    const c = parseInt(cell.dataset.c,10);
    const k = cellKey(r,c);
    if(!state.cellToPlacement.has(k)){
      cell.classList.add('dragOver');
      state.drag.lastOverCellKey = k;
    }
  }
}

function onDragEnd(e){
  window.removeEventListener('pointermove', onDragMove);
  document.querySelectorAll('.cell.dragOver').forEach(el => el.classList.remove('dragOver'));

  if(state.drag.ghostEl){
    state.drag.ghostEl.remove();
    state.drag.ghostEl = null;
  }

  const cell = cellFromPoint(e.clientX, e.clientY);
  state.drag.active = false;

  if(!cell){
    toast('Drop on a free active cell to start placement.', 'info');
    $('#selInfo').textContent = `Selected ${state.selectedCardId ?? '—'} (flip: off)`;
    render();
    return;
  }

  const r = parseInt(cell.dataset.r,10);
  const c = parseInt(cell.dataset.c,10);
  const k = cellKey(r,c);
  if(state.cellToPlacement.has(k)){
    toast('That cell is occupied.', 'bad');
    render();
    return;
  }

  // set first cell and let user pick second
  state.firstCell = {r,c};
  toast('Now tap an adjacent highlighted cell to finish.', 'info');
  render();
}

function wireUI(){
  $('#btnLoad').addEventListener('click', async () => {
    try{ await loadToday(); }
    catch(err){ console.error(err); toast(`Load failed: ${err.message}`, 'bad'); }
  });

  $('#btnCheck').addEventListener('click', () => {
    try{ onCheck(); }
    catch(err){ console.error(err); toast(`Check failed: ${err.message}`, 'bad'); }
  });

  $('#btnReset').addEventListener('click', () => onReset());
  $('#btnUndo').addEventListener('click', () => onUndo());
  $('#btnFlip').addEventListener('click', () => onFlip());

  $('#difficulty').addEventListener('change', () => {
    state.difficulty = $('#difficulty').value;
  });

  window.addEventListener('keydown', (e) => {
    if(e.key.toLowerCase() === 'f') onFlip();
    if(e.key.toLowerCase() === 'r') onReset();
    if(e.key.toLowerCase() === 'z') onUndo();
  });
}

wireUI();
render();
