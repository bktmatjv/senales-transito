'use strict';

// Dataset classes. Order defines the numeric label (0..4) used in the notebook.
const CLASSES = [
  { id: 'PARE', hint: 'Octágono con la palabra STOP' },
  { id: 'FLECHA', hint: 'Rombo con una flecha hacia la derecha' },
  { id: 'PROHIBIDO', hint: 'Círculo con una E y una línea diagonal' },
  { id: 'PEATON', hint: 'Rombo con una persona caminando' },
  { id: 'SEMAFORO', hint: 'Rectángulo vertical con tres círculos' },
];
const CLASS_IDS = CLASSES.map(c => c.id);
const SIZE = 200;
const PIXELS = SIZE * SIZE;
const LABEL_WIDTH = Math.max(...CLASS_IDS.map(id => id.length)); // 9 -> dtype <U9

// ---------- NPY v1.0 writer: ASCII header padded to 64 bytes, C order ----------
function npyBlob(descr, shape, chunks) {
  const shapeText = '(' + shape.join(', ') + (shape.length === 1 ? ',' : '') + ')';
  const dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shapeText + ', }';
  const padding = (64 - ((10 + dict.length + 1) % 64)) % 64;
  const header = new TextEncoder().encode(dict + ' '.repeat(padding) + '\n');
  const prefix = new Uint8Array(10);
  prefix.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(prefix.buffer).setUint16(8, header.length, true);
  return new Blob([prefix, header, ...chunks], { type: 'application/octet-stream' });
}

// Fixed-width little-endian UTF-32 strings, as NumPy stores dtype <U{width}.
function labelsBytes(labels, width) {
  const bytes = new Uint8Array(labels.length * width * 4);
  const view = new DataView(bytes.buffer);
  labels.forEach((label, i) => {
    [...label].forEach((ch, j) => view.setUint32((i * width + j) * 4, ch.codePointAt(0), true));
  });
  return bytes;
}

// ---------- Minimal ZIP writer (stored, no compression) ----------
const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function zipBlob(files) {
  const encoder = new TextEncoder();
  const DOS_DATE = (0 << 9) | (1 << 5) | 1; // 1980-01-01
  const parts = [];
  const central = [];
  let offset = 0;
  for (const file of files) {
    const name = encoder.encode(file.name);
    const crc = crc32(file.data);
    const size = file.data.length;

    const local = new Uint8Array(30 + name.length);
    const lv = new DataView(local.buffer);
    lv.setUint32(0, 0x04034b50, true);
    lv.setUint16(4, 20, true);
    lv.setUint16(6, 0x0800, true); // UTF-8 names
    lv.setUint16(8, 0, true); // stored
    lv.setUint16(10, 0, true);
    lv.setUint16(12, DOS_DATE, true);
    lv.setUint32(14, crc, true);
    lv.setUint32(18, size, true);
    lv.setUint32(22, size, true);
    lv.setUint16(26, name.length, true);
    lv.setUint16(28, 0, true);
    local.set(name, 30);
    parts.push(local, file.data);

    const entry = new Uint8Array(46 + name.length);
    const cv = new DataView(entry.buffer);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint16(4, 20, true);
    cv.setUint16(6, 20, true);
    cv.setUint16(8, 0x0800, true);
    cv.setUint16(10, 0, true);
    cv.setUint16(12, 0, true);
    cv.setUint16(14, DOS_DATE, true);
    cv.setUint32(16, crc, true);
    cv.setUint32(20, size, true);
    cv.setUint32(24, size, true);
    cv.setUint16(28, name.length, true);
    cv.setUint32(42, offset, true);
    entry.set(name, 46);
    central.push(entry);

    offset += local.length + size;
  }
  const centralSize = central.reduce((sum, e) => sum + e.length, 0);
  const end = new Uint8Array(22);
  const ev = new DataView(end.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, files.length, true);
  ev.setUint16(10, files.length, true);
  ev.setUint32(12, centralSize, true);
  ev.setUint32(16, offset, true);
  return new Blob([...parts, ...central, end], { type: 'application/zip' });
}

function dataUrlBytes(dataUrl) {
  const binary = atob(dataUrl.split(',')[1]);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

// ---------- Sequence helpers ----------
function shuffle(items) {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}

function makeSequence(n, mode) {
  if (mode === 'random') return Array.from({ length: n }, () => CLASS_IDS[Math.floor(Math.random() * CLASS_IDS.length)]);
  const order = shuffle([...CLASS_IDS]);
  return shuffle(Array.from({ length: n }, (_, i) => order[i % order.length]));
}

// ---------- State ----------
const $ = id => document.getElementById(id);
const canvas = $('drawing');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const exportCanvas = document.createElement('canvas');
exportCanvas.width = SIZE;
exportCanvas.height = SIZE;
const exportCtx = exportCanvas.getContext('2d');

let sequence = [];
let samples = [];
let goal = 0;
let mode = 'balanced';
let sessionId = '';
let active = false;
let pointer = null;
let last = null;
let hasInk = false;
let downloaded = { zip: null, x: null, y: null };
let revision = 0;

function message(text) { $('status').textContent = text; }
function canDraw() { return active && samples.length < goal; }
function currentLabel() { return mode === 'manual' ? $('classSelect').value : sequence[samples.length]; }
function hintFor(id) { return CLASSES.find(c => c.id === id)?.hint ?? ''; }
function countOf(id) { return samples.filter(s => s.label === id).length; }
function changed() { revision++; downloaded = { zip: null, x: null, y: null }; }

function clearDrawing() {
  pointer = null;
  last = null;
  ctx.clearRect(0, 0, SIZE, SIZE);
  hasInk = false;
  $('save').disabled = true;
}

function buildStaticUi() {
  for (const c of CLASSES) {
    const chip = document.createElement('span');
    chip.className = 'class-chip';
    chip.textContent = c.id;
    chip.title = c.hint;
    $('classChips').append(chip);

    const option = document.createElement('option');
    option.value = c.id;
    option.textContent = c.id;
    $('classSelect').append(option);

    const count = document.createElement('div');
    count.className = 'count';
    count.innerHTML = '<b id="count' + c.id + '">0</b><small>' + c.id + '</small>';
    $('counts').append(count);
  }
}

function renderPrompt(n) {
  if (!active) {
    $('prompt').textContent = 'Configura tu sesión y comienza.';
    $('phase').textContent = 'Listo para comenzar';
    return;
  }
  if (n === goal) {
    $('prompt').textContent = '¡Terminaste! Tu dataset está listo.';
    $('phase').textContent = 'Sesión completada';
    return;
  }
  const label = currentLabel();
  const target = document.createElement('span');
  target.className = 'target';
  target.textContent = label;
  const hint = document.createElement('span');
  hint.className = 'target-hint';
  hint.textContent = hintFor(label);
  $('prompt').replaceChildren(document.createTextNode('Dibuja la señal'), target, hint);
  $('phase').textContent = 'Caso ' + (n + 1) + ' de ' + goal;
}

function render() {
  const n = samples.length;
  const total = active ? goal : Number($('total').value) || 0;
  $('saved').textContent = n;
  $('goal').textContent = total;
  $('bar').style.width = (total ? (n / total) * 100 : 0) + '%';
  $('progress').setAttribute('aria-valuemax', total);
  $('progress').setAttribute('aria-valuenow', n);
  for (const id of CLASS_IDS) $('count' + id).textContent = countOf(id);

  // Class selector: free choice in manual mode, mirrors the prompt otherwise.
  if (active && mode !== 'manual' && n < goal) $('classSelect').value = sequence[n];
  $('classSelect').disabled = !(canDraw() && mode === 'manual');

  $('clear').disabled = !canDraw();
  $('save').disabled = !canDraw() || !hasInk;
  $('undo').disabled = !n;
  for (const id of ['downloadZip', 'downloadX', 'downloadY']) $(id).disabled = !n;
  $('start').textContent = active ? 'Nueva sesión' : 'Comenzar sesión →';
  $('total').disabled = active;
  $('mode').disabled = active;
  renderPrompt(n);

  const allDone = downloaded.zip === revision || (downloaded.x === revision && downloaded.y === revision);
  $('exportHint').textContent = n
    ? (allDone ? 'Archivos descargados.' : n + ' muestras · X: (' + n + ', 200, 200) · y: (' + n + ',)')
    : 'Guarda una muestra para habilitar las descargas.';

  $('gallery').replaceChildren();
  if (!n) {
    const empty = document.createElement('span');
    empty.className = 'empty';
    empty.textContent = 'Tus dibujos aparecerán aquí.';
    $('gallery').append(empty);
  }
  samples.slice(-3).forEach(sample => {
    const fig = document.createElement('figure');
    fig.className = 'thumb';
    const img = document.createElement('img');
    img.src = sample.png;
    img.alt = 'Dibujo guardado de ' + sample.label;
    const caption = document.createElement('figcaption');
    caption.textContent = sample.label;
    caption.title = sample.fileName;
    fig.append(img, caption);
    $('gallery').append(fig);
  });
}

function start() {
  const n = Number($('total').value);
  if (!Number.isInteger(n) || n < 1 || n > 1000) {
    message('Ingresa un número entero entre 1 y 1 000.');
    $('total').focus();
    return;
  }
  mode = $('mode').value;
  goal = n;
  sequence = mode === 'manual' ? [] : makeSequence(n, mode);
  sessionId = Date.now().toString(36).slice(-6);
  samples = [];
  active = true;
  changed();
  clearDrawing();
  render();
  message(mode === 'manual'
    ? 'Elige la clase en el selector, dibuja la señal y pulsa Guardar y continuar.'
    : 'Dibuja la señal indicada y pulsa Guardar y continuar.');
}

// ---------- Session controls ----------
$('start').addEventListener('click', () => { if (active) $('resetDialog').showModal(); else start(); });
$('cancelReset').onclick = () => $('resetDialog').close();
$('confirmReset').onclick = () => {
  active = false; sequence = []; samples = []; goal = 0;
  changed(); clearDrawing(); render();
  $('resetDialog').close();
  message('Configura la cantidad y comienza una nueva sesión.');
  $('total').focus();
};
$('total').oninput = () => { if (!active) render(); };
$('mode').onchange = () => {
  const hints = {
    balanced: 'Cantidades similares por señal, en orden aleatorio.',
    random: 'Cada señal se elige al azar; las cantidades pueden diferir.',
    manual: 'Tú eliges la clase de cada dibujo con el selector.',
  };
  $('modeHint').textContent = hints[$('mode').value];
};
$('classSelect').onchange = () => render();
$('brush').oninput = () => { $('brushValue').textContent = $('brush').value + ' px'; };

// ---------- Drawing ----------
function point(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((e.clientX - rect.left - canvas.clientLeft) * SIZE) / canvas.clientWidth,
    y: ((e.clientY - rect.top - canvas.clientTop) * SIZE) / canvas.clientHeight,
  };
}

canvas.addEventListener('pointerdown', e => {
  if (!canDraw() || pointer !== null || (e.pointerType === 'mouse' && e.button !== 0)) return;
  e.preventDefault();
  pointer = e.pointerId;
  canvas.setPointerCapture(pointer);
  last = point(e);
  ctx.fillStyle = '#000';
  ctx.beginPath();
  ctx.arc(last.x, last.y, Number($('brush').value) / 2, 0, Math.PI * 2);
  ctx.fill();
  hasInk = true;
  $('save').disabled = false;
});

canvas.addEventListener('pointermove', e => {
  if (e.pointerId !== pointer || !last) return;
  e.preventDefault();
  const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
  for (const ev of events.length ? events : [e]) {
    const p = point(ev);
    ctx.beginPath();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = Number($('brush').value);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    last = p;
  }
});

function endPointer(e) { if (e.pointerId === pointer) { pointer = null; last = null; } }
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);
canvas.addEventListener('lostpointercapture', endPointer);

// ---------- Save & export ----------
function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}

function downloadDataUrl(dataUrl, name) {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
}

$('clear').onclick = () => { clearDrawing(); message('Lienzo borrado. Dibuja la misma señal nuevamente.'); };

$('save').onclick = () => {
  if (!canDraw()) return;
  const rgba = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const alpha = new Uint8Array(PIXELS);
  let sum = 0;
  for (let i = 0; i < PIXELS; i++) { alpha[i] = rgba[i * 4 + 3]; sum += alpha[i]; }
  if (!sum) { message('El lienzo está vacío. Dibuja la señal antes de guardar.'); return; }

  // PNG: black strokes on a white background.
  exportCtx.fillStyle = '#fff';
  exportCtx.fillRect(0, 0, SIZE, SIZE);
  exportCtx.drawImage(canvas, 0, 0);
  const png = exportCanvas.toDataURL('image/png');

  const label = currentLabel();
  const fileName = label + '_' + sessionId + '_' + String(countOf(label) + 1).padStart(3, '0') + '.png';
  samples.push({ label, pixels: alpha, png, fileName });
  if ($('autoPng').checked) downloadDataUrl(png, fileName);

  changed();
  clearDrawing();
  render();
  message(canDraw()
    ? 'Guardado como ' + fileName + '. Continúa con la siguiente señal.'
    : 'Sesión completada. Descarga el ZIP o los archivos NumPy.');
};

$('undo').onclick = () => {
  if (!samples.length) return;
  samples.pop();
  changed();
  clearDrawing();
  render();
  message('Última muestra retirada. Dibuja nuevamente la señal indicada.');
};

$('downloadZip').onclick = () => {
  if (!samples.length) return;
  const files = samples.map(s => ({ name: s.label + '/' + s.fileName, data: dataUrlBytes(s.png) }));
  const csv = 'filename,label,label_id\n' + samples
    .map(s => s.label + '/' + s.fileName + ',' + s.label + ',' + CLASS_IDS.indexOf(s.label))
    .join('\n') + '\n';
  files.push({ name: 'labels.csv', data: new TextEncoder().encode(csv) });
  download(zipBlob(files), 'senales_' + sessionId + '.zip');
  downloaded.zip = revision;
  render();
};

$('downloadX').onclick = () => {
  if (!samples.length) return;
  download(npyBlob('|u1', [samples.length, SIZE, SIZE], samples.map(s => s.pixels)), 'X.npy');
  downloaded.x = revision;
  render();
};

$('downloadY').onclick = () => {
  if (!samples.length) return;
  const labels = samples.map(s => s.label);
  download(npyBlob('<U' + LABEL_WIDTH, [labels.length], [labelsBytes(labels, LABEL_WIDTH)]), 'y.npy');
  downloaded.y = revision;
  render();
};

window.addEventListener('beforeunload', e => {
  const exported = downloaded.zip === revision || (downloaded.x === revision && downloaded.y === revision);
  if ((samples.length && !exported) || hasInk) { e.preventDefault(); e.returnValue = ''; }
});

buildStaticUi();
render();
