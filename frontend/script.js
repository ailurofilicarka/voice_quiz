// state machine
const STATES = {
    IDLE: 'idle',
    SPEAKING: 'speaking',
    LISTENING: 'listening',
    PROCESSING: 'processing',
    ERROR: 'error',
};

let currentState = STATES.IDLE;

const STATE_LABELS = {
    idle: 'ready',
    speaking: 'speaking',
    listening: 'listening',
    processing: 'processing',
    error: 'error',
};

// change currentState
function setState(newState) {
    currentState = newState;

    document.getElementById('state-dot').className = 'state-dot' + newState;
    document.getElementById('state-label').textContent = STATE_LABELS[newState];
 
    bars.quiz.active = newState === STATES.SPEAKING;

    if (newState !== STATES.LISTENING) bars.you.active = false;
 
    document.getElementById('btn-mic').disabled = newState !== STATES.LISTENING;
 
    logDebug('info', 'state → ' + newState);
}

// end-of-quiz summary and error messages
function setStatus(text, isError = false) {
    const el = document.getElementById('status');
    el.textContent = text || '';
    el.classList.toggle('error', !!isError);
}

function setQuestion(text, dim = false) {
    const el = document.getElementById('question');
    el.textContent = text || '';
    el.classList.toggle('dim', dim);
}

/* QUIZ FLOW FUNCTIONS */

let runId = 0;

let config = {
    stt: 'whisper-groq',
    llm: 'llama-groq',
    tts: 'orpheus',
};

let totalQuestions = 5;
let topic = 'general knowledge';
let personality = 'classic';

let turnCount  = 0;
let correctCount = 0;
let isRunning  = false;
let results = [];

let currentQuestion = '';
let currentQuestionIndex = 0;

// MEDIA RECORDER STATE
let mediaRecorder = null;   // MediaRecorder instance
let recordingChunks = [];   // audio data chunks
let audioStream = null;      // raw mic stram from the browser

// AUDIO PLAYBACK STATE
let currentAudio = null;    // the audio object currently playing

// THEME
function toggleTheme() {
    const root = document.documentElement;
    const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
    root.dataset.theme = next;
 
    document.getElementById('theme-label').textContent = next === 'dark' ? 'Light' : 'Dark';
    document.getElementById('theme-icon').innerHTML = next === 'dark'
        ? '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>'
        : '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>';
}

// TTS PLAYBACK
async function speakText(text, token) {
    logDebug('info', `TTS input: "${text}"`);

    const response = await fetch('/api/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text, tts_model: config.tts }),
    });

    if (!response.ok) {
        throw new Error(`TTS failed (HTTP ${response.status})`);
    }

    const ttsLatency = parseInt(response.headers.get('X-TTS-Latency-Ms') || '0', 10);

    // read the response body as a blob (binary audio data)
    const audioBlob = await response.blob();
    logDebug('info', `TTS blob: ${audioBlob.size} bytes`);

    if (token !== runId) return 0;

    // temporary URL that points at the blob in memory
    const audioUrl = URL.createObjectURL(audioBlob);

    // play the audio and wait until it finishes
    await new Promise((resolve, reject) => {
        currentAudio = new Audio(audioUrl);
        // currentAudio.onloadedmetadata = () =>
        //     logDebug('info', `TTS duration: ${currentAudio.duration}s`);
        currentAudio.onended = () => resolve();
        currentAudio.onerror = () => reject(new Error('Audio playback failed'));
        // currentAudio.oncanplaythrough = () => currentAudio.play().catch(reject);
        currentAudio.oncanplaythrough = () => {
            if (token !== runId) {resolve(); return;}
            currentAudio.play().catch(reject);
        };
    });

    // clean up
    URL.revokeObjectURL(audioUrl);
    currentAudio = null;

    return ttsLatency;
}

// stop any currently playing audio
function stopAudio() {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
}


// QUIZ FLOW

async function startQuiz() {
    const token = ++runId;
    
    const requested = parseInt(document.getElementById('cfg-count').value, 10);
    totalQuestions = Math.min(20, Math.max(3, requested || 5));

    isRunning = true;
    turnCount = 0;
    correctCount = 0;
    currentQuestionIndex = 0;
    results = [];

    document.getElementById('screen-setup').hidden = true;
    document.getElementById('screen-quiz').hidden = false;
    document.getElementById('q-total').textContent = totalQuestions;
    document.getElementById('q-num').textContent = 1;
    setStatus('');
    setQuestion('');
    setTranscript('');
    renderTicks();
    updateScore();
    sizeCanvas();

    logDebug('info', `quiz started · ${topic} · ${personality} · ${totalQuestions} questions`);
    setState(STATES.PROCESSING);

    try {
        const response = await fetch('/api/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                topic: topic,
                num_questions: totalQuestions,
                personality: personality,
                config: config,
            }),
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        // if (!isRunning) return;         // user pressed stop
        if (token !== runId) return;

        currentQuestion = data.next_question;
        currentQuestionIndex = 0;

        setQuestion(currentQuestion);
        setState(STATES.SPEAKING);
        try {
            await speakText(data.message, token);
        } catch (err) {
            logDebug('warn', 'TTS failed: ' + err.message);
            await sleep(1500);
        }
        // if (!isRunning) return;
        if (token !== runId) return;
        
        setState(STATES.LISTENING);

    } catch (err) {
        logDebug('error', 'startQuiz failed: ' + err.message);
        setState(STATES.ERROR);
        setStatus('Could not start quiz.', true);
        isRunning = false;
    }
}

// speak the current question, then open the microphone
async function askQuestion(token) {
    document.getElementById('q-num').textContent = currentQuestionIndex + 1;

    logDebug('info', `Q${currentQuestionIndex + 1}: ${currentQuestion}`);
 
    setQuestion(currentQuestion);
    setTranscript('');
    setState(STATES.SPEAKING);
 
    try {
        await speakText(currentQuestion, token);
    } catch (err) {
        logDebug('warn', 'TTS failed: ' + err.message);
        setStatus('Question could not be read aloud — check the log.', true);
        await sleep(1500);
    }
 
    // if (!isRunning) return;
    if (token !== runId) return;
    setState(STATES.LISTENING);
}

async function pushToTalk() {
    if (!isRunning || currentState !== STATES.LISTENING) return;

    // 'stop' is pressed
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
        return;
    }

    // "speak" is pressed
    await startRecording();
}

async function startRecording() {
    const token = runId;
    try {
        // request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({audio: true});
    } catch (err) {
        logDebug('error', 'mic access failed' + err.message);
        setState(STATES.ERROR);
        setStatus('Could not access the microphone.', true);
        isRunning = false;
        return;
    }

    recordingChunks = [];

    // create the recorder
    mediaRecorder = new MediaRecorder(audioStream);

    // collect small audio chunks
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordingChunks.push(event.data);
        }
    };

    // assemble all chunks into a single Blob
    mediaRecorder.onstop = async () => {
        const mimeType = mediaRecorder.mimeType;
        const audioBlob = new Blob(recordingChunks, { type: mimeType });

        releaseMicrophone();

        await processRecording(audioBlob, mimeType, token);
    };

    mediaRecorder.start();

    bars.you.active = true;
    startAnalyser(audioStream);

    const mic = document.getElementById('btn-mic');
    mic.textContent = 'Stop';
    mic.classList.add('is-recording');
    logDebug('info', 'recording started');
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
    const mic = document.getElementById('btn-mic');
    mic.textContent = 'Speak';
    mic.classList.remove('is-recording');
}

function releaseMicrophone() {
    bars.you.active = false;
    stopAnalyser();

    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }
}

// PIPELINE: transcribe - evaluate - speak feedback - next question

async function processRecording(audioBlob, mimeType, token) {
    setState(STATES.PROCESSING);

    if (audioBlob.size < 1000) {
        logDebug('warn', 'recording too short (' + audioBlob.size + 'bytes)');
        setTranscript('(too short — speak a little longer)');
        setState(STATES.LISTENING);
        return;
    }

    try {
        // send audio to /api/transcribe
        const extension = mimeTypeToExtension(mimeType);
        const formData = new FormData();
        formData.append('audio', audioBlob, `recording.${extension}`);

        const transcribeResponse = await fetch(
            `/api/transcribe?stt_model=${encodeURIComponent(config.stt)}`,
            { method: 'POST', body: formData }
            // browser sets automatically Content-Type header here
        );

        if (!transcribeResponse.ok) {
            const detail = await transcribeResponse.text();
            throw new Error(`STT failed (HTTP ${transcribeResponse.status}): ${detail.slice(0, 200)}`);
        }

        const transcribeData = await transcribeResponse.json();

        const sttTranscript = transcribeData.transcript;
        const sttLatencyMs = transcribeData.latency_ms?.stt || 0;

        setTranscript(sttTranscript);
        logDebug('info', `transcript: "${sttTranscript}" (${sttLatencyMs}ms)`)

        // if (!isRunning) return;     // if user clicked 'stop'
        if (token !== runId) return;

        // send transcript to /api/evaluate
        const evaluateResponse = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                transcript: sttTranscript,
                question: currentQuestion,
                question_index: currentQuestionIndex,
                config: config,
            }),
        });

        if (!evaluateResponse.ok) {
            throw new Error(`Evaluate filed ${evaluateResponse.status}`);
        }

        const evalData = await evaluateResponse.json();
        // if (!isRunning) return;
        if (token !== runId) return;

        // update score and continue the quiz
        if (evalData.is_correct) correctCount++;
        turnCount++;
        results.push(!!evalData.is_correct);
 
        showVerdict(evalData.is_correct);
        renderTicks();
        updateScore();

        const readMs = Math.min(2500, Math.max(600, sttTranscript.length * 40));
        await sleep(readMs);
        // if (!isRunning) return;
        if (token !== runId) return;

        setTranscript(evalData.message, 'Host')

        // speak the LLM's feedback out loud
        setState(STATES.SPEAKING);
        let ttsMs = 0;
        try {
            ttsMs = await speakText(evalData.message, token);
        } catch (err) {
            logDebug('warn', 'TTS failed: ' + err.message);
            await sleep(1500); // fallback: time to read
        }

        recordTurn(evalData.is_correct, sttLatencyMs, evalData.latency_ms?.llm || 0, ttsMs);

        // if (!isRunning) return;
        if (token !== runId) return;

        // is quiz done?
        if (evalData.quiz_done) {
            finishQuiz();
            return;
        }

        currentQuestion = evalData.next_question;
        currentQuestionIndex++;

        await askQuestion(token);

    } catch (err) {
        logDebug('error', 'pipeline failed: ' + err.message);
        setState(STATES.ERROR);
        setStatus(err.message, true);
        isRunning = false;
    }
    
}

function mimeTypeToExtension(mimeType) {
    if (!mimeType) return 'webm';
    if (mimeType.includes('webm')) return 'webm';
    if (mimeType.includes('ogg')) return 'ogg';
    if (mimeType.includes('mp4')) return 'mp4';
    if (mimeType.includes('mp3')) return 'mp3';
    if (mimeType.includes('wav')) return 'wav';
    return 'webm';
}

function finishQuiz() {
    isRunning = false;
    setState(STATES.IDLE);
    setQuestion(`Quiz complete! Score: ${correctCount}/${totalQuestions}`);
    setTranscript('');
    setStatus('');
    document.getElementById('btn-mic').disabled = true;
    logDebug('info', `quiz complete · ${correctCount}/${totalQuestions}`);
}

// back to the setup screen
function stopQuiz() {
    runId++;
    isRunning = false;

    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
    releaseMicrophone();
    stopAudio();

    const mic = document.getElementById('btn-mic');
    mic.textContent = 'Speak';
    mic.classList.remove('is-recording');
 
    setState(STATES.IDLE);
    logDebug('info', `quiz stopped · ${correctCount}/${turnCount}`);
 
    document.getElementById('screen-quiz').hidden = true;
    document.getElementById('screen-setup').hidden = false;
}

// VERDICT

function showVerdict(correct) {
    const css = getComputedStyle(document.documentElement);
    verdict.colour = css.getPropertyValue(correct ? '--right' : '--danger').trim();
    verdict.until = performance.now() + 1400;
 
    // kick every bar outward, then let the normal decay bring it down
    const count = bars.quiz.levels.length;
    const shape = i => 0.55 + 0.45 * Math.sin(Math.PI * (i / count));
    for (let i = 0; i < count; i++) {
        bars.quiz.levels[i] = shape(i) * (correct ? 1 : 0.6);
        bars.you.levels[i]  = shape(i) * (correct ? 1 : 0.6);
    }
 
    const scoreEl = document.getElementById('score');
    scoreEl.classList.remove('pulse-right', 'pulse-wrong');
    void scoreEl.offsetWidth;                   // restart the CSS animation
    scoreEl.classList.add(correct ? 'pulse-right' : 'pulse-wrong');
    setTimeout(() => scoreEl.classList.remove('pulse-right', 'pulse-wrong'), 600);
 
    const box = document.getElementById('transcript-box');
    box.classList.remove('right', 'wrong');
    box.classList.add(correct ? 'right' : 'wrong');
    setTimeout(() => box.classList.remove('right', 'wrong'), 1600);
}

/* HELPER FUNCTIONS */

// update the score bar
function updateScore() {
    document.getElementById('score').textContent = correctCount;
}

// one tick per question
function renderTicks() {
    const el = document.getElementById('ticks');
    el.innerHTML = '';
    for (let i = 0; i < totalQuestions; i++) {
        const tick = document.createElement('span');
        let cls = 'tick';
        if (i < results.length)      cls += results[i] ? ' done' : ' wrong';
        else if (i === results.length) cls += ' now';
        tick.className = cls;
        el.appendChild(tick);
    }
}

// show the STT transcript
function setTranscript(text, label = 'Heard') {
    const el = document.getElementById('transcript');
    el.textContent = text || 'waiting for your answer';
    el.classList.toggle('empty', !text);
    document.querySelector('.transcript-label').textContent = label;
}

// async sleep helper
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// EQUALIZER

const canvas = document.getElementById('scope');
const ctx = canvas.getContext('2d');
 
const BAR_W = 3, BAR_GAP = 4, MAX_H = 46;
 
const bars = {
    quiz: { active: false, levels: [], seed: 0.0 },
    you:  { active: false, levels: [], seed: 9.3 },
};
 
const verdict = { colour: null, until: 0 };
 
function sizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
 
    const count = Math.floor(rect.width / (BAR_W + BAR_GAP));
    for (const ch of Object.values(bars)) {
        while (ch.levels.length < count) ch.levels.push(0);
        ch.levels.length = count;
    }
}
window.addEventListener('resize', sizeCanvas);
 
function getLevels(ch, i, count, t) {
    if (!ch.active) return 0;

    // real microphone spectrum for the user channel
    if (ch === bars.you && freqData) {
        const usable = Math.floor(freqData.length * 0.45);
        const bin = Math.min(usable - 1, Math.floor((i / count) * usable));
        return freqData[bin] / 255;
    }

    const syllable = 0.55 + 0.45 * Math.sin(t * 0.006 + ch.seed)
                          * Math.sin(t * 0.017 + ch.seed * 2);
    const p = i / count;
    const shape = Math.sin(Math.PI * p) ** 0.6;      // quieter towards the edges
    const detail = 0.45
        + 0.30 * Math.sin(p * 26 + t * 0.011 + ch.seed)
        + 0.25 * Math.sin(p * 57 - t * 0.008 + ch.seed * 3);
    return Math.max(0, syllable * shape * detail);
}
 
function drawChannel(ch, colour, offset, t) {
    const rect = canvas.getBoundingClientRect();
    const mid = rect.height / 2;
    const count = ch.levels.length;
 
    ctx.fillStyle = colour;
    for (let i = 0; i < count; i++) {
        const target = getLevels(ch, i, count, t);
        const k = target > ch.levels[i] ? 0.35 : 0.10;   // fast attack, slow decay
        ch.levels[i] += (target - ch.levels[i]) * k;
 
        const h = ch.levels[i] * MAX_H;
        if (h < 0.4) continue;                            // silent bars leave only the line
        const x = i * (BAR_W + BAR_GAP) + offset;
        ctx.beginPath();
        ctx.roundRect(x, mid - h, BAR_W, h * 2, BAR_W / 2);
        ctx.fill();
    }
}
 
function frame(t) {
    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);
 
    const css = getComputedStyle(document.documentElement);
    const mid = rect.height / 2;
    const flashing = performance.now() < verdict.until;

    if (analyser) analyser.getByteFrequencyData(freqData);
 
    // the resting line takes the colour of whoever is active
    const lineColour = flashing ? verdict.colour
        : bars.quiz.active ? css.getPropertyValue('--quiz').trim()
        : bars.you.active  ? css.getPropertyValue('--you').trim()
        : css.getPropertyValue('--border').trim();
 
    ctx.strokeStyle = lineColour;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(rect.width, mid);
    ctx.stroke();
 
    const quizColour = flashing ? verdict.colour : css.getPropertyValue('--quiz').trim();
    const youColour  = flashing ? verdict.colour : css.getPropertyValue('--you').trim();
 
    drawChannel(bars.quiz, quizColour, 0, t);
    drawChannel(bars.you,  youColour, (BAR_W + BAR_GAP) / 2, t);
 
    requestAnimationFrame(frame);
}


let audioCtx = null;
let analyser = null;
let freqData = null;

function startAnalyser(stream) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;                  // 128 frequency bins
    analyser.smoothingTimeConstant = 0.65;   // the analyser does some smoothing itself
    source.connect(analyser);
    freqData = new Uint8Array(analyser.frequencyBinCount);
}

function stopAnalyser() {
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    analyser = null;
    freqData = null;
}

/* DEBUG PANEL FUNCTIONS */

let debugVisible = true;

function toggleDebug() {
    debugVisible = !debugVisible;
    document.getElementById('debug-body').hidden = !debugVisible;
    document.getElementById('debug-caret').textContent = debugVisible ? 'hide ▲' : 'show ▼';
}

function updateConfig() {
    config.stt = document.getElementById('sel-stt').value;
    config.llm = document.getElementById('sel-llm').value;
    config.tts = document.getElementById('sel-tts').value;
    logDebug('info', `config updated: stt=${config.stt} llm=${config.llm} tts=${config.tts}`);
}

function recordTurn(correct, sttMs, llmMs, ttsMs) {
    const total = sttMs + llmMs + ttsMs;

    const msg = `STT ${sttMs}ms · LLM ${llmMs}ms · TTS ${ttsMs}ms · total ${total}ms · ${correct ? 'correct' : 'wrong'}`;
    logDebug(correct ? 'correct' : 'wrong', msg);
}

let logEntries = [];

// add a line to the debug log
function logDebug(type, message) {
    const ts = new Date().toLocaleTimeString('en', {
        hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
    logEntries.unshift({ ts, type, message, stt: config.stt, llm: config.llm, tts: config.tts });

    const box = document.getElementById('log');
    const div = document.createElement('div');
    div.className = 'log-entry';

    let cls = '';
    if (type === 'ok')   cls = 'log-ok';
    if (type === 'warn' || type === 'error') cls = 'log-err';

    div.innerHTML =
        `<span class="log-ts">${ts}</span>` +
        (cls ? `<span class="${cls}">${type.toUpperCase()}</span>` : '') +
        `<span>${message}</span>`;

    box.prepend(div);

    // keep only the last 50 log entries
    while (box.children.length > 50) box.removeChild(box.lastChild);
}

// Export the log as a csv file
function exportLog() {
    const csvField = (value) => `"${String(value).replace(/"/g, '""')}"`;

    const header = 'time,type,message,stt,llm,tts';
    const rows = logEntries.map(e =>
        [e.ts, e.type, e.message, e.stt, e.llm, e.tts].map(csvField).join(',')
    );
    const blob = new Blob(['\uFEFF' + [header, ...rows].join('\n')], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'quiz_log.csv';
    a.click();
}

function clearLog() {
    logEntries = [];
    document.getElementById('log-box').innerHTML =
        '<div class="log-entry"><span class="log-ts">—</span><span>Log cleared.</span></div>';
    document.getElementById('m-total').textContent = '—';
}

// SETUP SCREEN CONTROLS
function initChoiceGroup(containerId, onSelect) {
    const container = document.getElementById(containerId);
    container.addEventListener('click', (event) => {
        const btn = event.target.closest('[data-value]');
        if (!btn) return;

        container.querySelectorAll('[data-value]').forEach(el => {
            const active = el === btn;
            el.classList.toggle('is-active', active);
            el.setAttribute('aria-checked', active ? 'true' : 'false');
        });

        onSelect(btn.dataset.value);
    });
}

initChoiceGroup('category-row', value => { topic = value; });
initChoiceGroup('host-grid',    value => { personality = value; });

// slider readout
const countInput = document.getElementById('cfg-count');
countInput.addEventListener('input', () => {
    document.getElementById('count-value').textContent = countInput.value;
});

// BOOT

sizeCanvas();
requestAnimationFrame(frame);
updateConfig();
logDebug('info', 'app initialised, waiting for quiz start');