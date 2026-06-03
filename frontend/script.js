// state machine
const STATES = {
    IDLE: 'idle',
    SPEAKING: 'speaking',
    LISTENING: 'listening',
    PROCESSING: 'processing',
    ERROR: 'error',
};

let currentState = STATES.IDLE;

// emoji icons for state
const STATE_ICONS = {
    idle: '🎤',
    speaking: '🔊',
    listening: '👂',
    processing: '⏳',
    error: '⚠️',       // not implemented yet
};

const STATE_LABELS = {
    idle: 'Ready',
    speaking: 'Speaking',
    listening: 'Listening',
    processing: 'Processing',
    error: 'Error',
};

// change currentState
function setState(newState, message = '') {
    currentState = newState;

    const orb = document.getElementById('state-orb');
    orb.className = 'state-orb ' + newState;

    // add the pulse animation
    if (newState === STATES.LISTENING) {
        orb.classList.add('pulse');
    }

    // update orb icon, label, and message
    orb.textContent = STATE_ICONS[newState];
    document.getElementById('state-label').textContent = STATE_LABELS[newState];
    if (message) {
        document.getElementById('state-message').textContent = message;
    }

    // enable/disable buttons based on state
    document.getElementById('btn-start').disabled = newState !== STATES.IDLE;
    document.getElementById('btn-ptt').disabled = newState !== STATES.LISTENING;
    document.getElementById('btn-stop').disabled = newState === STATES.IDLE;

    logDebug('info', 'state → ' + newState);
}

/* QUIZ FLOW FUNCTIONS */

let config = {
    stt: 'whisper-openai',
    llm: 'gpt4o-mini',
    tts: 'openai-tts',
};

const NUM_QUESTIONS = 5;

let turnCount  = 0;
let correctCount = 0;
let isRunning  = false;

async function startQuiz() {
    isRunning = true;
    turnCount = 0;
    correctCount = 0;
    document.getElementById('score-bar').style.display = 'flex';
    updateScore();

    setState(STATES.SPEAKING, 'Preparing your first question...');

    // placeholder - simulate speaking for 1 sec
    await sleep(1000);
    if(!isRunning) return;
    setState(STATES.LISTENING, 'What is the capital of France?');
    setTranscript('');
}

async function pushToTalk() {
    if (!isRunning || currentState !== STATES.LISTENING) return;

    // placeholder - simulate recording for 1 sec
    setState(STATES.PROCESSING, 'Processing your answer...');
    setTranscript('Paris');

    await sleep(1000);

    if (!isRunning) return;

    // placeholder - fake result
    const isCorrect = Math.random() > 0.3;
    if (isCorrect) correctCount++;
    turnCount++;
    updateScore();
    recordTurn(isCorrect, 420, 680, 310);

    if (turnCount >= NUM_QUESTIONS) {
        setState(STATES.IDLE, `Quiz complete! You got ${correctCount} out of ${NUM_QUESTIONS} correct.`);

        document.getElementById('score-bar').style.display = 'none'
        isRunning = false;
        return;
    }

    setState(STATES.SPEAKING, isCorrect
        ? 'Correct! Paris is the capital of France. Next question coming up...'
        : 'Not quite — the answer is Paris. Next question coming up...'
    );

    await sleep(1000);

    if (!isRunning) return;

    setState(STATES.LISTENING, 'What is the largest planet in our solar system?');
    setTranscript('');
}

function stopQuiz() {
    isRunning = false;
    document.getElementById('score-bar').style.display = 'none';
    setState(STATES.IDLE,
        `Quiz finished! You got ${correctCount} out of ${turnCount} correct.`
    );
    setTranscript('');
}

/* HELPER FUNCTIONS */

// update the score bar
function updateScore() {
    document.getElementById('q-num').textContent = Math.min(turnCount + 1, NUM_QUESTIONS);
    document.getElementById('q-total').textContent = NUM_QUESTIONS;
    document.getElementById('score-correct').textContent = correctCount;
    document.getElementById('m-turns').textContent = turnCount;
    const acc = turnCount > 0
        ? Math.round((correctCount / turnCount) * 100) + '%'
        : '—';
    document.getElementById('m-acc').textContent = acc;
}

// show the STT transcript
function setTranscript(text) {
    document.getElementById('transcript-text').textContent = text || 'Your spoken answer will appear here...';
}

// async sleep helper - used for fake delays
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/* DEBUG PANEL FUNCTIONS */

let debugVisible = true;

function toggleDebug() {
    debugVisible = !debugVisible;
    document.getElementById('debug-body').classList.toggle('hidden', !debugVisible);
    document.getElementById('debug-toggle-btn').textContent =
        debugVisible ? 'hide ▲' : 'show ▼';
}

function updateConfig() {
    config.stt = document.getElementById('sel-stt').value;
    config.llm = document.getElementById('sel-llm').value;
    config.tts = document.getElementById('sel-tts').value;
    logDebug('info', `config updated: stt=${config.stt} llm=${config.llm} tts=${config.tts}`);
}

function recordTurn(correct, sttMs, llmMs, ttsMs) {
    const total = sttMs + llmMs + ttsMs;

    document.getElementById('m-total').textContent = total;

    const msg = `STT ${sttMs}ms · LLM ${llmMs}ms · TTS ${ttsMs}ms · total ${total}ms · ${correct ? 'correct' : 'wrong'}`;
    logDebug(correct ? 'correct' : 'wrong', msg);
}

let logEntries = [];

// add a line to the debug log
function logDebug(type, message) {
    const ts = new Date().toLocaleTimeString('en', {
        hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
    logEntries.unshift({ ts, type, message });

    const box = document.getElementById('log-box');

    // clear the "no turns yet" placeholder on first entry
    if (box.children.length === 1 &&
        box.children[0].querySelector('.log-ts')?.textContent === '—') {
        box.innerHTML = '';
    }

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
        [e.ts, e.type, e.message, config.stt, config.llm, config.tts].map(csvField).join(',')
    );
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' });
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

logDebug('info', 'app initialised — waiting for quiz start');