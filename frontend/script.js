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
    error: '⚠️',
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
    stt: 'whisper-groq',
    llm: 'llama-groq',
    tts: 'orpheus',
};

const NUM_QUESTIONS = 5;

let turnCount  = 0;
let correctCount = 0;
let isRunning  = false;

let currentQuestion = '';
let currentQuestionIndex = 0;

// MEDIA RECORDER STATE
let mediaRecorder = null;   // MediaRecorder instance
let recordingChunks = [];   // audio data chunks
let audioStream = null;      // raw mic stram from the browser

// AUDIO PLAYBACK STATE
let currentAudio = null;    // the audio object currently playing

async function speakText(text) {
    const response = await fetch('/api/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({text: text, tts_model: config.tts}),
    });

    if (!response.ok) {
        throw new Error(`TTS failed (HTTP ${response.status})`);
    }

    const ttsLatency = parseInt(response.headers.get('X-TTS-Latency-Ms') || '0', 10);

    // read the response body as a blob (binary audio data)
    const audioBlob = await response.blob();

    // temporary URL that points at the blob in memory
    const audioUrl = URL.createObjectURL(audioBlob);

    // play the audio and wait until it finishes
    await new Promise((reslove, reject) => {
        currentAudio = new Audio(audioUrl);
        currentAudio.onended = () => reslove();
        currentAudio.onerror = () => reject(new Error('Audio playback failed'));
        currentAudio.play().catch(reject);
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

async function startQuiz() {
    isRunning = true;
    turnCount = 0;
    correctCount = 0;
    currentQuestionIndex = 0;
    document.getElementById('score-bar').style.display = 'flex';
    updateScore();

    setState(STATES.SPEAKING, 'Preparing your first question...');

    try {
        const response = await fetch('/api/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                topic: 'general knowledge',
                num_questions: NUM_QUESTIONS,
                config: config,
            }),
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        if (!isRunning) return;         // user pressed stop

        currentQuestion = data.next_question;
        currentQuestionIndex = 0;

        // speak the question out loud before opening the mic
        setState(STATES.SPEAKING, currentQuestion);
        try {
            await speakText(currentQuestion);
        } catch (err) {
            logDebug('warn', 'TTS failed: ' + err.message);
            await sleep(1500); // give time to read the text
        }
        if (!isRunning) return;

        setState(STATES.LISTENING, currentQuestion);
        setTranscript('');

    } catch (err) {
        logDebug('error', 'startQuiz failed: ' + err.message);
        setState(STATES.ERROR, 'Could not start quiz.');
        isRunning = false;
    }
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
    try {
        // request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({audio: true});
    } catch (err) {
        logDebug('error', 'mic access failed' + err.message);
        setState(STATES.ERROR, 'Could not access microphone!');
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

        await processRecording(audioBlob, mimeType);
    };

    mediaRecorder.start();

    document.getElementById('btn-ptt').textContent = '⏹ Stop';
    setState(STATES.LISTENING, '🔴 Recording...');
    logDebug('info', 'recording started');
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
    document.getElementById('btn-ptt').textContent = '🎙 Speak';
}

function releaseMicrophone() {
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }
}

async function processRecording(audioBlob, mimeType) {
    setState(STATES.PROCESSING, 'Transcribing your answer...');

    if (audioBlob.size < 1000) {
        logDebug('warn', 'recording too short (' + audioBlob.size + 'bytes)');
        setState(STATES.LISTENING, currentQuestion);
        setTranscript('(too short - speak longer)');
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
            throw new Error(`STT failed (HTTP ${transcribeResponse.status})`);
        }

        const transcribeData = await transcribeResponse.json();

        const sttTranscript = transcribeData.transcript;
        const sttLatencyMs = transcribeData.latency_ms?.stt || 0;

        setTranscript(sttTranscript);
        logDebug('info', `transcript: "${sttTranscript}" (${sttLatencyMs}ms)`)

        if (!isRunning) return;     // if user clicked 'stop'

        // send transcript to /api/evaluate
        setState(STATES.PROCESSING, 'Evaluating your answer...');

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
        if (!isRunning) return;

        // update score and continue the quiz
        if (evalData.is_correct) correctCount++;
        turnCount++;
        updateScore();

        // speak the LLM's feedback out loud
        setState(STATES.SPEAKING, evalData.message);
        let ttsMs = 0;
        try {
            ttsMs = await speakText(evalData.message);
        } catch (err) {
            logDebug('warn', 'TTS failed: ' + err.message);
            await sleep(1500); // fallback: time to read
        }

        recordTurn(
            evalData.is_correct,
            sttLatencyMs,
            evalData.latency_ms?.llm || 0,
            ttsMs,
        );

        if (!isRunning) return;

        // is quiz done?
        if (evalData.quiz_done) {
            setState(STATES.IDLE,
                `Quiz complete! You got ${correctCount} out of ${NUM_QUESTIONS} correct.`);
            document.getElementById('score-bar').style.display = 'none';
            isRunning = false;
            return;
        }

        currentQuestion = evalData.next_question;
        currentQuestionIndex++;

        setTranscript('');
        setState(STATES.SPEAKING, currentQuestion);
        try {
            await speakText(currentQuestion);
        } catch (err) {
            logDebug('warn', 'TTS failed: ', err.message);
            await sleep(1500);
        }
        if (!isRunning) return;

        setState(STATES.LISTENING, currentQuestion);

    } catch (err) {
        logDebug('error', 'pipeline failed: ' + err.message);
        setState(STATES.ERROR, err.message);
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

function stopQuiz() {
    isRunning = false;

    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
    releaseMicrophone();
    stopAudio();

    document.getElementById('btn-ptt').textContent = '🎙 Speak';
    document.getElementById('score-bar').style.display = 'none';
    setState(STATES.IDLE,
        `Quiz stopped. You got ${correctCount} out of ${turnCount} correct.`
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

// async sleep helper
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
    logEntries.unshift({ ts, type, message, stt: config.stt, llm: config.llm, tts: config.tts });

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

logDebug('info', 'app initialised — waiting for quiz start');