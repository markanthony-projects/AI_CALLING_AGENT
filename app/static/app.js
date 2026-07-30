let ws;
let audioContext;
let audioStream;
let source;
let processor;
let playbackNode;
let audioQueue = [];
let isPlaying = false;

const startBtn = document.getElementById('startBtn');
const hangupBtn = document.getElementById('hangupBtn');
const statusText = document.getElementById('statusText');
const pulseRing = document.getElementById('pulseRing');

const urlParams = new URLSearchParams(window.location.search);
const campaignId = urlParams.get('campaign_id');
const callToken = urlParams.get('token');
let callSid = urlParams.get('call_sid');

function muLawToLinear(muLawByte) {
    muLawByte = ~muLawByte;
    let sign = (muLawByte & 0x80) ? -1 : 1;
    let exponent = (muLawByte & 0x70) >> 4;
    let mantissa = muLawByte & 0x0F;
    let sample = (mantissa << 3) + 132;
    sample <<= exponent;
    return sign * (sample - 132);
}

statusText.innerText = "Ready to connect";

startBtn.onclick = async () => {
    if (!campaignId) {
        statusText.innerText = "Missing campaign_id in the URL.";
        return;
    }
    if (!callSid) {
        callSid = 'test-' + Math.random().toString(36).slice(2, 11);
    }

    startBtn.style.display = 'none';
    hangupBtn.style.display = 'flex';
    statusText.innerText = "Calling...";
    
    try {
        audioStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    } catch(e) {
        statusText.innerText = "Mic access denied.";
        hangupBtn.style.display = 'none';
        startBtn.style.display = 'flex';
        return;
    }

    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
    if (audioContext.state === 'suspended') {
        await audioContext.resume();
    }
    
    // Ensure it connects to the Uvicorn backend (8000) even if opened via VS Code Live Server or file:///
    const host = (window.location.host && window.location.port === "8000") ? window.location.host : "127.0.0.1:8000";
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const query = callToken ? `?token=${encodeURIComponent(callToken)}` : '';
    const wsUrl = `${protocol}//${host}/ws/browser/${campaignId}/${callSid}${query}`;
    ws = new WebSocket(wsUrl);
    
    ws.onopen = async () => {
        statusText.innerText = "Connected! Speak now.";
        pulseRing.classList.add('active');
        
        await audioContext.audioWorklet.addModule('audio-processor.js?v=2');
        source = audioContext.createMediaStreamSource(audioStream);
        processor = new AudioWorkletNode(audioContext, 'resampling-processor');
        
        // Start streaming Twilio Start frame
        ws.send(JSON.stringify({ event: "start", streamSid: callSid }));
        
        processor.port.onmessage = (event) => {
            if (ws.readyState === WebSocket.OPEN) {
                const uint8Array = new Uint8Array(event.data);
                let binary = '';
                for (let i = 0; i < uint8Array.byteLength; i++) {
                    binary += String.fromCharCode(uint8Array[i]);
                }
                const base64 = btoa(binary);

                const msg = {
                    event: "media",
                    streamSid: callSid,
                    media: { payload: base64 }
                };
                ws.send(JSON.stringify(msg));
            }
        };
        
        source.connect(processor);
        processor.connect(audioContext.destination);
    };
    
    ws.onmessage = async (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.event === "media") {
                const binary = atob(data.media.payload);
                const bytes = new Uint8Array(binary.length);
                for(let i=0; i<binary.length; i++) bytes[i] = binary.charCodeAt(i);
                
                const floatArray = new Float32Array(bytes.length);
                for(let i=0; i<bytes.length; i++) {
                    let pcm = muLawToLinear(bytes[i]);
                    floatArray[i] = pcm / 32768.0;
                }
                
                const audioBuffer = audioContext.createBuffer(1, floatArray.length, 8000);
                audioBuffer.getChannelData(0).set(floatArray);
                
                audioQueue.push(audioBuffer);
                playNextAudio();
            } else if (data.event === "clear") {
                audioQueue = [];
                if (playbackNode) {
                    playbackNode.stop();
                    playbackNode.disconnect();
                }
            }
        } catch(e) {}
    };
    
    ws.onclose = () => {
        hangup();
    };
};

function playNextAudio() {
    if (isPlaying || audioQueue.length === 0) return;
    isPlaying = true;
    const buffer = audioQueue.shift();
    playbackNode = audioContext.createBufferSource();
    playbackNode.buffer = buffer;
    playbackNode.connect(audioContext.destination);
    playbackNode.onended = () => {
        isPlaying = false;
        playNextAudio();
    };
    playbackNode.start();
}

function hangup() {
    statusText.innerText = "Call ended";
    pulseRing.classList.remove('active');
    hangupBtn.style.display = 'none';
    startBtn.style.display = 'flex';
    
    if(ws) ws.close();
    if(processor) { processor.disconnect(); processor = null; }
    if(source) { source.disconnect(); source = null; }
    if(audioStream) { audioStream.getTracks().forEach(t => t.stop()); audioStream = null; }
    if(audioContext) { audioContext.close(); audioContext = null; }
}

hangupBtn.onclick = hangup;
