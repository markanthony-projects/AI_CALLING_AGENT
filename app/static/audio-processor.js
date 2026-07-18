function linearToMuLaw(pcm_val) {
    let sign = (pcm_val >> 8) & 0x80;
    if (sign !== 0) pcm_val = -pcm_val;
    if (pcm_val > 32635) pcm_val = 32635;
    pcm_val += 132;
    let exponent = 7;
    for (let expMask = 0x4000; (pcm_val & expMask) === 0 && exponent > 0; exponent--, expMask >>= 1) { }
    let mantissa = (pcm_val >> (exponent + 3)) & 0x0F;
    let muLawByte = ~(sign | (exponent << 4) | mantissa);
    return muLawByte & 0xFF;
}

class ResamplingProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = [];
        this.outSampleRate = 8000;
        this.inSampleRate = 48000;
        this.ratio = this.inSampleRate / this.outSampleRate;
        this.counter = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input && input.length > 0) {
            const channel = input[0];
            for (let i = 0; i < channel.length; i++) {
                this.counter++;
                if (this.counter >= this.ratio) {
                    this.buffer.push(channel[i]);
                    this.counter -= this.ratio;
                }
            }
            
            if (this.buffer.length >= 160) {
                const uint8Array = new Uint8Array(this.buffer.length);
                for (let i = 0; i < this.buffer.length; i++) {
                    let s = Math.max(-1, Math.min(1, this.buffer[i]));
                    let pcm16 = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    uint8Array[i] = linearToMuLaw(pcm16);
                }
                
                // AudioWorkletGlobalScope does not have btoa(), so we post the raw buffer
                this.port.postMessage(uint8Array.buffer, [uint8Array.buffer]);
                this.buffer = [];
            }
        }
        return true;
    }
}
registerProcessor('resampling-processor', ResamplingProcessor);
