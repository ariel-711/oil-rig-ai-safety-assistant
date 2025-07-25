from flask import Flask, request, jsonify, send_file, render_template_string
import io
import os
from ai_assistant_voice_goodvoice_02 import VoiceAssistant
from langdetect import detect

app = Flask(__name__)
assistant = VoiceAssistant()

# Modern ChatGPT Voice-like UI with Wake Word Detection ("Hey Rig")
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>EGR – Process Safety Innovation AI Voice Chat</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
  <style>
    body {
      font-family: 'Inter', Arial, sans-serif;
      background: linear-gradient(120deg, #0a192f 60%, #274690 100%);
      margin: 0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
    }
    .chat-container {
      width: 100%;
      max-width: 600px;
      min-width: 0;
      margin: 48px auto;
      background: #fff;
      border-radius: 22px;
      box-shadow: 0 8px 48px 0 #0003, 0 1.5px 8px #27469022;
      padding: 0 0 32px 0;
      overflow: hidden;
      box-sizing: border-box;
      transition: max-width 0.2s, width 0.2s, box-shadow 0.2s, border-radius 0.2s;
      display: flex;
      flex-direction: column;
      justify-content: flex-start;
      align-items: stretch;
    }
    .header { background: linear-gradient(90deg, #1a365d 60%, #274690 100%); color: #fff; padding: 24px 32px 16px 32px; text-align: center; }
    .header h1 { margin: 0; font-size: 2em; letter-spacing: 1px; }
    .header .wake-status { font-size: 1em; margin-top: 8px; color: #b3c7e6; }
    .chat-box { height: 420px; overflow-y: auto; border: none; padding: 24px 32px 0 32px; background: #f7fafd; }
    .msg { margin: 18px 0; display: flex; align-items: flex-start; }
    .msg.user { justify-content: flex-end; }
    .msg .bubble { max-width: 75%; padding: 14px 18px; border-radius: 16px; font-size: 1.08em; line-height: 1.5; }
    .msg.user .bubble { background: #274690; color: #fff; border-bottom-right-radius: 4px; margin-left: 40px; }
    .msg.assistant .bubble { background: #e9f1fb; color: #1a365d; border-bottom-left-radius: 4px; margin-right: 40px; }
    .avatar { width: 36px; height: 36px; border-radius: 50%; background: #dbeafe; display: flex; align-items: center; justify-content: center; font-size: 1.3em; margin-right: 12px; }
    .msg.user .avatar { display: none; }
    .msg.assistant .avatar { margin-left: 0; margin-right: 12px; }
    .input-row { display: flex; align-items: center; margin: 24px 32px 0 32px; }
    .input-row input { flex: 1; padding: 12px; border-radius: 10px; border: 1px solid #b3c7e6; font-size: 1.1em; background: #f7fafd; }
    .input-row button { margin-left: 10px; border-radius: 10px; border: none; background: #1a365d; color: #fff; padding: 12px 22px; font-size: 1.1em; font-weight: 600; cursor: pointer; transition: background 0.2s; }
    .input-row button:hover { background: #274690; }
    .mic-btn { background: #e9f1fb; color: #1a365d; border: none; border-radius: 50%; width: 48px; height: 48px; margin-left: 10px; cursor: pointer; font-size: 1.3em; display: flex; align-items: center; justify-content: center; transition: background 0.2s, color 0.2s, box-shadow 0.2s; }
    .mic-btn.listening {
      background: #00e676 !important; /* bright green */
      color: #fff !important;
      box-shadow: 0 0 12px 4px #00e67699;
      border: 2px solid #00c853;
    }
    .wake-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: #b3c7e6; margin-right: 8px; vertical-align: middle; transition: background 0.2s; }
    .wake-indicator.active { background: #00e676; }
    @media (max-width: 800px) {
      .chat-container {
        max-width: 98vw;
        width: 100vw;
        border-radius: 0;
        box-shadow: none;
        margin: 0;
      }
      .chat-box, .input-row, .header {
        padding-left: 8px;
        padding-right: 8px;
      }
    }

    @media (min-width: 1200px) {
      .chat-container {
        max-width: 900px;
        padding-bottom: 48px;
        border-radius: 28px;
        box-shadow: 0 12px 64px 0 #0004, 0 2px 12px #27469022;
      }
      .chat-box, .input-row, .header {
        padding-left: 48px;
        padding-right: 48px;
      }
    }
  </style>
</head>
<body>
  <div class="chat-container">
    <div class="header">
      <h1>EGR <span style="font-size:0.7em;">Process Safety Innovation</span></h1>
      <div class="wake-status"><span class="wake-indicator" id="wake-indicator"></span> Wake word: <b>Hey Rig</b> <span id="wake-status-msg">(listening...)</span></div>
    </div>
    <div class="chat-box" id="chat"></div>
    <div class="input-row">
      <input id="user-input" type="text" placeholder="Type your message or say 'Hey Rig'..." autocomplete="off" />
      <button onclick="sendMessage()">Send</button>
      <button class="mic-btn" id="mic-btn" onclick="activateMic()" title="Click to speak (no wake word needed)">🎤</button>
    </div>
    <div style="display: flex; justify-content: flex-end; margin: 8px 32px 0 32px;">
      <label for="lang-select" style="margin-right: 8px; color: #1a365d; font-weight: 600;">Language:</label>
      <select id="lang-select" style="padding: 6px 12px; border-radius: 8px; border: 1px solid #b3c7e6; font-size: 1em;">
        <option value="en">English</option>
        <option value="pt">Português</option>
      </select>
    </div>
    <audio id="audio" style="display:none;"></audio>
  </div>
  <script>
    const chat = document.getElementById('chat');
    const input = document.getElementById('user-input');
    const audio = document.getElementById('audio');
    const micBtn = document.getElementById('mic-btn');
    const wakeIndicator = document.getElementById('wake-indicator');
    const wakeStatusMsg = document.getElementById('wake-status-msg');

    let isListening = false;
    let wakeActive = false;
    let recognition;

    function appendMessage(sender, text) {
      const div = document.createElement('div');
      div.className = 'msg ' + (sender === 'You' ? 'user' : 'assistant');
      if (sender === 'You') {
        div.innerHTML = `<div class="bubble">${text}</div>`;
      } else {
        div.innerHTML = `<span class="avatar">🦺</span><div class="bubble">${text}</div>`;
      }
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    }

    async function sendMessage() {
      const text = input.value.trim();
      if (!text) return;
      appendMessage('You', text);
      input.value = '';
      const resp = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      });
      const data = await resp.json();
      appendMessage('Assistant', data.answer);
      playTTS(data.answer, data.lang);
    }
    function getSelectedLang() {
      const sel = document.getElementById('lang-select');
      return sel ? sel.value : 'en';
    }
    async function sendMessage() {
      const text = input.value.trim();
      if (!text) return;
      const lang = getSelectedLang();
      appendMessage('You', text);
      input.value = '';
      const resp = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, lang })
      });
      const data = await resp.json();
      appendMessage('Assistant', data.answer);
      playTTS(data.answer, lang);
    }

    let lastAudioUrl = null;
    async function playTTS(text, lang) {
      const resp = await fetch('/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, lang })
      });
      const blob = await resp.blob();
      if (lastAudioUrl) {
        URL.revokeObjectURL(lastAudioUrl);
      }
      const url = URL.createObjectURL(blob);
      lastAudioUrl = url;
      audio.pause();
      audio.currentTime = 0;
      audio.src = url;
      audio.load();
      // Play as soon as possible, with multiple fallbacks
      const playPromise = audio.play();
      if (playPromise !== undefined) {
        playPromise.catch(() => {
          audio.oncanplaythrough = function() {
            audio.play();
          };
        });
      } else {
        audio.oncanplaythrough = function() {
          audio.play();
        };
      }
      // Extra fallback in case oncanplaythrough doesn't fire
      setTimeout(() => { if (audio.paused) audio.play(); }, 500);
    }

    // Wake word detection and mic logic
    function setupWakeWord() {
      if (!('webkitSpeechRecognition' in window)) {
        wakeStatusMsg.textContent = '(speech recognition not supported)';
        return;
      }
      const lang = getSelectedLang();
      let recogLang = lang === 'pt' ? 'pt-BR' : 'en-US';
      recognition = new webkitSpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = recogLang;
      recognition.onstart = () => {
        isListening = true;
        wakeStatusMsg.textContent = '(listening...)';
      };
      recognition.onend = () => {
        isListening = false;
        if (!wakeActive) setTimeout(() => recognition.start(), 500); // restart if not actively using mic
      };
      recognition.onresult = function(event) {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          transcript += event.results[i][0].transcript;
        }
        if (!wakeActive && /hey rig/i.test(transcript)) {
          wakeActive = true;
          wakeIndicator.classList.add('active');
          wakeStatusMsg.textContent = '(wake word detected!)';
          recognition.stop();
          setTimeout(() => startMic(true), 400); // start mic for command
        }
      };
      recognition.start();
    }

    function activateMic() {
      if (!('webkitSpeechRecognition' in window)) {
        alert('Speech recognition not supported in this browser.');
        return;
      }
      // Stop background recognition if running
      if (recognition && isListening) {
        recognition.onend = null;
        recognition.stop();
      }
      micBtn.classList.add('listening');
      const lang = getSelectedLang();
      let recogLang = lang === 'pt' ? 'pt-BR' : 'en-US';
      let micRecognition = new webkitSpeechRecognition();
      micRecognition.lang = recogLang;
      micRecognition.interimResults = false;
      micRecognition.onresult = function(event) {
        input.value = event.results[0][0].transcript;
        sendMessage();
      };
      micRecognition.onend = function() {
        micBtn.classList.remove('listening');
        // After manual mic, resume background wake word recognition
        if (recognition) {
          wakeActive = false;
          wakeIndicator.classList.remove('active');
          wakeStatusMsg.textContent = '(listening...)';
          setTimeout(() => recognition.start(), 500);
        }
      };
      micRecognition.start();
    }

    // Mic button now works on click only (no need to hold)

    // Enter key to send
    input.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') sendMessage();
    });

    // Start wake word detection on load
    window.onload = setupWakeWord;
  </script>
</body>
</html>
'''

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    message = data.get("message", "")
    lang = data.get("lang", None)
    if lang not in ['en', 'pt']:
        try:
            lang = detect(message)
        except Exception:
            lang = 'en'
    if lang == 'pt':
        response = assistant.rag_chain_pt.invoke(message)
    else:
        response = assistant.rag_chain_en.invoke(message)
    return jsonify({"answer": response, "lang": lang})

@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "en")
    from gtts import gTTS
    import tempfile
    tts = gTTS(text=text, lang=lang if lang in ['en', 'pt'] else 'en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tf:
        tts.save(tf.name)
        tf.seek(0)
        audio_bytes = tf.read()
    os.remove(tf.name)
    return send_file(io.BytesIO(audio_bytes), mimetype="audio/mpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
     