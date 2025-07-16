from gtts import gTTS
import tempfile
import os
import base64
import json

def handler(request):
    if request.method == "OPTIONS":
        return ("", 204, {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "POST, OPTIONS", "Access-Control-Allow-Headers": "*"})
    try:
        data = request.get_json() if hasattr(request, 'get_json') else json.loads(request.body.decode())
        text = data.get("text", "")
        lang = data.get("lang", "en")
        tts = gTTS(text=text, lang=lang if lang in ['en', 'pt'] else 'en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tf:
            tts.save(tf.name)
            tf.seek(0)
            audio_bytes = tf.read()
        os.remove(tf.name)
        # Return as base64 for Vercel compatibility
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        return ({"audio": audio_b64, "mimetype": "audio/mpeg"}, 200, {"Access-Control-Allow-Origin": "*"})
    except Exception as e:
        return ({"error": str(e)}, 500, {"Access-Control-Allow-Origin": "*"})
