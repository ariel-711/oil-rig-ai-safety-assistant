from ai_assistant_voice_goodvoice_02 import VoiceAssistant
from langdetect import detect
import json

def handler(request):
    if request.method == "OPTIONS":
        return ("", 204, {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "POST, OPTIONS", "Access-Control-Allow-Headers": "*"})
    try:
        data = request.get_json() if hasattr(request, 'get_json') else json.loads(request.body.decode())
        message = data.get("message", "")
        lang = data.get("lang", None)
        assistant = VoiceAssistant()
        if lang not in ['en', 'pt']:
            try:
                lang = detect(message)
            except Exception:
                lang = 'en'
        if lang == 'pt':
            response = assistant.rag_chain_pt.invoke(message)
        else:
            response = assistant.rag_chain_en.invoke(message)
        return ({"answer": response, "lang": lang}, 200, {"Access-Control-Allow-Origin": "*"})
    except Exception as e:
        return ({"error": str(e)}, 500, {"Access-Control-Allow-Origin": "*"})
