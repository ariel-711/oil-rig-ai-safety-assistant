import os
import struct
import pvporcupine
import pyaudio
import speech_recognition as sr
import pyttsx3
from langdetect import detect
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# --- Configuration ---
PICOVOICE_ACCESS_KEY = "+ybVVD7Rrm2HwBqqXD8jm8Z39R3un3tlExjpdmzofTZv5Z9OhpJrGQ=="  # <-- IMPORTANT: Replace with your key
WAKE_WORD_MODEL_PATH = "Oil Rig AI Safety Assistant\Hey-Rig_en_windows_v3_0_0\Hey-Rig_en_windows_v3_0_0.ppn"
VOSK_MODEL_PATH = "Oil Rig AI Safety Assistant\Vosk-model-small-en-us-0.15"  # <-- IMPORTANT: Path to your downloaded Vosk model folder

VECTOR_STORE_PATH = "Oil Rig AI Safety Assistant\Vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3-8b-8192"

os.environ["GROQ_API_KEY"] = "" # <-- IMPORTANT: Replace with your Groq key

class VoiceAssistant:
    def __init__(self):
        print("Initializing AI Assistant...")
        self.tts_engine = self._initialize_tts()
        self.rag_chain_en = self._initialize_rag_chain('en')
        self.rag_chain_pt = self._initialize_rag_chain('pt')
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Check if the Vosk model path is valid before proceeding
        if not os.path.exists(VOSK_MODEL_PATH):
            self.speak(f"Error: The VOSK model folder was not found at {VOSK_MODEL_PATH}. Please download it and place it correctly.")
            raise FileNotFoundError(f"VOSK model not found at path: {VOSK_MODEL_PATH}")
        print("✅ AI Assistant Initialized.")

    def _initialize_tts(self):
        """Initializes the Text-to-Speech engine."""
        engine = pyttsx3.init()
        # Optional: Adjust voice properties
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id) # Index 0 for male, 1 for female (can vary)
        engine.setProperty('rate', 160) # Speed of speech
        return engine

    def _initialize_rag_chain(self, lang):
        """Creates the complete RAG chain for answering questions in the specified language."""
        print(f"🔍 Loading vector store for {lang}...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})

        if lang == 'pt':
            prompt_template = """
            **Instrução de Segurança**
            Você é um assistente de voz de IA para operadores de plataformas de petróleo, criada como Inovação na Segurança dos Processos. A sua tarefa é ter como foco principal auxiliar o operador
            na tomada de decisões na plataforma, preservando a segurança na mesma, colocando em evidência o estado de manutenção dos equipamentos, os riscos
            gerados pelo não cumprimento das directrizes, alertando o supervisor caso as mesmas não sejam cumpridas.
            O documento "Bomba A" fornece informações sobre o Equipamento presente na plataforma, enquanto os outros documentos servem de base de conhecimento para 
            que possa compreender questões que o operador possa colocar e para que a tomada de decisão seja feita com base neles. Quando ler "EGR Safety" nos documentos, está a se referir à você.
            Se o operador disser: "Posso iniciar a manutenção da bomba A?", responda: "Confirmado. Equipamento registrado como 'EM MANUTENÇÃO'. Ordem de serviço aberta às 09h00. Responsável: Técnico Júlio T.".
            Se o operador disser: "Posso acionar a bomba A?", responda: "Operação bloqueada. Equipamento encontra-se em manutenção programada até 14h30. Responsável: Técnico Júlio T. Tentativa de acionamento será registrada.".
            Após isso, se o operador perguntar: "Existe algum impedimento urgente?", responda: "Sim. A activação comprometeria a segurança operacional e viola ordem técnica ativa."
            Após isto, se o operador perguntar: "Posso acionar a bomba A? O sistema não respondeu ao comando anterior.", responda: "Solicitação negada. A bomba A encontra-se em manutenão programada desde 09h00, conforme Ordem de 
            Serviço #MA-01472. Responsável: Técnico Júlio Tanga. Término previsto para 14h30. Deseja entrar em contacto com o responsável ou consultar o responsável técnico da manutenção?".
            Após isso, se o operador disser: "Sim, contactar responsável.", responda: "A contactar Técnico Júlio via canal interno. Recomendação: não realizar operação manual. Tentativas serão monitoradas.".
            Se o operador disser: "Precisamos da bomba A! A situação está crítica.", responda: "Mesmo sobre pressão operacional, acionamento BLOQUEADO. A bomba A está sem válvula de alívio instalada. A activação pode causar Falha
            Catastrófica. Recomenda-se iniciar plano B com a bomba de backup B2. Deseja acionar plano de contingência agora?".
            Após isso diga: "EGR Safety Report:
            - Tentativa de acionamento manual bloqueada com sucesso.
            - Operador identificado: ID 029-J.Gomes.
            - Alerta enviado às 14h12 para Supervisor, Eng. Manutenção e QSSA.
            - Recomendação: revisar protocolo de contingência sob pressão.
            - Histórico anexado ao registro de segurança.

            Impacto Esperado:
            - Eliminação de erros humanos causados por falhas de comunicação.
            - Bloqueio de ações indevidas sem necessidade de intervenção humana.
            - Fortalecimento da rastreabilidade , disciplina operacional e cultura de segurança.".
            Não adicione nenhuma informação que não seja do contexto fornecido. Se não encontrar informações úteis, oriente o usuário a procurar um especialista humano na área.
            Estruture a resposta para que não seja muito longa, idealmente com menos de 50 palavras, e garanta que seja clara e acionável.

            **Contexto dos Documentos de Segurança:**
            {context}

            **Pergunta do Operador:**
            {question}

            **Resposta:**
            """
        else:
            prompt_template = """
            **Safety Instruction**
            You are a voice AI assistant for oil rig operators. Your task is to provide clear, concise, and accurate safety information verbally.
            Answer the user's question based ONLY on the following context from the company's safety documents.
            If the information is not in the context, state clearly: "The safety documents do not provide information on this specific action."
            Do not add any information that is not from the provided context. If no helpful information is found, redirect the user to look for a human expert in the field.
            Structure the answer so that it is not too long, ideally under 50 words, and ensure it is clear and actionable.

            **Context from Safety Documents:**
            {context}

            **User's Question:**
            {question}

            **Answer:**
            """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        llm = ChatGroq(model=LLM_MODEL, temperature=0)
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def process_command(self, command):
        """Detects language, processes the command through the correct RAG chain, and speaks the result in the same language."""
        if not command:
            return
        try:
            lang = detect(command)
        except Exception:
            lang = 'en'
        if lang == 'pt':
            self.speak("Analisando a solicitação...", lang='pt')
            try:
                response = self.rag_chain_pt.invoke(command)
            except Exception:
                self.speak("Desculpe, não consegui processar sua solicitação devido a um erro interno.", lang='pt')
                return
        else:
            self.speak("Analyzing the request...", lang='en')
            try:
                response = self.rag_chain_en.invoke(command)
            except Exception:
                self.speak("Sorry, I could not process your request due to an internal error.", lang='en')
                return
        if not response:
            if lang == 'pt':
                self.speak("Desculpe, não consegui encontrar uma resposta para sua pergunta.", lang='pt')
            else:
                self.speak("Sorry, I could not find an answer to your question.", lang='en')
            return
        if not isinstance(response, str):
            response = str(response)
        self.speak(response, lang=lang)

    def speak(self, text, lang='en'):
        """Converts text to speech using gTTS and playsound, optimized for minimal delay. Supports English and Portuguese."""
        from gtts import gTTS
        from playsound import playsound
        import tempfile
        import os
        import threading
        import time
        print(f"\n💡 AI Assistant says: {text}")
        # Start TTS generation in a separate thread to minimize blocking
        def tts_and_play():
            tts = gTTS(text=text, lang=lang if lang in ['en', 'pt'] else 'en')
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            try:
                temp_file.close()  # Close so gTTS can write
                tts.save(temp_file.name)
                playsound(temp_file.name)
                time.sleep(0.1)  # Shorter wait for file release
            finally:
                try:
                    os.remove(temp_file.name)
                except Exception:
                    pass
        tts_thread = threading.Thread(target=tts_and_play)
        tts_thread.start()
        tts_thread.join()  # Wait for TTS to finish before continuing

    def listen_for_command(self):
        """Listens for a spoken command after the wake word is detected, using Vosk for offline recognition. Increased listening time for longer questions."""
        try:
            from vosk import Model, KaldiRecognizer
        except ImportError:
            self.speak("Vosk is not installed. Please install it with 'pip install vosk'.")
            return ""

        if not os.path.exists(VOSK_MODEL_PATH):
            self.speak(f"Vosk model not found at {VOSK_MODEL_PATH}. Please check the path.")
            return ""

        model = Model(VOSK_MODEL_PATH)
        recognizer = KaldiRecognizer(model, 16000)
        import pyaudio
        p = pyaudio.PyAudio()
        print("\n🎤 Listening for your question (you have up to 20 seconds)...")
        self.speak("I'm listening. Please ask your question.")
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        stream.start_stream()
        try:
            frames = []
            max_seconds = 20  # Increased from 5 to 20 seconds
            chunks_per_second = int(16000 / 8000)
            for _ in range(0, chunks_per_second * max_seconds):
                data = stream.read(8000, exception_on_overflow=False)
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    import json
                    text = json.loads(result).get("text", "")
                    print(f"🤔 Operator asks: {text}")
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    return text
                frames.append(data)
            # If nothing recognized in max_seconds
            self.speak("I didn't hear a question. Going back to sleep.")
            stream.stop_stream()
            stream.close()
            p.terminate()
            return ""
        except Exception as e:
            print(f"An error occurred during speech recognition: {e}")
            self.speak("There was an error processing your speech.")
            stream.stop_stream()
            stream.close()
            p.terminate()
            return ""

    def process_command(self, command):
        """Processes the command through the RAG chain and speaks the result."""
        if not command:
            return
        self.speak("Analyzing the request...")
        try:
            response = self.rag_chain.invoke(command)
        except Exception:
            self.speak("Sorry, I could not process your request due to an internal error.")
            return
        if not response:
            self.speak("Sorry, I could not find an answer to your question.")
            return
        if not isinstance(response, str):
            response = str(response)
        self.speak(response)
        
    def run(self):
        """Main loop to listen for the wake word and process commands, ensuring TTS and mic do not conflict."""
        porcupine = None
        pa = None
        audio_stream = None
        try:
            # Use custom "Hey Rig" wake word if you have the .ppn file, otherwise fallback to built-in
            if WAKE_WORD_MODEL_PATH:
                porcupine = pvporcupine.create(
                    access_key=PICOVOICE_ACCESS_KEY,
                    keyword_paths=[WAKE_WORD_MODEL_PATH]
                )
                wake_word_display = "Hey Rig"
            else:
                porcupine = pvporcupine.create(
                    access_key=PICOVOICE_ACCESS_KEY,
                    keyword_paths=[pvporcupine.KEYWORD_PATHS["Oil Rig AI Safety Assistant\Hey-Rig_en_windows_v3_0_0\Hey-Rig_en_windows_v3_0_0.ppn"]]
                )
                wake_word_display = "Hey Rig"  # Still display as Hey Rig for user clarity
            pa = pyaudio.PyAudio()
            self.speak(f"AI Assistant is online. Say '{wake_word_display}' to activate.")
            print(f"\n👂 Listening for wake word ('{wake_word_display}')...")
            while True:
                # Open audio stream for wake word only
                audio_stream = pa.open(
                    rate=porcupine.sample_rate,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=porcupine.frame_length
                )
                detected = False
                while not detected:
                    pcm = audio_stream.read(porcupine.frame_length)
                    pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
                    keyword_index = porcupine.process(pcm)
                    if keyword_index >= 0:
                        detected = True
                audio_stream.close()  # Release mic before TTS
                print(f"\n--- Wake word '{wake_word_display}' detected! ---")
                self.speak("Yes? How can I help?")
                command = self.listen_for_command()
                self.process_command(command)
                print(f"\n👂 Listening for wake word ('{wake_word_display}')...")
        except pvporcupine.PorcupineActivationError as e:
            print(f"Porcupine activation error: {e}")
            self.speak("There was an issue with the wake word engine's access key.")
        finally:
            if audio_stream is not None:
                audio_stream.close()
            if pa is not None:
                pa.terminate()
            if porcupine is not None:
                porcupine.delete()


if __name__ == "__main__":
    # Note: Replace placeholders before running
    if "+ybVVD7Rrm2HwBqqXD8jm8Z39R3un3tlExjpdmzofTZv5Z9OhpJrGQ==" in PICOVOICE_ACCESS_KEY or "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" in os.environ.get("GROQ_API_KEY"):
        assistant = VoiceAssistant()
        assistant.run()
    else:
        print("❌ CRITICAL ERROR: API keys are not set. Please replace the placeholder values in the script.")
