import os
import struct
import pvporcupine
import pyaudio
import speech_recognition as sr
import pyttsx3
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# --- Configuration ---
PICOVOICE_ACCESS_KEY = "+ybVVD7Rrm2HwBqqXD8jm8Z39R3un3tlExjpdmzofTZv5Z9OhpJrGQ=="  # <-- IMPORTANT: Replace with your key
WAKE_WORD_MODEL_PATH = None
VOSK_MODEL_PATH = "Oil Rig AI Safety Assistant\Vosk-model-small-en-us-0.15"  # <-- IMPORTANT: Path to your downloaded Vosk model folder

VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3-8b-8192"

os.environ["GROQ_API_KEY"] = "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" # <-- IMPORTANT: Replace with your Groq key

class VoiceAssistant:
    def __init__(self):
        print("Initializing AI Assistant...")
        self.tts_engine = self._initialize_tts()
        self.rag_chain = self._initialize_rag_chain()
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

    def _initialize_rag_chain(self):
        """Creates the complete RAG chain for answering questions."""
        print("🔍 Loading vector store...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})

        prompt_template = """
        **Safety Instruction**
        You are a voice AI assistant for oil rig operators. Your task is to provide clear, concise, and accurate safety information verbally.
        Answer the user's question based ONLY on the following context from the company's safety documents.
        If the information is not in the context, state clearly: "The safety documents do not provide information on this specific action."
        Keep your answers direct and to the point for voice delivery.
        Do not add any information that is not from the provided context, if no helpfull information found, redirect the user to look for a human expert in the field.

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

    def speak(self, text):
        """Converts text to speech."""
        print(f"\n💡 AI Assistant says: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def listen_for_command(self):
        """Listens for a spoken command after the wake word is detected."""
        with self.microphone as source:
            print("\n🎤 Listening for your question...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                # Use Vosk for offline recognition
                command = self.recognizer.recognize_vosk(audio, model_path=VOSK_MODEL_PATH)
                print(f"🤔 Operator asks: {command}")
                return command
            except sr.WaitTimeoutError:
                self.speak("I didn't hear a question. Going back to sleep.")
                return ""
            except sr.UnknownValueError:
                self.speak("I couldn't understand that. Please try again.")
                return ""
            except Exception as e:
                print(f"An error occurred during speech recognition: {e}")
                self.speak("There was an error processing your speech.")
                return ""

    def process_command(self, command):
        """Processes the command through the RAG chain and speaks the result."""
        if command:
            self.speak("Analyzing the request...")
            response = self.rag_chain.invoke(command)
            self.speak(response)
        
    def run(self):
        """Main loop to listen for the wake word and process commands."""
        porcupine = None
        pa = None
        audio_stream = None
        try:
            # For "Hey Rig", you need a custom model file (.ppn) from the Picovoice Console.
            # For this example, we use the built-in "Hey Porcupine".
            # To create "Hey Rig", go to the Picovoice Console, train it, and download the .ppn file.
            # Then set WAKE_WORD_MODEL_PATH = 'path/to/your/hey_rig.ppn'
            
            porcupine = pvporcupine.create(
                access_key=PICOVOICE_ACCESS_KEY,
                # Use a built-in keyword if you don't have a custom one
                keyword_paths=[pvporcupine.KEYWORD_PATHS['porcupine']] 
                # Or use your custom one: keyword_paths=['path/to/hey_rig.ppn']
            )

            pa = pyaudio.PyAudio()
            audio_stream = pa.open(
                rate=porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=porcupine.frame_length
            )
            
            self.speak("AI Assistant is online. Say 'Hey Rig' to activate.")
            print("\n👂 Listening for wake word ('Hey Rig')...")

            while True:
                pcm = audio_stream.read(porcupine.frame_length)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                keyword_index = porcupine.process(pcm)
                if keyword_index >= 0:
                    print("\n--- Wake word detected! ---")
                    self.speak("Yes? How can I help?")
                    command = self.listen_for_command()
                    self.process_command(command)
                    print("\n👂 Listening for wake word ('Hey Rig')...")

        except pvporcupine.PorcupineActivationError as e:
            print(f"Porcupine activation error: {e}")
            self.speak("There was an issue with the wake word engine's access key.")
        #except Exception as e:
         #   print(f"An unexpected error occurred: {e}")
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
