import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# --- Configuration ---
VECTOR_STORE_PATH = "Oil Rig AI Safety Assistant\Vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3-8b-8192" # A fast model available via Groq

# --- Set up API Key ---
# Best practice is to set this as an environment variable
# For example, in your terminal: export GROQ_API_KEY='your_api_key_here'
os.environ["GROQ_API_KEY"] = "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" # Replace with your key

def create_rag_chain():
    """Creates the complete RAG chain for answering questions."""
    
    # 1. Load the existing vector store
    print("🔍 Loading vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={'k': 3}) # Retrieve top 3 relevant chunks

    # 2. Define the prompt template
    # This guides the LLM to answer based only on the provided context.
    prompt_template = """
    **Safety Instruction**
    You are an AI assistant for oil rig operators. Your task is to provide clear, concise, and accurate safety information.
    Answer the user's question based ONLY on the following context from the company's safety documents. If the information is not in the context, state clearly: "The safety documents do not provide information on this specific action."
    Do not add any information that is not from the provided context. Redirect the user to look for a human expert in the field.

    **Context from Safety Documents:**
    {context}

    **User's Question:**
    {question}

    **Answer:**
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # 3. Initialize the LLM
    llm = ChatGroq(model=LLM_MODEL, temperature=0) # Temperature 0 for factual, non-creative answers

    # 4. Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ AI Assistant is ready.")
    return rag_chain

def ask_question(chain, question):
    """Asks a question to the RAG chain and prints the response."""
    print("\n🤔 Operator asks:", question)
    print("\n💡 AI Assistant says:")
    response = chain.invoke(question)
    print(response)

# --- Main execution block ---
if __name__ == "__main__":
    # Create the chain once
    ai_chain = create_rag_chain()

    # --- Example Questions ---
    ask_question(ai_chain, "What are the main risks associated with hot work?")
    ask_question(ai_chain, "What is the procedure for a manual handling task?")
    ask_question(ai_chain, "How do I operate the new drilling equipment model XYZ?")

    # --- Interactive Loop ---
    print("\n--- Enter your questions below (type 'exit' to quit) ---")
    while True:
        user_question = input("Your action/question: ")
        if user_question.lower() == 'exit':
            break
        ask_question(ai_chain, user_question)