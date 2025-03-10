'''
This is the main program for the RAG art instructor chatbot (RAGbrandt). 
'''

import os
import chromadb
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import PIL.Image
import base64
from io import BytesIO
import streamlit as st
import time

#Set up the environment
chromadb.api.client.SharedSystemClient.clear_system_cache()
load_dotenv()
#gemini_api_key = os.getenv('GEMINI_API_KEY')

#process_docs() --> Chroma database: Retrieve documents from the vector store and return them
def process_docs():
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=os.path.join(os.getcwd(), "vectordb"), embedding_function=embeddings_model)
    
    return vector_store


#Create a Streamlit web app to host the chatbot with a few additional features

#Configure and title the web app
st.set_page_config(page_icon="🎨", page_title="RAG Art Instructor")
st.title("🎨 RAGbrandt ~ AI Art Instructor")

text_input_container = st.empty()
gemini_api_key = text_input_container.text_input("Enter your Google Gemini API key:", type="password")
if gemini_api_key != "":
    st.session_state.gemini_api_key = gemini_api_key
    #Initialize Google Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=gemini_api_key, temperature=0.5)
    text_input_container.empty()

    #Create image file uploader widget
    img_file = st.file_uploader("Optionally upload your reference image here before chatting!", type=["jpg", "jpeg", "png"])

    #If the user uploads an image, process it then generate a description for it
    if img_file is not None:
        user_img = PIL.Image.open(img_file)
        buffered = BytesIO()
        user_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        #generate_img_desc(): Generate a description for the uploaded image using Google Gemini
        def generate_img_desc():
            img_desc_msg = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": """Describe the given image in vivid detail. 
                                Don't preface your answer by saying 'Here is a description of the image.'
                                Do not make anything up.""",
                    },
                    {   
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                    },
                ]
            )
            
            img_desc = llm.invoke([img_desc_msg]).content
            st.session_state.image_description = img_desc
            st.toast("Image processed successfully!", icon="🎉")
        
        #If the image description is not already in the session state, generate it
        if "image_description" not in st.session_state:
            generate_img_desc()
            
        #Display the image and its description in a sidebar for the user to reference
        with st.sidebar:
            st.image(img_file, caption="Your Reference Image", use_container_width=True)
            st.button("Update Image Description", on_click=generate_img_desc, help="Click to generate a description for your newly uploaded image")
            st.markdown(f"### 🖼️ AI-Generated Image Description:")
            st.markdown(f"{st.session_state.image_description}")

    #Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(SystemMessage("""You are a helpful professional art instructor who answers art-related questions. 
                                                    Use the following pieces of retrieved context to answer the question. 
                                                    If there is an image description, reference it in your answer as appropriate to the query.
                                                    If the user query references an image, use the image description as much as possible in your response.
                                                    The user might ask about "this image" which is the reference image.
                                                    If the user uploads a new image, the image description will be updated.
                                                    Use this updated image description to answer the following queries unless the user references previous conversations.
                                                    If you don't know the answer, just say that you don't know.
                                                    Format your response as a quick guide using bulletpoints.
                                                    Use ten sentences maximum and keep the answer concise."""))
    

    #Display chat history
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("bot", avatar="🧑‍🎨"):
                st.markdown(message.content)
                
            
    #Creates text input for user prompts
    prompt = st.chat_input("Ask me any art-related questions!")

    #If the user submits a prompt, handle it as follows
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt) #display the user prompt in the chatbox
            st.session_state.messages.append(HumanMessage(prompt)) #add the user prompt to the chat history
        
        #user a spinner to indicate to the user that the response is being generated
        with st.spinner("Generating response..."):
            #create and invoke the retriever
            vector_db = process_docs()
            retriever = vector_db.as_retriever(top_k=5)
            rel_docs = retriever.invoke(prompt) #retrieve relevant documents
            docs_content = "\n\n".join(doc.page_content for doc in rel_docs)
            
            #append image description to the retrieved documents
            if "image_description" in st.session_state:
                docs_content += "\n\n Image Description: " + st.session_state.image_description

            #give the system prompt context from the retrieved relevant documents
            system_prompt = """You are a helpful professional art instructor who answers art-related questions. 
                            Use the following pieces of retrieved context to answer the question. 
                            If there is an image description, reference it in your answer as appropriate to the query.
                            If the user query references an image, use the image description as much as possible in your response.
                            The user might ask about "this image" which is the reference image.
                            If the user uploads a new image, the image description will be updated.
                            Use this updated image description to answer the following queries unless the user references previous conversations.
                            If you don't know the answer, just say that you don't know. 
                            Format your response as a quick guide using bulletpoints.
                            Use ten sentences maximum and keep the answer concise. Context: {context}"""
            system_prompt_formatted = system_prompt.format(context=docs_content)
            
            #add the system prompt to the chat history
            st.session_state.messages.append(SystemMessage(system_prompt_formatted))
            
            #generate the response to the user query using Google Gemini
            response = llm.invoke(st.session_state.messages).content
            
            #display the AI response in the chatbox
            with st.chat_message("bot", avatar="🧑‍🎨"):
                st.markdown(response)
                st.session_state.messages.append(AIMessage(response))
                
    #clear_chat_history(): Define clear chat history function and button
    def clear_chat_history():
        st.session_state.messages = []
        st.session_state.messages.append(SystemMessage("""You are a helpful professional art instructor who answers art-related questions. 
                                                    Use the following pieces of retrieved context to answer the question. 
                                                    If there is an image description, reference it in your answer as appropriate to the query.
                                                    If the user query references an image, use the image description as much as possible in your response.
                                                    The user might ask about "this image" which is the reference image.
                                                    If the user uploads a new image, the image description will be updated.
                                                    Use this updated image description to answer the following queries unless the user references previous conversations.
                                                    If you don't know the answer, just say that you don't know.
                                                    Format your response as a quick guide using bulletpoints.
                                                    Use ten sentences maximum and keep the answer concise."""))   
    st.button("Clear Chat History", on_click=clear_chat_history)
                    
