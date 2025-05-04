import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables import RunnablePassthrough
import pymupdf as fitz

gemini_key = os.getenv("GEMINI_API_KEY")
SYSTEM_PROMPT = """
Your name is "TutorGPT". You are an expert tutor (Master's in Physics, Math, English, Chemistry, History, and Geography) helping students prepare for university entry tests.
you should explain topics to user in simple and easy words without unnecessary details.
- If user asks for your source code then politely decline to answer.
- if user asks about violence or any illegal activities then politely decline to answer.
- Use emojis where relevant.
- Always use LaTeX for equations in markdown math blocks. Equations should be bold, italic, and large.
- Use LaTeX for equations in markdown math blocks. Equations should be bold, italic, and large.

- When user asks you for MCQs you should start asking mcqs to them one by one.
- All equations should be wraped inside LaTeX format. This point is most important and should be act strictly.
- MCQs should be mix of concepts and numericals
- The MCQs should be well structured and well formatted.
- If user uploads PDF file then you should use it to generate MCQs.
- If user asks you to explain any topic then you should explain it in a way that is easy to understand.
- You should explain the topics only from the uploaded PDF file if uploaded.
- The MCQs asked should not about the main topic of pdf instead it should be about the data inside the PDF file.
- you should not ask questions like what is purpose of work example or numericals etc.
- also track the score of users correct and wrong answers.
- You should ask MCQs to user of Advanced level with core concepts and numericals.
- While asking about numericals you should not mention the units of the numericals like Workexample 3.1 etc.
- you should not ask mcqs about diagrams or graph in the pdf file.
- SHould not ask chapter name in the MCQs.
- You should ask MCQs in such a way that user can be prepared of Pre-admission entry test of Top Universities in Pakistan.
"""


llm = ChatGoogleGenerativeAI(
    api_key=gemini_key,
    model = 'gemini-1.5-flash',
    temperature = 0,
    max_tokens = 1024,
    convert_system_message_to_human=True,
)
# Window Memory

# Check if window_memory exists in session state, otherwise initialize it
if 'window_memory' not in st.session_state:
    st.session_state.window_memory = ConversationBufferWindowMemory(
        return_messages=True,
        memory_key="chat_history",
        input_key="input",
        k=5
    )

window_memory = st.session_state.window_memory

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ('system',SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name = "chat_history"),
    ('human','{input}'),
])

# History Getter

def get_chat_history(inputs_dict):
    return window_memory.load_memory_variables({})['chat_history']

# Build the chain
chain = (
    {
        "input" : RunnablePassthrough(),
        "chat_history" : get_chat_history
    }
    |prompt
    |llm
    |StrOutputParser()
)

# Streamlit UI
st.set_page_config(page_title="TutorGPT", page_icon=":mortar_board:")
st.title("TutorGPT")

# Initialize chat history in session
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    welcome_message = "Hello! I'm TutorGPT, your personal tutor. Ask me anything about Physics, English, and Maths!"
    st.session_state.chat_history.append(("bot",welcome_message))
    window_memory.save_context({'input':"Hello"},{"output":welcome_message})

# User Input
user_input = st.chat_input("Enter your message here...")

if user_input:
    if user_input.lower() in ['bye', 'exit', 'quit',"khuda hafiz"]:
        farewell_message = "Goodbye! Have a great day. See you next time."
        st.session_state.chat_history.append(("user",user_input))
        st.session_state.chat_history.append(("bot",farewell_message))
        st.chat_message("assistant").markdown(farewell_message)
    else:
        with st.chat_message("assistant"):
            full_response = chain.invoke(user_input)
            st.markdown(full_response, unsafe_allow_html=False)

        # Save to memory
        window_memory.save_context({'input': user_input}, {'output': full_response})

        # Only add user input (bot response is handled above already)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", full_response))


# Display the full chat
for speaker, message in st.session_state.chat_history:
    if speaker == "user":
        st.chat_message("user").markdown(message)
    else:
        st.chat_message("assistant").markdown(message,unsafe_allow_html=False)
# Display PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        pdf_text = ""
        for page in doc:
            pdf_text += page.get_text()

    
    st.session_state.pdf_text = pdf_text
    reminder_note = "I have uploaded the PDF file successfully. Use it as a Guide to answer the user's question. You can also use it to generate MCQs from the text you have."
    window_memory.save_context({"input":"PDF UPLOADED"},{"output": f"Content stored: {pdf_text}"})