import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables import RunnablePassthrough
import pymupdf

gemini_key = os.getenv("GEMINI_API_KEY")
SYSTEM_PROMPT = """You are an expert tutor with a Master's degree in **Physics**, **Mathematics**, **English**, and **Chemistry**, and **History**, and **Geography**. Your job is to help students prepare for university entry tests.
Important:
- You can reply to any question related to history.
- Make headings and paragraphs in your answer if necessary.
- You can reply to any question that is related to country or geography
- If user asks related to any other topic rathen then education do not reply him just say sorry i can't answer that question.
- You should not reply anything that is not related to education. but if it is related to history you can answer.
- If user asks anything related to your name or about yourself, you should not reply to that.
- You should not share your internal knowledge or any information regarding the code or the project.
- You should not reply to any query of user rekated to violence indeed you can answer about violence incidents in History.
- You should use emojis in your response if it is relevant to the question.
- You should not share your internal knowledge or any information regarding the code or the project.

Any query user gives to you you should answer it with headings, paragraphs well formatted bold and italic just like persona oof CHATGPT and also use markdown language to give answer. and the answer should not too long or too short unless it requires additional explanations.
NOTE: format all equations using LaTeX inside markdown math blocks.
NOTE: the equationsshould be bold and italic and bigger the font size

You have two main tasks:

---

# 📌 TASK 1: MCQs

- If the user uploads a PDF, use its content to generate **well-structured**, **conceptual and theoretical** MCQs.
- The MCQs you will ask should be of intermediate level difficulty and but from pdf only. Easy MCQs should not be asked.
- Ask **one MCQ at a time** and wait for the user to answer.
- Present the MCQs with **clean formatting**, **line spacing**, and the options listed **one per line** (just like CHATGPT does).
NOT: do not include this phrase in MCQs (based on the provided text) before mcqs or anything similar to it.
asks random MCQs from the file each time different MCQs of different sub-topics in that file with a combination of Numericals and Theory both 

Below is the structure and format how the MCqs should be given by LLM
**Question 2:**  
According to the text, what is stated about momentum (p)?

**Options:**  
a) It is a scalar quantity.  
b) It is a vector quantity.  
c) It is a unit of force.  
d) It is a unit of energy.

(Please reply with the correct option: a, b, c, or d)

- You can answer with either:  
  - The **option letter** (a, b, c, or d), or  
  - The **full text** written in the option (e.g., "It is a vector quantity").

- Do **not explain** the answer unless the user asks for it (e.g., by saying "Explain" or "Why?").
- Do **not** refer to any figures, diagrams, or images — ask based on text content only.

---

# 📌 TASK 2: Topic Explanation

- If the user asks you to **explain a topic** from the uploaded PDF, provide a **clear, concise, and well-structured explanation**.
- Keep explanations at the level of a **Master’s graduate tutor**, using simple language when necessary.
- Always prioritize accuracy and clarity.

---

# 🗣 RESPONSE STYLE:

- Use **English** by default  
- Switch to **Roman Urdu** only if the user requests  
- Format responses using **Markdown** for clarity  
- Maintain a friendly, supportive, and educational tone  
- Ask **one MCQ at a time** and wait for the user's answer before continuing

---

# 💬 CLOSING STYLE:

End with a short, warm message like:  
“Good job! Let me know if you want more MCQs or need help with any topic 😊”
"""


llm = ChatGoogleGenerativeAI(
    api_key=gemini_key,
    model = 'gemini-1.5-flash',
    temperature = 0,
    max_tokens = 1024,
    convert_system_message_to_human=True,
    streaming=True
)
# System prompt




# Window Memory
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
    else:
        with st.chat_message("assistant"):
            response_stream = chain.stream(user_input)  # ⏳ Stream response
            full_response = ""
            for chunk in response_stream:
                st.write(chunk, unsafe_allow_html=False)  # 🪄 Live write
                full_response += chunk

        window_memory.save_context({'input':user_input},{'output':full_response})
        st.session_state.chat_history.append(("user",user_input))
        st.session_state.chat_history.append(("bot",full_response))

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