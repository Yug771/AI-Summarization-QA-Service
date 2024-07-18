import streamlit as st
import sqlite3
import bcrypt
import re
import PyPDF2
from docx import Document
import io, threading
from summarization import AbstractiveSummarization, ExtractiveSummarization
from QA import Generate_embedding,Generate_answer

def init_db():
    conn = sqlite3.connect('summarization_app.db')
    c = conn.cursor()
    
    # users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password BLOB)''')
    
    # summary table
    c.execute('''CREATE TABLE IF NOT EXISTS summaries
                 (id INTEGER PRIMARY KEY,
                  input_text TEXT,
                  summary TEXT,
                  username TEXT,
                  user_id INTEGER,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    
    # QA table
    c.execute('''CREATE TABLE IF NOT EXISTS QuestionAnswer
                 (id INTEGER PRIMARY KEY,
                  question TEXT,
                  answer TEXT,
                  text TEXT,
                  username TEXT,
                  user_id INTEGER,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
     
    conn.commit()
    conn.close()

#-------------------Database add functions---------------------------#

def add_summary(input_text, summary, username, user_id):
    conn = sqlite3.connect('summarization_app.db')
    c = conn.cursor()
    
    try:
        # Use parameterized query to avoid SQL injection
        c.execute('''INSERT INTO summaries (input_text, summary, username, user_id)
                     VALUES (?, ?, ?, ?)''', (input_text, summary, username, user_id))
        conn.commit()
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

def add_qa(question, answer, text, username, user_id):
    conn = sqlite3.connect('summarization_app.db')
    c = conn.cursor()
    c.execute('''INSERT INTO QuestionAnswer (question, answer, text, username, user_id)
                 VALUES (?, ?, ?, ?, ?)''', (question, answer, text, username, user_id))
    conn.commit()
    conn.close()

def get_user_id(username):
    conn = sqlite3.connect('summarization_app.db')
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def get_summaries(username=None):
    conn = sqlite3.connect('summarization_app.db')
    c = conn.cursor()
    if username:
        c.execute("SELECT * FROM summaries WHERE username = ?", (username,))
    else:
        c.execute("SELECT * FROM summaries")
    results = c.fetchall()
    conn.close()
    return results

def get_qa(username=None):
    conn = sqlite3.connect('summarization_app.db')
    c = conn.cursor()
    if username:
        c.execute("SELECT * FROM qa WHERE username = ?", (username,))
    else:
        c.execute("SELECT * FROM QuestionAnswer")
    results = c.fetchall()
    conn.close()
    return results

def get_latest_user_text(username):
    conn = sqlite3.connect('summarization_app.db')
    c = conn.cursor()
    c.execute("""
        SELECT input_text 
        FROM summaries 
        WHERE username = ? 
        ORDER BY id DESC 
        LIMIT 1
    """, (username,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else ""

#-----------------------User----------------------------------#

def create_user(username, password):
    conn = sqlite3.connect('summarization_app.db')
    c = conn.cursor()
    try:
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        # hashed_password = hashed_password.decode('utf8')
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('summarization_app.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return bcrypt.checkpw(password.encode('utf-8'), result[0])
    return False

def username_exists(username):
    conn = sqlite3.connect('summarization_app.db')
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    return result is not None

def is_valid_username(username):
    return re.match(r'^[a-zA-Z0-9_]{3,20}$', username) is not None

def is_valid_password(password):
    return len(password) >= 8 and re.search(r'\d', password) and re.search(r'[A-Z]', password) and re.search(r'[a-z]', password)


#---------------------Generate summary function-----------------#

def abstractive_summarization(text, model):
    return f"Abstractive summary using {model}: " + text[:100] + "..."

def extractive_summarization(text, model):
    return f"Extractive summary using {model}: " + text[:100] + "..."

def answer_question(question):
    return f"Answer to '{question}': This is a placeholder answer based on the question."


#-----------------------------Supporting functions----------------------------------#

def read_file_content(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
        return "\n".join(page.extract_text() for page in pdf_reader.pages)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(io.BytesIO(uploaded_file.getvalue()))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    else:
        return "Unsupported file type"


#----------------------------Main code----------------------------------------#

# Initialize database
init_db()

# Set page config
st.set_page_config(layout="wide")

# Initialize session states
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'text_inserted' not in st.session_state:
    st.session_state.text_inserted = False
if 'current_text' not in st.session_state:
    st.session_state.current_text = ""
if 'current_summary' not in st.session_state:
    st.session_state.current_summary = ""
if 'embedding_generated' not in st.session_state:
    st.session_state.embedding_generated = False
if 'embedding_generated_text' not in st.session_state:
    st.session_state.embedding_generated_text = ""

# Custom CSS for collapsible sidebar and layout
st.markdown("""
<style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    } 
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    .stRadio > div {
        flex-direction: row;
    }
    .stRadio > div > label {
        margin-right: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Login")
    
    # Login Section
    with st.expander("Login", expanded=not st.session_state.logged_in):
        if not st.session_state.logged_in:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login"):
                    if username and password:
                        if verify_user(username, password):
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
                    else:
                        st.warning("Please enter username and password correctly")
            with col2:
                if st.button("Sign Up"):
                    if username and password:
                        if is_valid_username(username):
                            if is_valid_password(password):
                                if not username_exists(username):
                                    if create_user(username, password):
                                        st.success("User created successfully. You can now log in.")
                                    else:
                                        st.error("An error occurred. Please try again.")
                                else:
                                    st.error("Username already exists. Please choose a different one.")
                            else:
                                st.error("Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, and one digit.")
                        else:
                            st.error("Username must be 3-20 characters long and can only contain letters, numbers, and underscores.")
                    else:
                        st.warning("Please enter both username and password")

    # Question Answering Section
    st.title("Question Answering")
    with st.expander("Question Answering", expanded=False):
        if st.session_state.logged_in:
            if st.session_state.text_inserted:
                if not st.session_state.embedding_generated:
                    st.warning("Generating embedding. Please wait...")
                else:
                    question = st.text_input("Write your question here")
                    if st.button("Get Answer"):
                        if question:
                            answer = Generate_answer(question)
                            st.text_area("Answer", value=answer, height=800)
                            # st.markdown(f"**Answer:**\n\n{answer}")
                            
                            # Add to database
                            user_id = get_user_id(st.session_state.username)
                            add_qa(question, answer, st.session_state.current_text, st.session_state.username, user_id)
                        else:
                            st.warning("Please enter a question.")
            else:
                st.warning("Please insert text or upload a file before asking questions.")
        else:
            st.warning("Please log in to use the QA feature.")

# Main content
st.title("AI Summarization Service")

if st.session_state.logged_in:
    # Input method selection
    input_method = st.radio("Choose input method:", ("Text", "File Upload"), horizontal=True)

    if input_method == "Text":
        user_text = st.text_area("Enter your text here", height=100)
        if user_text:
            st.session_state.text_inserted = True
            st.session_state.current_text = user_text
            if st.session_state.current_text != st.session_state.embedding_generated_text:
                print("Thread starts")
                threading.Thread(target=Generate_embedding, args=(user_text,)).start()
                st.session_state.embedding_generated = True
                st.session_state.embedding_generated_text = user_text
            st.success("Text submitted successfully!")
        else:
            st.warning("Please enter some text.")
    else:  # File Upload
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])
        if uploaded_file is not None:
            user_text = read_file_content(uploaded_file)
            if user_text != "Unsupported file type":
                st.session_state.text_inserted = True
                st.session_state.current_text = user_text
                if st.session_state.current_text != st.session_state.embedding_generated_text:
                    print("Thread starts")
                    threading.Thread(target=Generate_embedding, args=(user_text,)).start()
                    st.session_state.embedding_generated = True
                    st.session_state.embedding_generated_text = user_text
                st.success("File uploaded successfully!")
            else:
                st.error("Unsupported file type. Please upload a txt, pdf, or docx file.")

    # Summarization section
    if st.session_state.text_inserted:
        st.subheader("Summarization")
        summarization_method = st.radio("Choose summarization method:", ("Abstractive", "Extractive"), horizontal=True)

        if summarization_method == "Abstractive":
            model = st.selectbox("Select model", ["OpenAI", "Gemini", "Llama", "Mixtral", "Gemma", "Pegasus"])
        else:
            model = st.selectbox("Select model", ["LuhnSumy", "BERT"])

        if st.button("Summarize"):
            if st.session_state.current_text:
                if summarization_method == "Abstractive":
                    summarizer = AbstractiveSummarization(model)
                else:
                    summarizer = ExtractiveSummarization(model)
                
                summary = summarizer.summarize(st.session_state.current_text)
                st.session_state.current_summary = summary
                
                # Add to database
                user_id = get_user_id(st.session_state.username)
                add_summary(st.session_state.current_text, summary, st.session_state.username, user_id)
                
                # st.success("Summary generated successfully!")
            else:
                st.warning("Please input text or upload a file before summarizing.")

        # Display the current summary
        if st.session_state.current_summary:
            st.text_area("Summary", value=st.session_state.current_summary, height=200)
            

else:
    st.warning("Please log in using the sidebar to access the summarization service.")