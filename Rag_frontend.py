import streamlit as st
import uuid
from Rag_backend import build_faiss_from_pdf
st.title("Rag based Application ")

from Rag_backend import app
def generate_thread():
   thread=uuid.uuid4()
   return thread
if 'message_history' not in st.session_state:
   st.session_state['message_history']=[]
if 'thread_id' not in st.session_state:
   st.session_state['thread_id']=generate_thread()
if 'thread_list' not in st.session_state:
   st.session_state['thread_list']=[st.session_state['thread_id']]

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Save file locally
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    # Call your function on the saved file
    vectorstore = build_faiss_from_pdf("uploaded_file.pdf")

    # Show output
    st.write("File Processes")



for message in st.session_state['message_history']:
   with st.chat_message(message['role']):
      st.text(message['content'])
input_data=st.chat_input('Type here')
if input_data:
   config={'thread_id':st.session_state['thread_id']}
# First query
   with st.chat_message('user'):
     st.text(input_data)
   st.session_state['message_history'].append({'role':'user','content':input_data})
   result = app.invoke({"query": input_data, "history": []},config=config)
   
   with st.chat_message("AI"):
    result = st.write_stream(
        message_chunk.content   # stream only answer
        for message_chunk, metadata in app.stream(
            {"query":input_data,"history":[] },
            config={"configurable": {"thread_id": st.session_state['thread_id']}},
            stream_mode="messages",
        )
    )
   st.session_state['message_history'].append({'role':'AI','content':result})
