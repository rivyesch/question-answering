# import streamlit as st  
# from functions import *
# import base64

# # Initialize the API key in session state if it doesn't exist
# if 'api_key' not in st.session_state:
#     st.session_state.api_key = ''

# def display_pdf(uploaded_file):

#     """
#     Display a PDF file that has been uploaded to Streamlit.

#     The PDF will be displayed in an iframe, with the width and height set to 700x1000 pixels.

#     Parameters
#     ----------
#     uploaded_file : UploadedFile
#         The uploaded PDF file to display.

#     Returns
#     -------
#     None
#     """
#     # Read file as bytes:
#     bytes_data = uploaded_file.getvalue()
    
#     # Convert to Base64
#     base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    
#     # Embed PDF in HTML
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    
#     # Display file
#     st.markdown(pdf_display, unsafe_allow_html=True)


# def load_streamlit_page():

#     """
#     Load the streamlit page with two columns. The left column contains a text input box for the user to input their OpenAI API key, and a file uploader for the user to upload a PDF document. The right column contains a header and text that greet the user and explain the purpose of the tool.

#     Returns:
#         col1: The left column Streamlit object.
#         col2: The right column Streamlit object.
#         uploaded_file: The uploaded PDF file.
#     """
#     st.set_page_config(layout="wide", page_title="LLM Tool")

#     # Design page layout with 2 columns: File uploader on the left, and other interactions on the right.
#     col1, col2 = st.columns([0.5, 0.5], gap="large")

#     with col1:
#         # st.header("Input your API key")
#         # st.text_input('OpenAI API key', type='password', key='api_key',
#         #             label_visibility="collapsed", disabled=False)
#         st.header("Upload file")
#         uploaded_file = st.file_uploader("Please upload your PDF document:", type= "pdf")

#     return col1, col2, uploaded_file


# # Make a streamlit page
# col1, col2, uploaded_file = load_streamlit_page()

# # Process the input
# if uploaded_file is not None:
#     with col2:
#         display_pdf(uploaded_file)
        
#     # Load in the documents
#     documents = get_pdf_text(uploaded_file)
#     st.session_state.vector_store = create_vectorstore_from_texts(documents, file_name=uploaded_file.name)
#     st.write("Input Processed")

# # Generate answer
# with col1:
#     if st.button("Generate table"):
#         with st.spinner("Generating answer"):
#             # Load vectorstore:

#             answer = query_document(vectorstore = st.session_state.vector_store, 
#                                     query = {"Give me the title, summary, publication date, and authors of the research paper."},
#                                     acess_key_id = st.secrets["AWS_ACCESS_KEY_ID"],
#                                     secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
#                                     )
                            
#             placeholder = st.empty()
#             placeholder = st.write(answer)


















import streamlit as st
from functions import *
import base64

# Initialize the API key in session state if it doesn't exist
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

def display_pdf(uploaded_file):
    """
    Display a PDF file that has been uploaded to Streamlit.
    The PDF will be displayed in an iframe, with the width and height set to 700x1000 pixels.
    """
    # Read file as bytes:
    bytes_data = uploaded_file.getvalue()
    
    # Convert to Base64
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    
    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    
    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)


def load_streamlit_page():
    """
    Load the streamlit page with two columns. 
    Left: API key input & PDF uploader, Right: Greeting and Purpose.
    """
    st.set_page_config(layout="wide", page_title="LLM Tool")

    # Design page layout with 2 columns: File uploader on the left, and other interactions on the right.
    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        st.header("Upload file")
        uploaded_file = st.file_uploader("Please upload your PDF document:", type="pdf")

    return col1, col2, uploaded_file


# Main streamlit page
col1, col2, uploaded_file = load_streamlit_page()

# Process the input
if uploaded_file is not None:
    with col2:
        display_pdf(uploaded_file)
        
    # Load in the documents from the PDF
    documents = get_pdf_text(uploaded_file)  # Assuming this function extracts the text from PDF
    st.session_state.vector_store = create_vectorstore_from_texts(documents)
    st.write("PDF Processed Successfully")

# Query and generate an answer
with col1:
    query_text = st.text_input("Ask a question about the document:", "")
    
    if st.button("Generate Answer"):
        with st.spinner("Generating answer..."):
            if query_text and 'vector_store' in st.session_state:
                answer = query_document(
                    vectorstore=st.session_state.vector_store,
                    query=query_text,
                    access_key_id = st.secrets["AWS_ACCESS_KEY_ID"],
                    secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
                )
                cleaned_response = clean_text(answer)
                st.write(cleaned_response)
            else:
                st.warning("Please upload a document and ask a question.")
