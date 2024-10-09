import os
import re
import streamlit as st
from streamlit import session_state as ss
from groq import Groq
import pandas as pd
from docx import Document
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from openai import OpenAI

llm_api = st.secrets["llm_api_key"]
qdrant_api = st.secrets["qdrant_api_key"]
openai_key=st.secrets["openai_api_key"]

# Set the page configuration
st.set_page_config(page_title="Talk2log", layout="wide")

##-- Settings for embedding
emb_model = SentenceTransformer('all-mpnet-base-v2')
tag_descriptions = pd.read_csv('docs/tag_descriptions.csv')

@st.cache_data
def get_embeddings(text):
    """
    Get embeddings for the provided text using the SentenceTransformer model.
    """
    model = emb_model
    doc_embeddings = model.encode(text)
   
    return doc_embeddings

# Create embeddings for each description
tag_descriptions['embedding'] = tag_descriptions['desc'].apply(get_embeddings)

##-- Connect to the vector database
qdrant_client = QdrantClient(
    url="https://9817dd27-777f-45cb-9bfe-78a2a8e14b88.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=qdrant_api,
)
vector_size=tag_descriptions['embedding'].iloc[0].shape[0]
distance="Cosine"

##-- Upsert the vectors to the collection
collection_name = "tags_description"
collections = qdrant_client.get_collections()
collections = [collection.name for collection in collections.collections]

if collection_name not in collections:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=distance)
    )

    points = [
        PointStruct(id=i, vector=embedding, payload={"tag": tag, "description": desc})
        for i, (tag, desc, embedding) in enumerate(zip(tag_descriptions['tag'], tag_descriptions['desc'], tag_descriptions['embedding']))
    ]

    # Insert into Qdrant
    qdrant_client.upsert(collection_name=collection_name, points=points)

def gen_summary_message(selected_file, support_info):
    # Open the file with the log data
    with open(selected_file, "r") as f:
        log_data = f.read()   

    messages=[
    {
        "role": "system",
        "content": """
    You are a multilingual expert engineer managing the operation and maintenance of a highly complex float glass production line. Your primary task is to analyze text log messages from this system and translate them into concise, actionable narratives. Each log message contains vital details, including machine statuses, alarms, operator actions, and adjustments in the system's performance.
    You are provided with a dictionary of tags and their corresponding descriptions, which you will use to offer insightful interpretations of technical terms. Your job is to interpret these logs and explain what happened and why it happened, making it easy for technical people to understand.""",
    },
    {
    "role": "user",
    "content": f"""
    Instructions for Log Analysis:
    - Focus on significant events, operational changes, and actions taken by the operators to ensure the smooth running of the system.
    - Identify any alarms or critical issues that may affect the system, and describe their impact (e.g., production delays, equipment failures).
    - When explaining the events, refer to the descriptions from the tag dictionary rather than using raw tag names. This will provide more context for your narrative.
    - Be mindful of the time range of events and reference time whenever possible.
    - Don't make any assumptions; base your analysis solely on the information provided in the log data. Just describe what you see in the logs.
    - Don't mention the log data directly in your narrative; use it to extract relevant information for your analysis.
    - Use Markdown for clear structuring of the response, with sections for the analysis. Bold all variables extracted from the log, such as machine names, operator names, actions, etc., except times and dates.

    Structure of the Narrative:
    - A summary of the overall system's operation during the specified time range.
    - A description of critical issues, why they occurred, and their impact.
    - A breakdown of operator interventions and their significance.
    - Length: The narrative should contain no more than 500 words.

    Dictionary of tags: {support_info}  
    Log Data: {log_data}
    """
    }
            ]
    
    return messages

@st.cache_data
def get_openai_response(model, messages, temperature=0.2, top_p=0.1):

    # Example API call (chat completion with GPT-4 model)
    client = OpenAI(
        # This is the default and can be omitted
        api_key=openai_key,
    )

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )

    # Print the response
    return chat_completion.choices[0].message.content, chat_completion.usage.completion_tokens, chat_completion.usage.prompt_tokens

def search_log_entry(log_entry, model, client, threshold):
    # Generate embedding for the log entry
    log_embedding = model.encode(log_entry)

    # Search in Qdrant for all vectors that have a score higher than the threshold
    result = client.search(
        collection_name="tags_description",
        query_vector=log_embedding,
        limit=5,
        score_threshold=threshold,
    )

    # Filter results further if needed and return only those above the threshold
    matching_results = [
        {"tag": res.payload['tag'], "description": res.payload['description']}
        for res in result if res.score > threshold
    ]

    return matching_results

def get_support_info(log_file, model, client, threshold):
    
    with open(log_file, "r") as f:
        log_data = f.read()

    # Assuming log_data is a multi-line string
    log_entries = log_data.splitlines()

    # Dictionary to store unique tags and their corresponding description
    unique_results = {}

    # Iterate through each line (log entry)
    for log_entry in log_entries:
        # Run search_log_entry for each line of log_data
        matching_results = search_log_entry(log_entry, model, client, threshold)
        
        # Loop through the matching results and store them if they have unique tags
        for result in matching_results:
            tag = result['tag']
            description = result['description']
            
            # Add to the dictionary only if the tag is not already present
            if tag not in unique_results:
                unique_results[tag] = description

    # Now, unique_results contains only unique tags and their descriptions
    # You can convert this to a list or whatever format you need
    tag_descriptions = [{"tag": tag, "description": description} for tag, description in unique_results.items()]
    
    return tag_descriptions

##-- Variables
llm_model = "gpt-4o-mini"
temperature = 0.2
top_p = 0.1

# Streamlit app

## - Create session state manager
if 'stage' not in ss:
    ss.stage = 0

def set_stage(stage):
    ss.stage = stage

# Header
st.title("💬 Talk2log")
st.write("🌟 Welcome! I'm a tool to translate complex industrial log files into engaging narratives.")

# Sidebar for file and language selection
st.sidebar.header("Select the log file")
log_dir = './logs'  # Specify the log directory
log_files = [f for f in os.listdir(log_dir) if f.endswith('.txt')]
log_files = sorted(log_files)

if log_files:

    # Create a dropdown to select a log file in the sidebar
    selected_file = st.sidebar.selectbox("📄 Select a log file:", log_files)
    file_path = os.path.join(log_dir, selected_file)

    # Updated language options in the sidebar
    languages_in_native = [
        "English",
        "Português (Brasil)",
        "Français",
        "Deutsch",
        "Español",
        "Italiano"
    ]
    language = st.sidebar.selectbox("🌐 Select Language:", languages_in_native)

    st.sidebar.button("📝 Analyze the log", on_click=set_stage, args = (1,))
    if ss.stage > 0:
        support_info = get_support_info(file_path, emb_model, qdrant_client, 0.5)
        # Display the selected log file
        with st.expander("Log file content", expanded=False, icon = "📄"):
            with open(file_path, "r") as f:
                log_data = f.read()  # Read the file content
                st.text(log_data)

        with st.expander("Support file content", expanded=False, icon = "📄"):
                st.text(support_info)
        
        # Display the log analysis section
        st.subheader("🧠 Log analysis")
        messages = gen_summary_message(file_path, support_info)
        response, num_tokens_response, num_tokens_prompt = get_openai_response(llm_model, messages, temperature, top_p)

        st.write(response)
else:
    st.write("❌ No log files found in the current directory.")
