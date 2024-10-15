import os
import re
import streamlit as st
from streamlit import session_state as ss
import pandas as pd
from docx import Document
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from openai import OpenAI
import json
import cohere
from PIL import Image
import io
import base64

cohere_key=st.secrets["cohere_api_key"]
qdrant_api = st.secrets["qdrant_api_key"]
openai_key=st.secrets["openai_api_key"]

# Set the page configuration
st.set_page_config(page_title="talk2log :: turning data into dialogue", layout="wide", page_icon="assets/images/favicon.ico")

##-- Settings for embedding
emb_model = SentenceTransformer('all-mpnet-base-v2')
tag_descriptions = pd.read_csv('docs/tag_descriptions.csv')

@st.cache_data(show_spinner=False)
def get_embeddings(text):
    """
    Get embeddings for the provided text using the SentenceTransformer model.
    """
    model = emb_model
    doc_embeddings = model.encode(text)
   
    return doc_embeddings

tag_descriptions['cons_desc'] = tag_descriptions['tag'] + ' ' + tag_descriptions['desc']

# Create embeddings for each description
tag_descriptions['embedding'] = tag_descriptions['cons_desc'].apply(get_embeddings)

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

@st.cache_data(show_spinner=False)
def gen_summary_messages (selected_file, support_info):
    # Open the file with the log data
    with open(selected_file, "r") as f:
        log_data = f.read()   

    messages=[
    {
        "role": "system",
        "content": """
    You are an expert engineer managing the operation and maintenance of a highly complex float glass production line.
    Your primary task is to analyze text log messages from industrial control system (PLC, SCADA, etc.) and translate them into concise narratives using story-telling techniques.
    You are provided with a dictionary of machine and device tags and their corresponding descriptions.
    
    Instructions for Log Analysis:
    - Don't mention the log data directly in your narrative.
    - Translate all tags and variables from the log data into plain English using the dictionary of tags and use them in the narrative.
    - Bold all variables extracted from the log, such as machine names, operator names, actions, etc., except times and dates.
    - Mention the time or period of the event in the narrative, such as "During the night shift on [date]", "At 10:30 AM", "45 minutes later", etc.
    - DO NOT make any assumptions, comments or conclusions about the events, such as: "indicating "This may have caused...", "This could have led to...", "This might have been due to...", etc.

    Structure of the Narrative:
    - Title: "# Line Operations: [date in format MMMM DD, YYYY]"
    - Overall summary - Start with a summary of the overall system's operation during the specified time range. Think of this as the introduction to the story.
    - Critical events - Then, describe critical events, machine failures, malfunctions, stoppages (DO NOT include operators interventions or alarm events in this section). This should be addressed to the maintenance team. Group by machine or device for better clarity.
    - Operational intervention - The same way, describe the operator interventions. This should be addressed the operations team.
    - Length: The narrative should contain no more than 500 words.
    - Use Markdown for clear structuring of the response, with sections for the analysis.

    Here's an example of how you can structure your narrative:

# Line Operations: October 15, 2024

## Overall Summary
During the early morning shift on October 15, 2024, the float glass production line performed efficiently, with the majority of machines operating within standard parameters. However, a few critical events occurred that impacted production. These issues required maintenance attention and operational intervention to ensure smooth continuation of the process.

## Critical Events
### Furnace Hopper
- At 2:45 AM, the batch plant system reported an error in the raw material dosing system, indicating a failure in the feed of glass raw material. This caused a momentary halt in the material flow into the furnace.

### BC4 Conveyor
- At 3:20 AM, the BC4 conveyor system experienced an unexpected stoppage. This disruption led to a drop in the furnace glass level, triggering a level alarm at 3:25 AM. The level dropped below the operational target, impacting the overall flow of material into the furnace.

### Stirrer
- At 4:00 AM, the furnace stirrer stopped unexpectedly. The system automatically restarted five minutes later, which resumed stirrer's rotation at 5 RPM

### Ribbon Break
- At 4:40 AM, a ribbon break occurred at the bath exit, leading to a rapid decrease in the bath exit temperature. The temperature dropped below the acceptable range, which would need adjustment to restore ideal conditions for ribbon formation.

## Operational Intervention
### Glass Level Adjustment by Operator: STUSBP01
- At 3:30 AM, the operator at STUSBP01 acknowledged the low glass level alarm triggered by the BC4 stoppage. The operator increased the material feed rate to raise the glass level in the furnace, preventing further interruptions in the production flow.

### Manual Temperature Reduction by Furnace Operator
- At 5:05 AM, due to the ribbon break, the operator at STUSBP01 manually increased the bath exit temperature by 5°C to stabilize and facilitate recovery of the ribbon formation process.

### Furnace Stirrer Speed Increase
- At 4:45 AM, following the automatic restart of the furnace stirrer, the operator manually increased the stirrer speed to 6 RPM to accelerate temperature recovery in the bath. 
        """,
    },
    {
    "role": "user",
    "content": f"""
    Log Data: {log_data}
    Dictionary of tags: {support_info}  
    """
    }
            ]
    
    return messages

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def get_cohere_response(messages, llm_model, temperature=0.3, top_p=0.3):
    """ Get a response from the Cohere model."""

    co = cohere.ClientV2(cohere_key)

    response = co.chat(
        model=llm_model,
        messages=messages,
        temperature=temperature,
        p=top_p,)

    return response.message.content[0].text


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

def get_support_info(log_file, emb_model, client, threshold):
    
    with open(log_file, "r") as f:
        log_data = f.read()

    # Assuming log_data is a multi-line string
    log_entries = log_data.splitlines()

    # Dictionary to store unique tags and their corresponding description
    unique_results = {}

    # Iterate through each line (log entry)
    for log_entry in log_entries:
        # Run search_log_entry for each line of log_data
        matching_results = search_log_entry(log_entry, emb_model, client, threshold)
        
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
    tag_descriptions = json.dumps(tag_descriptions)
    
    return tag_descriptions

def load_css(file_name):
    """ Load external CSS file."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

##-- Variables
llm_model = "command-r-plus-08-2024"
temperature = 0.3
top_p = 0.3

# Streamlit app
## - Create session state manager
if 'stage' not in ss:
    ss.stage = 0

def set_stage(stage):
    ss.stage = stage

# Header
load_css('css/styles.css')
st.image("assets/images/talk2log_logo.png", width=200)
st.write("**👋 Hi! I'm an AI tool that will help you transform complex log files into insightful and easy-to-understand narratives.**")
st.write("This is a demo version of the tool, and it is designed to assist you in analyzing log files from industrial control systems.")
st.write("To get started, select a sample log file from the sidebar and click the **Analyze the log** button.")

# Sidebar for file and language selection
st.sidebar.header("Get started")
log_dir = './logs'  # Specify the log directory
log_files = [f for f in os.listdir(log_dir) if f.endswith('.txt')]
log_files = sorted(log_files)

if log_files:

    # Create a dropdown to select a log file in the sidebar
    selected_file = st.sidebar.selectbox("📄 Select a log file:", log_files)
    file_path = os.path.join(log_dir, selected_file)

    st.sidebar.button("Analyze the log", on_click=set_stage, args = (1,))
    if ss.stage > 0:
        support_info = get_support_info(file_path, emb_model, qdrant_client, 0.4)
        # Display the selected log file
        with st.expander("Log file content", expanded=False, icon = "📄"):
            with open(file_path, "r") as f:
                log_data = f.read()  # Read the file content
                st.text(log_data)

        with st.expander("Support file content", expanded=False, icon = "📄"):
                st.json(support_info)
        
        # Display the log analysis section
        st.subheader("🧠 Log analysis")
        messages = gen_summary_messages(file_path, support_info)
        response = get_cohere_response(messages, llm_model, temperature, top_p)

        st.write(response)
else:
    st.write("❌ No log files found in the current directory.")

bg_logo = Image.open("assets/images/bganal_bw.png")

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

st.markdown(
    f'<br><div style="text-align: center;"><img src="data:image/png;base64,{image_to_base64(bg_logo)}" width="130"></div>',
    unsafe_allow_html=True,
)
