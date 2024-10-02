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

llm_api = st.secrets["llm_api_key"]
qdrant_api = st.secrets["qdrant_api_key"]

# Set the page configuration
st.set_page_config(page_title="Talk2Log", layout="wide")

docs_dir = './docs'  # Specify the directory containing your documents

@st.cache_data
def get_docs(docs_dir):
    """
    Get all Word and Excel files in the specified directory
    """
    extracted_data = extract_all_text(docs_dir)
    split_texts = split_texts_to_paragraphs(extracted_data)
    filtered_sentences = filter_short_sentences(split_texts)
    flattened_sentences = [sentence for sublist in filtered_sentences for sentence in sublist]

    return flattened_sentences

# -- Extract data from the Word and Excel documents
def extract_text_from_word(file_path):
    doc = Document(file_path)
    return "\n\n".join([para.text for para in doc.paragraphs])

def extract_text_from_excel(file_path):
    df = pd.read_excel(file_path)
    df['sentence'] = df.apply(lambda row: f"The potential cause for {row['Condition']} is {row['Potential Cause']}", axis=1)
    sentences = ". ".join(df['sentence'].tolist())
    return sentences

def get_all_docs_and_xls_files(docs_dir):
    word_files = []
    excel_files = []
    
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.doc') or file.endswith('.docx'):
                word_files.append(os.path.join(root, file))
            elif file.endswith('.xls') or file.endswith('.xlsx'):
                excel_files.append(os.path.join(root, file))
    
    return word_files, excel_files

def extract_all_text(docs_dir):
    word_files, excel_files = get_all_docs_and_xls_files(docs_dir)
    all_text = []

    # Process Word files
    for file in word_files:
        text = extract_text_from_word(file)
        all_text.append(text)

    # Process Excel files
    for file in excel_files:
        text = extract_text_from_excel(file)
        all_text.append(text)

    return all_text


## -- Split the text into paragraphs
def split_texts_to_paragraphs(extracted_data):
    """
    Splits each text in the provided list into paragraphs.
    """
    split_para = []
    for entry in extracted_data:
        paragraphs = entry.split("\n\n")
        split_para.append(paragraphs)

    return split_para

## -- Filter the sentences longer than 5 words and flatten the list
def filter_short_sentences(split_texts, min_words=15):
    """
    Filters out sentences that have less than `min_words` words while retaining their metadata.
    """
    filtered_data = []
    for entry in split_texts:
        filtered_sentences = [sentence for sentence in entry if len(sentence.split()) >= min_words]
        filtered_data.append(filtered_sentences)
    return filtered_data

def get_embeddings(text):
    model_vec = SentenceTransformer("all-mpnet-base-v2")
    return model_vec.encode(text)

## -- Store vectorized sentences in a vector database (Qdrant)
def upsert_vectors(client, collection_name, vectorized_sentences):
    # Prepare points for upsert
    points = [
        PointStruct(id=i + 1, vector=vector)
        for i, vector in enumerate(vectorized_sentences)
    ]

    # Perform the upsert operation
    operation_info = client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points,
    )

    return operation_info

# -- Get the documents and extract the sentences
flattened_sentences = get_docs(docs_dir)

# -- Check if the collection already exists
qdrant_client = QdrantClient(
    url="https://9817dd27-777f-45cb-9bfe-78a2a8e14b88.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=qdrant_api,
)
collection_name = "docs_collection"
collections = qdrant_client.get_collections()
collections = [collection.name for collection in collections.collections]

if collection_name not in collections:
        
    vectorized_sentences = get_embeddings(flattened_sentences)

    # Create the collection
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    # Upsert the vectors to the collection
    upsert_vectors(qdrant_client, collection_name, vectorized_sentences)

## -- Function to search for similar sentences (vector embeddings) in the collection
def retrieve_and_rerank(query, top_k=10):
    query_embedding = get_embeddings(query)
    results = qdrant_client.query_points(
    collection_name="docs_collection",
    query=query_embedding,
    with_payload=False,
    limit=top_k
).points
    
    return results

## -- Get the sentences similar to the query
def get_sentences_from_results(reranked_results, split_texts):
    """ Extracts the sentences from the reranked results. """
    sentence_indices = [result.id - 1 for result in reranked_results]
    sentences = [split_texts[i] for i in sentence_indices if 0 <= i < len(split_texts)]
    return "\n\n".join(sentences)


# Function to get log story from Groq API
@st.cache_data
def get_log_story_groq(model, selected_file, language, focus):
    """
    Generate a narrative based on the log data using the Groq API.
    """
    # Open the file with the log data
    with open(selected_file, "r") as f:
        log_data = f.read()

    client = Groq(
        api_key=llm_api,
    )

    chat_completion = client.chat.completions.create(
        messages=[
{
    "role": "system",
    "content": "You are a multilingual smart engineer responsible for maintaining a complex manufacturing system of corrugated box production line. Your task is to review system log messages and craft an engaging narrative explaining the system's operation to the team."
},
{
    "role": "user",
    "content": f"""
- **Task**: Write a clear and coherent story summarizing the system's status and events based on the provided logs.
  
- **Narrative Instructions**:
  - **Title**: "System Operation Summary" (use header 4 format, `####`).
  - Capture just the important details from the logs, focusing on system failures and maintenance or anything that can affect production's KPIs.
  - Do **not** refer to the log itself in the narrative.
  - Use engaging, technical language and provide specific details, logically connecting events.
  - Do **not** make assumptions or invent details when information is lacking.
  - Use 24-hour time format (e.g., HH:MM). For failure times, include seconds (e.g., 10:30:15), but omit milliseconds.
  - Start with extracted start and end times from the log: "**Start**: YYYY-MM-DD HH:MM | **End**: YYYY-MM-DD HH:MM".
  - Mention specific times when relevant (e.g., "At 10:30"). For previously mentioned times, use "At the same time" or time differences like "3 hours later."
  - Format all variables from the logs in **bold**, except the hour/time.
  - Use Markdown formatting, **never HTML**.

- **Failure Summary**:
  - Only include **real system failures** (ignore alarms, warnings, or other non-failure alerts).
  - Include a table under a header 4 `####` "Failure Summary
  | Time       | Machine ID | Failure Condition |
  |------------|------------|-------------------|
  | 09:31:21 | F4 | F4 exceeded max vibration limit |
  | 09:31:21 | M3 | Pressure sensor reading outside of range |
    - Do not include anything in the table that do not accomplish to this format.

  - **Group failure conditions** related to the same failure into a **single failure event**. Each row in the table below should summarize all related events that led to the failure.
  - Accurately describe machine conditions during failures without assuming extra details.

- Write the completion in {language}, but do **not** translate any tags or extracted variables from the log files.
- **LOG DATA**: {log_data}
"""
}
        ],
        model=model,
    )

    return chat_completion.choices[0].message.content

# Function to get log story from Groq API
@st.cache_data
def get_summary_machine(model, selected_file, machine_id, time):
    """
    Generate a narrative based on the log data using the Groq API.
    """
    # Open the file with the log data
    with open(selected_file, "r") as f:
        log_data = f.read()

    client = Groq(
        api_key=llm_api,
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are a multilingual smart engineer who is responsible for the maintenance of a complex manufacturing system.
You will receive log messages from this system and create a engaging narrative to explain to the team what happened.""",
            },
            {
"role": "user",
"content": f"""
- What failures occurred in machine ID {machine_id} at {time}?
- What events led to the failure?
- What was the machine's state before the failure?
- What was the failure condition?
- What was the time of the failure?
- Write a clear and detailed paragraph to describe the machine's state, the failure, and any relevant events.
- Do not refer to the log itself in the story.
- If there is lack of information, do not assume or invent details.
- All times should be in the 24-hour format, using the format HH:MM. If it refers to a failure, include the seconds (e.g., 10:30:15), but NEVER the miliseconds.
LOG DATA: {log_data}
""",
            }
        ],
        model=model,
    )

    return chat_completion.choices[0].message.content

def get_rca_groq(model, summary_machine, support_info, language):

    client = Groq(
        api_key=llm_api,
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are a multilingual smart engineer responsible for conducting an analysis of the potential causes affecting the machine. You will receive documents containing descriptions of events of failure that happened in the past.""",
            },
            {
                "role": "user",
                "content": f"""Your task is to list the most probable causes grounded in the past events. The analysis should be based on the following instructions:
1. Compare all the past events and extract the most relevant information that could lead to the failure, considering the machine's state and events that anteceded the failure.
2. Do not use all the information from the support documents, only the most relevant parts to support your analysis.
3. Bold all variables extracted from the log files, except the times and dates.
4. Do not explicitly refer to the support documents in the analysis.
5. Report Description: [{summary_machine}]
6. Past failure's root cause analysis: [{support_info}]
7. The analysis should be written in {language}.

Output Format:
Facts (header 4 ####)
- List all relevant details about the failure extracted from the report description using bullet points information. Try to group the information in a logical way to facilitate the analysis.
Always include the machine ID, the failure condition, and the time of the failure.

Potential Causes (header 4 ####)
List the potential causes derived from the support information. Use the following format for each cause:
- **Condition 1:** [Describe the machine condition].
    - [List the specific causes associated with the condition].

- **Condition 2:** [Describe the machine condition].
    - [List the specific causes associated with the condition].

- Indent the potential causes list under the condition.
- Every condition should have at least one potential cause.
""",
            }
        ],
        model=model,
    )

    return chat_completion.choices[0].message.content

## -- Extract machine IDs from the log narrative
@st.cache_data
def extract_failures(text):
    """
    Extracts Machine IDs from the provided text that match the pattern | [uppercase letter][number] |.
    """
    # Regex to extract Machine IDs that follow the pattern | [uppercase letter][number] |
    pattern = r'\|\s*([\d:]+(?:\.\d{1,3})?)\s*\|\s*([A-Z]+\d*)\s*\|'
    matches = re.findall(pattern, text)
    times = [match[0] for match in matches]
    machine_ids = [match[1] for match in matches]

    return times, machine_ids

# Streamlit app

## - Create session state manager
if 'stage' not in ss:
    ss.stage = 0

def set_stage(stage):
    ss.stage = stage

# Header
st.title("💬 Talk2Log")
st.write("🌟 Welcome! I'm a tool to help you analyze log data and find potential causes for system failures.")

# Sidebar for file and language selection
st.sidebar.header("Select the log file")
log_dir = './logs'  # Specify the log directory
log_files = [f for f in os.listdir(log_dir) if f.endswith('.txt')]
log_files = sorted(log_files)

if log_files:

    model_names = [
        "llama-3.1-70b-versatile",
        # "gemma-7b-it",
        # "gemma2-9b-it",
        # "llama-3.1-8b-instant",
        # "llama-3.2-11b-text-preview",
        # "llama-3.2-11b-vision-preview",
        # "llama-3.2-1b-preview",
        # "llama-3.2-3b-preview",
        # "llama-3.2-90b-text-preview",
        # "llama-guard-3-8b",
        # "llama3-70b-8192",
        # "llama3-8b-8192",
        # "llama3-groq-70b-8192-tool-use-preview",
        # "llama3-groq-8b-8192-tool-use-preview",
        # "llava-v1.5-7b-4096-preview",
        # "mixtral-8x7b-32768"
    ]
    
    # Select the model to use
    model = st.sidebar.selectbox("🧠 Select the model to use:", model_names)

    # Create a dropdown to select a log file in the sidebar
    selected_file = st.sidebar.selectbox("📄 Select a log file:", log_files)
    file_path = os.path.join(log_dir, selected_file)

    # Select the focus of analysis
    focus = st.sidebar.selectbox("🎯 Select the focus of analysis:", ["System Failure", "System Performance"])

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
        # Display the selected log file
        with st.expander("Log file content", expanded=False, icon = "📄"):
            with open(file_path, "r") as f:
                log_data = f.read()  # Read the file content
                st.text(log_data) 
        
        # Display the log analysis section
        st.subheader("🧠 Log analysis")
        narrative = get_log_story_groq(model, file_path, language, focus)
        st.write(narrative)
        times, machine_ids = extract_failures(narrative)
        failures = [f"{machine_id} at {time}" for machine_id, time in zip(machine_ids, times)]
        if failures:
            st.subheader("🕵️‍♂️ Looking for potential causes?")
            selected_machine_id = st.selectbox("Select a machine ID to view further details:", failures)
            pattern = r'(\w+)\s*at\s*(\d{2}:\d{2}:\d{2})'
            match = re.search(pattern, selected_machine_id)
            machine_id = match.group(1)
            time = match.group(2)

            st.button("🔍 Find potential causes", on_click=set_stage, args = (2,))
            if ss.stage > 1:
                reranked_results = retrieve_and_rerank(narrative, top_k=5)
                sentences = get_sentences_from_results(reranked_results, flattened_sentences)
                summary_machine = get_summary_machine(model, file_path, machine_id, time)
                potential_causes = get_rca_groq(model, summary_machine, sentences, language)
                st.subheader(f"🧩 Potential Causes for Machine {machine_id} failure")
                with st.expander("Machine report summary", expanded=False, icon = "📄"):
                    st.write(summary_machine)                
                with st.expander("Support Information", expanded=False, icon = "📄"):
                    st.write(sentences)
                st.write(potential_causes)
        else:
            st.write("❌ No machine failures detected in the log data.")
else:
    st.write("❌ No log files found in the current directory.")
