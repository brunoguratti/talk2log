import os
import json
import streamlit as st
from groq import Groq

# Function to get log story from Groq API
def get_log_story_groq(model, selected_file, language, focus):
    # Construct the file name
    # Open the json file with the log data
    with open(selected_file, "r") as f:
        log_data = json.load(f)

    client = Groq(
        api_key=st.secrets["api_key"],
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a multilingual smart engineer who is responsible for the maintenance of a complex manufacturing system.
CONTEXT: This system manages the production of corrugated boxes by coordinating tasks, machines, feeder machines, buffer areas, and logistics. Machines cut and shape cardboard into boxes, while feeders supply the necessary paper for corrugation. Tasks include processing and transferring materials, with states like "IN_PROGRESS" or "COMPLETED" tracking progress. Machines can be "PROCESSING" or in "FAILURE," and feeder machines handle paper input, with states like "RDY" or "MAINTENANCE." Buffer areas store materials between stages, and logistics ensure smooth movement between machines. The platform as a whole operates in states like "RUNNING" or "STOPPED," reflecting overall production activity.
You will receive json log messages from this system and create a engaging narrative to explain to the team what happened.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
- Don't make a list of events, but a coherent story that captures the evolving state of the system based on the provided logs.
- Use a engaging and technical language and be as specific as possible, conecting the events in a logical way.
- Do not refer to the log itself in the story.
- All times should be in the 24-hour format, using the format HH:MM. If it refers to a failure, include the seconds (e.g., 10:30:15), but NEVER the miliseconds.
- Start the narrative extracting start and end date from the log files: "**Start**: 2024-05-15 10:40 AM | **End**: 2024-05-15 17:40 AM".	
- When referencing a specific moment, include the time (e.g., "At 10:30 AM"). If the time was previously mentioned, use "At the same time" or indicate the time difference, such as "3 hours later."
- Use Markdown to format the text, never HTML.
- Bold all variables extracted from the log files, except the times and dates.
- Your completion should be written in {language}. Do not translate the data extracted from the logs.
- The focus of analysis should be on: {focus}.
- If the focus is on failures, provide a detailed failure analysis:
    - Include a section with a header h3 (use markdown ###) "Failure Summary"
    - Include a table with the following columns: "Time", "Machine ID", "Pre-failure state" and extract exactly this information from the logs.
    - Include a written summary of the failure pattern, analysing the pre-failure states and times.
- If Performance focus is selected, do not include any summary.

LOG DATA: {log_data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            }
        ],
        model=model,
    )

    return chat_completion.choices[0].message.content


# Streamlit app
st.set_page_config(page_title="Talk2Log", layout="wide")

# Header
st.title("💬 Talk2Log")
st.write("Welcome! Select a JSON log file and generate a narrative based on its content. 🌟")

# Sidebar for file and language selection
st.sidebar.header("Select the log file")
log_dir = './logs'  # Specify the log directory
json_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]

if json_files:

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
    
    # Select the model to use (standard is "llama-3.1-70b-versatile)
    model = st.sidebar.selectbox("🧠 Select the model to use:", model_names)

    # Create a dropdown to select a log file in the sidebar
    selected_file = st.sidebar.selectbox("🔍 Select a JSON log file:", json_files)
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

    if st.sidebar.button("📝 Analyze the log"):
        st.subheader("🧠 Log analysis...")
        # Call the appropriate function based on the selected service
        narrative = get_log_story_groq(model, file_path, language, focus)

        # Display the generated narrative
        st.write(narrative)
else:
    st.write("❌ No JSON log files found in the current directory.")