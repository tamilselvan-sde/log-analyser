import os
import json
import time
import streamlit as st
from PIL import Image
from groq import Groq
from config.paths import MainPath

# Paths to folders
folder_path = f"{MainPath.folder_path}/log_analyser/model_outputs"
error_level_path = f"{MainPath.folder_path}/log_analyser/visualization/error_level"
keyword_clustering_path = f"{MainPath.folder_path}/log_analyser/visualization/keyword_clustering"
sentimental_analysis_path = f"{MainPath.folder_path}/log_analyser/visualization/sentimental_analysis"

# Function to troubleshoot the error using Groq API
def troubleshoot_error(api_key, error_message):
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"How to solve this error? {error_message}"
                }
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        result = ""
        for chunk in completion:
            result += chunk.choices[0].delta.content or ""
        return result
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit Sidebar Navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", ["Troubleshooting", "Plots and DataFrames"])

if selected_page == "Troubleshooting":
    st.title("Troubleshooting Guide (Groq)")
    st.markdown("---")  # Add a horizontal line for separation

    # Path to the sentiment analysis JSON file
    sentiment_output_path = os.path.join(folder_path, "sentiment_analysis_results.json")

    # Load the JSON file as a list of dictionaries
    try:
        with open(sentiment_output_path, "r") as file:
            log_data = [json.loads(line) for line in file if line.strip()]
    except Exception as e:
        st.error(f"Error loading sentiment analysis file: {e}")
        st.stop()

    # Filter messages where Confidence >= 0.71
    filtered_messages = [
        log["message"]
        for log in log_data
        if log.get("Confidence", 0) >= 0.71
    ]

    if not filtered_messages:
        st.warning("No messages with Confidence >= 0.71 found.")
    else:
        st.markdown("### 🔍 High Confidence Messages Identified")
        api_key = st.text_input("Enter your Groq API Key:", type="password", placeholder="Paste your Groq API key here...")
        if api_key.strip():
            for idx, message in enumerate(filtered_messages, start=1):
                st.subheader(f"Message {idx}:")
                st.write(f"**Message:** {message}")

                try:
                    # Add a 3-second delay between each request
                    time.sleep(3)

                    # Use the troubleshoot_error function to get a solution
                    with st.spinner("Getting solution from Groq..."):
                        solution = troubleshoot_error(api_key, message)

                    # Display the model's response with colored background
                    st.markdown(
                        f"""
                        <div style="background-color: #f0f0f5; padding: 10px; border-radius: 5px;">
                            <strong>Solution:</strong> {solution}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Error processing message: {e}")

                # Add spacing between each message
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.error("Please provide your API key to troubleshoot errors.")

elif selected_page == "Plots and DataFrames":
    st.title("Log Analyzer Outputs Viewer")
    st.markdown("---")  # Add a horizontal line for separation

    # List all files in the main data directory
    files = os.listdir(folder_path)
    json_files = [file for file in files if file.endswith(".json")]
    csv_files = [file for file in files if file.endswith(".csv")]

    # List all PNGs in visualization folders
    error_images = [os.path.join(error_level_path, img) for img in os.listdir(error_level_path) if img.endswith(".png")]
    keyword_images = [os.path.join(keyword_clustering_path, img) for img in os.listdir(keyword_clustering_path) if img.endswith(".png")]
    sentiment_images = [os.path.join(sentimental_analysis_path, img) for img in os.listdir(sentimental_analysis_path) if img.endswith(".png")]

    # Function to display file content
    def display_file(file_path):
        if file_path.endswith(".json"):
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                    try:
                        data = json.loads(content)  # Attempt to load full JSON
                        st.json(data)
                    except json.JSONDecodeError:
                        st.warning("File contains multiple JSON objects. Displaying each object.")
                        for line in content.splitlines():
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    st.json(data)
                                except json.JSONDecodeError:
                                    st.error(f"Invalid JSON line: {line}")
            except Exception as e:
                st.error(f"Error reading JSON file: {e}")
        elif file_path.endswith(".csv"):
            try:
                df = pd.read_csv(file_path)
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error loading CSV file: {e}")

    # Display JSON and CSV files in tabs
    if json_files or csv_files:
        st.header("JSON and CSV Outputs")
        file_tabs = st.tabs([os.path.splitext(file)[0] for file in (csv_files + json_files)])
        for idx, file in enumerate(csv_files + json_files):
            with file_tabs[idx]:
                file_path = os.path.join(folder_path, file)
                st.subheader(f"File: {file}")
                display_file(file_path)
    else:
        st.write("No JSON or CSV files found in the specified directory.")

    # Function to display images
    def display_images(image_paths, title):
        st.header(title)
        for img_path in image_paths:
            st.subheader(os.path.basename(img_path))
            image = Image.open(img_path)
            st.image(image, use_column_width=True)

    # Display PNG images from visualization folders
    if error_images:
        display_images(error_images, "Error Level Visualizations")

    if keyword_images:
        display_images(keyword_images, "Keyword Clustering Visualizations")

    if sentiment_images:
        display_images(sentiment_images, "Sentimental Analysis Visualizations")
