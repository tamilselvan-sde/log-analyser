import os
import pandas as pd
import json
import logging
import streamlit as st
from PIL import Image
from datetime import datetime
from time import sleep
from dotenv import load_dotenv
from logs_data.log_generator import LogGenerator
from config.paths import MainPath
from logs_preprocessor import LogParser
from model_components.SentimentAnalysis import SentimentAnalysis
from model_components.log_sensitive_data_parser import SensitiveDataParser
from model_components.clustering_keywords import KeywordClustering
from model_components.AnomalyPrediction import AnomalyPrediction
from model_components.Root_Cause_Analysis import RootCauseAnalysis
from model_components.Threshold_Alert import ThresholdAlert
from model_components.DBSCAN_Clustering import DBSCANClustering
from model_components.HDBSCAN_Clustering import HDBSCANClustering
from Plot_Analysis import LogLevelVisualizer, SentimentVisualizer, KeywordClusteringVisualizer
from groq import Groq

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    filename="log_analyzer.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Paths to folders
folder_path = f"{MainPath.folder_path}/log_analyser/model_outputs"
error_level_path = f"{MainPath.folder_path}/log_analyser/visualization/error_level"
keyword_clustering_path = f"{MainPath.folder_path}/log_analyser/visualization/keyword_clustering"
sentimental_analysis_path = f"{MainPath.folder_path}/log_analyser/visualization/sentimental_analysis"

# Function to save DataFrame output

def save_output(df: pd.DataFrame, output_path: str, filename: str) -> None:
    csv_path = os.path.join(output_path, f"{filename}.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved CSV output: {csv_path}")

    json_path = os.path.join(output_path, f"{filename}.json")
    df.to_json(json_path, orient="records", lines=True)
    logging.info(f"Saved JSON output: {json_path}")


def save_json_output(data: dict | str, output_path: str, filename: str) -> None:
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON string: {e}")
            return

    json_path = os.path.join(output_path, f"{filename}.json")
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    logging.info(f"Saved JSON output: {json_path}")

# Groq Error Troubleshooting

def troubleshoot_error(api_key: str, error_message: str):
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

# Streamlit Dashboard

def streamlit_dashboard():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Troubleshooting", "Plots and DataFrames"])

    if selected_page == "Troubleshooting":
        st.title("Troubleshooting Guide (Groq)")
        st.markdown("---")

        # Path to sentiment analysis CSV
        sentiment_output_path = os.path.join(folder_path, "sentiment_analysis_results.csv")

        try:
            sentiment_df = pd.read_csv(sentiment_output_path)
        except Exception as e:
            st.error(f"Error loading sentiment analysis file: {e}")
            return

        filtered_messages = sentiment_df[
            (sentiment_df["Sentiment"] == "Negative") & (sentiment_df["Confidence"] > 0.71)
        ]["message"]

        if filtered_messages.empty:
            st.warning("No negative sentiment messages with confidence greater than 0.71 found.")
        else:
            api_key = st.text_input("Enter your Groq API Key:", type="password")
            if st.button("🚀 Analyze Errors"):
                if not api_key:
                    st.error("Please provide your Groq API Key.")
                else:
                    for idx, message in enumerate(filtered_messages, start=1):
                        st.subheader(f"Error {idx}:")
                        st.write(f"**Error Message:** {message}")

                        try:
                            solution = troubleshoot_error(api_key, message)
                            sleep(3)  # Add 3-second delay between API calls
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

    elif selected_page == "Plots and DataFrames":
        st.title("Log Analyzer Outputs Viewer")
        st.markdown("---")

        files = os.listdir(folder_path)
        json_files = [file for file in files if file.endswith(".json")]
        csv_files = [file for file in files if file.endswith(".csv")]

        error_images = [os.path.join(error_level_path, img) for img in os.listdir(error_level_path) if img.endswith(".png")]
        keyword_images = [os.path.join(keyword_clustering_path, img) for img in os.listdir(keyword_clustering_path) if img.endswith(".png")]
        sentiment_images = [os.path.join(sentimental_analysis_path, img) for img in os.listdir(sentimental_analysis_path) if img.endswith(".png")]

        def display_file(file_path):
            if file_path.endswith(".json"):
                try:
                    with open(file_path, "r") as f:
                        content = f.read()
                        try:
                            data = json.loads(content)
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

        def display_images(image_paths, title):
            st.header(title)
            for img_path in image_paths:
                st.subheader(os.path.basename(img_path))
                image = Image.open(img_path)
                st.image(image, use_column_width=True)

        if error_images:
            display_images(error_images, "Error Level Visualizations")

        if keyword_images:
            display_images(keyword_images, "Keyword Clustering Visualizations")

        if sentiment_images:
            display_images(sentiment_images, "Sentimental Analysis Visualizations")

if __name__ == "__main__":
    streamlit_dashboard()
