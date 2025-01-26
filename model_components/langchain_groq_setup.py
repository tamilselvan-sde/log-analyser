import pandas as pd
from langchain_groq import ChatGroq
from config.paths import MainPath  # Import MainPath for folder path configuration

# Path to the saved sentiment analysis CSV
sentiment_output_path = f"{MainPath.folder_path}/log_analyzer/log_analyser/model_outputs/sentiment_analysis_results.csv"

# Load the CSV file as a DataFrame
sentiment_df = pd.read_csv(sentiment_output_path)

# Filter messages where Sentiment is "Negative" and Confidence >= 0.70
filtered_messages = sentiment_df[
    (sentiment_df["Sentiment"] == "Negative") & (sentiment_df["Confidence"] >= 0.70)
]["message"]

# Initialize ChatGroq with your API key and model name
llm = ChatGroq(
    temperature=0,
    groq_api_key="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",  # Replace with your actual API key
    model_name="llama-3.1-70b-versatile"  # Model name
)

# Process each filtered message
for message in filtered_messages:
    # Assign the message to the error variable
    error = message
    print(f"Processing message: {error}")

    # Pass the error message to the model
    response = llm.invoke(f"what is this error, and how to solve it:\n{error}")

    # Print the output from the model
    print(f"Model Response: {response.content}")
