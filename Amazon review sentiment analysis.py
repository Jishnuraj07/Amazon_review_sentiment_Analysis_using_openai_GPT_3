import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from openai import AzureOpenAI

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint="Your_endpoint",
    api_key="your api key",
    api_version="apiversion"
)

# Read the CSV file into a DataFrame
file_path = r"filepath for csv data with reviewText column"
data = pd.read_csv(file_path)

# Process reviews in chunks
chunk_size = 50  # Number of reviews to process in each chunk
num_chunks = len(data) // chunk_size + 1

# Create a Streamlit app
st.title("Sentiment Analysis Results")

# Define emoji labels with text
emoji_labels = {
    "Joy/Happiness": "Joy/Happiness 😊",
    "Surprise": "Surprise 😮",
    "Neutral": "Neutral 😐",
    "Positive": "Positive 👍",
    "Sadness": "Sadness 😔",
    "Anger": "Anger 😡",
    "Fear": "Fear 😨",
    "Disgust": "Disgust 🤢",
    "Negative": "Negative 👎"
}

# Perform sentiment analysis and display results
for i in range(num_chunks):
    start_index = i * chunk_size
    end_index = min((i + 1) * chunk_size, len(data))
    chunk_reviews = ' '.join(data['reviewText'].iloc[start_index:end_index])

    # Create chat completion for each chunk
    completion = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[
            {"role": "user", "content": chunk_reviews},
            {"role": "assistant", "content": "Performing sentiment analysis on the provided based on the parameter Joy/Happiness, Surprise, Neutral, Positive, Sadness, Anger, Fear, Disgust, Negative and show in term of percentage ."}
        ],
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    # Extract sentiment analysis results
    response_text = completion.choices[0].message.content
    sentiment_results = {}

    # Parse sentiment results
    for line in response_text.split('\n'):
        parts = line.split(': ')
        if len(parts) == 2:
            sentiment_results[emoji_labels.get(parts[0], parts[0])] = float(parts[1].replace('%', ''))

    # Display sentiment analysis results
    st.subheader(f"Chunk {i + 1} Response:")

    # Create pie chart
    fig_pie = go.Figure(data=[go.Pie(labels=list(sentiment_results.keys()), values=list(sentiment_results.values()))])
    fig_pie.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Create bar chart with interactivity
    df_bar = pd.DataFrame.from_dict(sentiment_results, orient='index', columns=['Percentage'])
    df_bar.index.name = 'Sentiment'
    df_bar.reset_index(inplace=True)
    bar_chart = px.bar(df_bar, x='Sentiment', y='Percentage', title='Sentiment Analysis', labels={'Sentiment': 'Emotion', 'Percentage': 'Percentage (%)'})
    bar_chart.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                              marker_line_width=1.5, opacity=0.6)
    bar_chart.update_layout(clickmode='event+select')
    st.plotly_chart(bar_chart, use_container_width=True)
