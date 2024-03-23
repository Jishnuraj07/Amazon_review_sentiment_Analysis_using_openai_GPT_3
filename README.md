

# Sentiment Analysis Streamlit App

This project is a Streamlit web application for performing sentiment analysis on a dataset of reviews using the OpenAI GPT-3.5 model via AzureopenAI API key using Chat playground. The sentiment analysis results are visualized using pie charts and bar charts, with emojis representing different sentiment labels.

## Features

- **Sentiment Analysis**: The app processes reviews in chunks, performs sentiment analysis using the GPT-3.5 model, and displays the results in terms of percentages for various sentiment categories.
- **Visualizations**: It provides visualizations in the form of pie charts and interactive bar charts, allowing users to explore sentiment distributions easily.
- **Emoji Representation**: Sentiment labels in the visualizations are represented using emojis, making the charts more intuitive and engaging.

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/<username>/<repository>.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Replace the placeholder CSV file path with the path to your dataset in the `app.py` file.

4. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

5. Explore the sentiment analysis results and visualizations generated by the app.

## Dependencies

- `pandas`: For data manipulation and analysis.
- `streamlit`: For building interactive web applications.
- `plotly`: For creating interactive and publication-quality graphs.
- `openai`: For integrating with the OpenAI GPT-3.5 model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
