import streamlit as st
import pandas as pd
import json
from preprocessing import *
from models import *

# Define a function for each page
def page_accueil():
    st.title("Project Overview")
    st.write("""
Welcome to this interactive data science project! In this application, we will guide you through several key steps of a data science workflow.\n

**Phase 1: Data Exploration and Preparation**\n
- Clean the data: Handle missing values, outliers, and inconsistencies.\n
- Visualize the data: Create insightful visualizations to uncover patterns, trends, and relationships.\n
- Draw initial conclusions based on the analysis.\n

**Phase 2: Supervised and Unsupervised Learning**\n
- Build a supervised text processing model using NLP techniques.\n
- Create an interactive app for users to submit text and get predictions.\n

We hope this project will help you explore essential data science concepts and build real-world applications.
    """)
    st.write("This project was done by Jade BETTOYA and Thibault BIVILLE (DIA 1 Group).")

def preprocessing_page():
    st.title("Preprocessing Steps and Explanations")
    st.subheader("Raw Data")
    st.write("""
The data used for this project consists of several Excel files. Here's a sample of the first 10 rows in the DataFrame after concatenation:
    """)
    data_concat = pd.read_csv('./cleaned_data/concat_data.csv', delimiter=',')
    st.dataframe(data_concat.head(10))

    st.subheader("Cleaning Steps")
    st.write("""
**The cleaning steps included:**\n 
1. Filling NaN values in the 'Author' column with 'Unknown'.\n
2. Translating French reviews into English using the `deep_translator` library.\n
3. Deleting duplicates and normalizing text (lowercase, punctuation removal).\n
4. Correcting typos using `SymSpell` with English and French dictionaries.\n
5. Removing stopwords for better focus on relevant words.\n

After cleaning, here's a sample of the dataset:
    """)
    data_clean = pd.read_csv('./cleaned_data/cleaned_rev.csv', delimiter=',')
    st.dataframe(data_clean.head(10))

def topic_page():
    st.title("Topic Modeling & List of Topics")
    st.subheader("Topic Modeling")
    st.write("""
We applied LDA (Latent Dirichlet Allocation) to the corrected English reviews to detect 5 topics. Here are the words associated with each topic:
    """)
    with open('./cleaned_data/topics_word.json', 'r') as file:
        json_data = json.load(file)

    topics = [f"Topic {i+1}" for i in range(len(json_data))]
    df_tw = pd.DataFrame(json_data)
    df_tw.insert(0, 'Topic', topics)
    st.table(df_tw.set_index('Topic'))

    st.subheader('List of Topics')
    st.write("""
Using ChatGPT to analyze topic keywords, we named each topic for better understanding:
    """)
    with open('./cleaned_data/topics_name.json', 'r') as file:
        json_name = json.load(file)

    df_tn = pd.DataFrame(list(json_name.items()), columns=["Topic", "Description"])
    st.table(df_tn.set_index('Topic'))

def topic_classif_page():
    st.title("Topic Classification")
    st.subheader("Zero-Shot Model for Topic Classification")
    st.write("""
After obtaining our various possible topics in the previous step, we saved this data in a [JSON file](./cleaned_data/topics_name.json) 
to keep it for later use in the topic classification page.\n
In order to classify our reviews, we will use a **Zero-shot classification model** via Hugging Face. This model is based on **BART-Large**, 
which has been trained on the **MultiNLI dataset**. This enables the model to perform classification without requiring task-specific training data.

For more details on the BART model and its capabilities, you can visit the [Hugging Face BART page](https://huggingface.co/facebook/bart-large-mnli).\n

The MultiNLI dataset can be explored in more detail through its [official page](https://www.nyu.edu/projects/bowman/multinli/).\n
""")
    
    st.subheader("Try it Yourself ! :")
    # Saisie de l'avis utilisateur
    review = st.text_area("Enter your review for topic classification:")

    if st.button("Classify Topic"):
        if review.strip():
            # Appeler la fonction de classification
            with st.spinner("Classifying the review..."):

                result = call_zero_shot_topic(review)
            
            # Afficher les résultats
            st.write("**Input Review:** ", review)
            st.write("**Predicted Topic:** ", result["labels"][0])
            scores_df = pd.DataFrame({
                'Topic': result["labels"],
                'Confidence Score': result["scores"]
            })

            # Display the confidence scores in a table
            st.subheader("Confidence Scores")
            st.table(scores_df.set_index('Topic'))
        else:
            st.error("Please enter a valid review.")

def star_predict_page():
    st.title("Star Rating Prediction")
    st.subheader("How we Implemented the Star Prediction")
    st.write("""
To predict star ratings from user reviews, we developed a natural language processing (NLP) pipeline that leverages a pre-trained transformer model. This model is fine-tuned to classify reviews based on their content and assign them a rating in the form of stars. Here is a detailed explanation of how we implemented this feature:

### 1. **Data Preparation**:
   - **Training Data**: We first collected a dataset of user reviews with corresponding star ratings. The reviews provide textual descriptions of the user experience, while the star ratings serve as the target variable.
   - **Text Preprocessing**: We used the cleaned data produced during the previous steps of the project, for this part the columns 'note' and 'avis_cor_en' have been used.
             
### 2. **Model Selection and Fine-tuning**:
   - **Transformer Model**: We used a transformer-based model, **DistilBERT**, which is a smaller, faster version of BERT, designed for text classification tasks. This model is capable of understanding the semantic meaning of the text and can make predictions based on the context.
   - **Fine-tuning**: We fine-tuned the pre-trained **DistilBERT** model on our dataset of reviews and star ratings. This step ensures that the model learns the specific patterns in the data that correlate with star ratings.

### 3. **Saving the Model**:
   - To ensure fast performance and avoid retraining the model every time, we saved the fine-tuned model in a directory. By doing this, we can easily reload the model whenever necessary without the need for expensive retraining. We use Hugging Face's `save_pretrained` method to save the model along with its tokenizer.
   - We then load the saved model into cache during the app startup. Caching the model ensures faster loading times, making the app more responsive when users enter their reviews.

### 4. **Star Prediction**:
   - **Prediction**: It is passed through the fine-tuned **DistilBERT** model, which outputs the predicted star rating. The model predicts a class corresponding to a star rating, where the class is mapped to a star count (e.g., class 0 = 1 star, class 1 = 2 stars, etc.).

""")
    st.write("Enter a review, and the model will predict the star rating.")
    review = st.text_area("Enter your review:")
    if st.button("Predict"):
        if review.strip():
            predicted_class = process_input_star(review)  # Call the prediction function
            st.write(f"Predicted star rating: {predicted_class + 1} stars")
        else:
            st.write("Please enter a valid review.")

# Sidebar Navigation
st.sidebar.title("Navigation")
if "page" not in st.session_state:
    st.session_state.page = "Accueil"

# Update session state based on sidebar buttons
if st.sidebar.button("Project Overview"):
    st.session_state.page = "Accueil"
if st.sidebar.button("Preprocessing"):
    st.session_state.page = "Preprocessing"
if st.sidebar.button("Topic Modeling"):
    st.session_state.page = "Topic Modeling"
if st.sidebar.button("Topic Classification"):
    st.session_state.page = "Topic Classification"
if st.sidebar.button("Star Rating"):
    st.session_state.page = "Star Rating"


# Render the selected page
if st.session_state.page == "Accueil":
    page_accueil()
elif st.session_state.page == "Preprocessing":
    preprocessing_page()
elif st.session_state.page == "Topic Modeling":
    topic_page()
elif st.session_state.page == "Topic Classification":
    topic_classif_page()
elif st.session_state.page == "Star Rating":
    star_predict_page()
