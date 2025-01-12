import re
import pandas as pd
from symspellpy import SymSpell, Verbosity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
import nltk

# Téléchargement des stopwords pour NLTK
nltk.download('stopwords')

# Initialisation de SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

# 1. Nettoyage des textes
def clean_text(text):
    """
    Nettoie le texte en supprimant les espaces, les ponctuations et en mettant en minuscules.
    """
    text = text.strip()
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Supprime la ponctuation
    return text

# 2. Correction orthographique
def correct_spelling_fast(text):
    """
    Corrige l'orthographe d'un texte en anglais en utilisant SymSpell.
    """
    words = text.split()
    corrected_words = []
    for word in words:
        # Utilisation de SymSpell pour corriger chaque mot
        correction = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected_words.append(correction[0].term if correction else word)
    return ' '.join(corrected_words)

# 3. Extraction des mots fréquents
def frequent_words(df, column, lang='en', ngram_range=(1, 1), top_n=10):
    """
    Extrait les mots ou n-grammes les plus fréquents dans une colonne de texte.
    """
    if lang == 'en':
        stop_words = stopwords.words('english')
    else:
        stop_words = None

    vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range)
    X = vectorizer.fit_transform(df[column])

    word_freq = X.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()
    word_freq_dict = dict(zip(words, word_freq))

    # Retourner les n mots les plus fréquents
    sorted_words = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:top_n]

# 4. Topic modeling avec LDA
def topic_modeling(df, column, n_topics=5):
    """
    Applique le topic modeling (LDA) sur une colonne de texte et génère des noms pour les sujets.
    """
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df[column])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    terms = vectorizer.get_feature_names_out()

    def get_topic_name(topic_idx):
        topic_words = lda.components_[topic_idx]
        sorted_word_idx = topic_words.argsort()[-10:]
        topic_keywords = [terms[i] for i in sorted_word_idx]
        return " ".join(topic_keywords[:3])

    for idx, topic in enumerate(lda.components_):
        topic_name = get_topic_name(idx)
        print(f"Topic {idx + 1} ({topic_name}):")
        print([terms[i] for i in topic.argsort()[-10:]])
        print()

    return lda

# 5. Pipeline principal
if __name__ == "__main__":
    # Charger le dataset
    df = pd.read_csv("data.csv")  # Remplacez par le chemin de votre fichier CSV

    # Nettoyage et correction orthographique
    df['avis_cor_en'] = df['avis_en'].apply(lambda x: correct_spelling_fast(clean_text(x)))

    # Sauvegarder les résultats de nettoyage
    df.to_csv('avis_corriges.csv', index=False)

    # Extraction des mots fréquents
    top_10_words_en = frequent_words(df, 'avis_cor_en', lang='en', ngram_range=(1, 1), top_n=10)
    print("Top 10 frequent words in English:", top_10_words_en)

    # Topic modeling
    lda_model = topic_modeling(df, 'avis_cor_en', n_topics=5)
