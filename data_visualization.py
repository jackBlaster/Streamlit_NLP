import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_word_embeddings(model, num_words_to_display=50, title="Word Embeddings Visualization (English)"):
    """
    Visualise les vecteurs de mots en réduisant leur dimensionnalité avec PCA.
    
    Paramètres :
    - model : modèle Word2Vec contenant les embeddings.
    - num_words_to_display : nombre de mots à afficher.
    - title : titre du graphique.
    """
    # Récupération des mots et de leurs vecteurs
    words = list(model.wv.index_to_key)
    word_vectors = [model.wv[word] for word in words]

    # Réduction de la dimensionnalité avec PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(word_vectors)

    # Limiter à un sous-ensemble de mots pour une meilleure visualisation
    words_subset = words[:num_words_to_display]
    pca_result_subset = pca_result[:num_words_to_display]

    # Création du graphique
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_result_subset[:, 0], pca_result_subset[:, 1], alpha=0.7)

    for i, word in enumerate(words_subset):
        plt.annotate(word, (pca_result_subset[i, 0], pca_result_subset[i, 1]), fontsize=8, alpha=0.7)

    plt.title(title)
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Exemple d'utilisation avec un modèle Word2Vec anglais
    from gensim.models import Word2Vec  # Assurez-vous que le modèle est déjà formé ou chargé

    # Charger un modèle pré-entraîné ou entraîner un nouveau modèle
    model_w2v_en = Word2Vec.load("model_w2v_en.model")  # Remplacez par le chemin de votre modèle

    # Visualisation des embeddings anglais
    visualize_word_embeddings(model_w2v_en, num_words_to_display=50, title="Word Embeddings Visualization (English)")
