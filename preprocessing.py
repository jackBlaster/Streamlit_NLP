import pandas as pd
from deep_translator import GoogleTranslator

def preprocess_data(df):
    """
    Fonction de prétraitement des données.
    Remplit les valeurs manquantes, traduit les avis en anglais et supprime les doublons.

    Paramètres :
        df (pd.DataFrame) : Le DataFrame contenant les données.

    Retourne :
        pd.DataFrame : Le DataFrame prétraité.
    """
    # Remplir les valeurs manquantes dans la colonne 'auteur'
    df['auteur'] = df['auteur'].fillna('Inconnu')

    # Supprimer les lignes où la colonne 'note' est null
    df = df.dropna(subset=['note'])

    # Filtrer les lignes avec des avis_en nulls
    avis_to_translate = df[df['avis_en'].isnull()]

    # Fonction pour traduire un avis en français vers l'anglais
    def translate_avis(avis):
        return GoogleTranslator(source='fr', target='en').translate(avis)

    # Traduire les avis en français vers l'anglais pour les lignes avec avis_en nulls
    df.loc[df['avis_en'].isnull(), 'avis_en'] = avis_to_translate['avis'].apply(translate_avis)

    # Supprimer les doublons
    duplicates_count = df.duplicated().sum()
    print(f"Nombre de doublons dans le DataFrame : {duplicates_count}")
    df = df.drop_duplicates()

    return df

# Exemple d'utilisation
# df = pd.read_csv('votre_fichier.csv')
# df = preprocess_data(df)
# df.to_csv('fichier_pretraite.csv', index=False)
