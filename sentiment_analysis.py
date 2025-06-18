import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji
import logging
import pickle
import warnings
warnings.filterwarnings('ignore')

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)

# NLTK kaynaklarını indir
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Duygu eşleştirme sözlükleri
sentiment_mapping = {
    'Positive': 0,
    'Negative': 1,
    'Neutral': 2,
    'Irrelevant': 3
}

inverse_mapping = {v: k for k, v in sentiment_mapping.items()}

def tokenize(text):
    return text.split()

def clean_urls(text):
    """URL'leri temizle"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, '', text)

def clean_emojis(text):
    """Emoji'leri metne dönüştür"""
    return emoji.demojize(text, delimiters=(' ', ' '))

def clean_special_chars(text):
    """Özel karakterleri temizle"""
    # Noktalama işaretlerini boşluğa çevir
    text = re.sub(r'[^\w\s]', ' ', text)
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_text(text):
    """Metni ön işle"""
    if not isinstance(text, str):
        return ""
    
    # Küçük harfe çevir
    text = text.lower()
    
    # URL'leri temizle
    text = clean_urls(text)
    
    # Emoji'leri metne dönüştür
    text = clean_emojis(text)
    
    # Özel karakterleri temizle
    text = clean_special_chars(text)
    
    # Tokenize et
    tokens = text.split()
    
    # Stopwords'leri kaldır
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Negasyon işleme
    negations = {'not', 'no', 'never', 'none', 'neither', 'nor', 'cannot'}
    for i, token in enumerate(tokens):
        if token in negations and i + 1 < len(tokens):
            tokens[i + 1] = 'not_' + tokens[i + 1]
    
    return ' '.join(tokens)

def load_data(file_path):
    """Veri setini yükle ve ön işle"""
    try:
        # Sütun isimleri olmadan yükle
        df = pd.read_csv(file_path, header=None, names=['id', 'entity', 'sentiment', 'text'])
        logging.info(f"Veri seti yüklendi: {len(df)} örnek")
        
        # Metinleri temizle
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        
        # Boş metinleri filtrele
        df = df[df['cleaned_text'].str.len() > 0]
        
        # Sınıf dağılımını kontrol et
        class_distribution = df['sentiment'].value_counts()
        logging.info("\nSınıf dağılımı:\n" + str(class_distribution))
        
        return df
    except Exception as e:
        logging.error(f"Veri yükleme hatası: {str(e)}")
        raise

def train_model(X_train, y_train):
    """Modeli eğit"""
    try:
        # Update TF-IDF vectorizer settings to use the normal function
        vectorizer = TfidfVectorizer(max_features=10000, stop_words=None, min_df=1, tokenizer=tokenize)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        
        # Model eğitimi
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_tfidf, y_train)
        
        return model, vectorizer
    except Exception as e:
        logging.error(f"Model eğitimi hatası: {str(e)}")
        raise

def save_model(model, vectorizer, filepath='sentiment_model.pkl'):
    """Modeli kaydet"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump((model, vectorizer), f)
        logging.info(f"Model kaydedildi: {filepath}")
    except Exception as e:
        logging.error(f"Model kaydetme hatası: {str(e)}")
        raise

def load_model(filepath='sentiment_model.pkl'):
    """Modeli yükle"""
    try:
        with open(filepath, 'rb') as f:
            model, vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        logging.error(f"Model yükleme hatası: {str(e)}")
        raise

def predict_sentiment(text, model=None, vectorizer=None):
    try:
        if model is None or vectorizer is None:
            model, vectorizer = load_model()
        # Tip kontrolü
        if isinstance(vectorizer, str):
            raise TypeError("Vectorizer yüklenemedi veya yanlış tipte.")
        cleaned_text = preprocess_text(text)
        text_tfidf = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_tfidf)[0]
        return inverse_mapping[prediction]
    except Exception as e:
        logging.error(f"Tahmin hatası: {str(e)}")
        raise

def main():
    try:
        # Veri setlerini yükle
        train_df = load_data('cleaned_training.csv')
        val_df = load_data('twitter_validation.csv')
        
        # Eğitim ve test verilerini ayır
        X_train = train_df['cleaned_text']
        y_train = train_df['sentiment'].map(sentiment_mapping)
        # NaN içeren satırları filtrele
        mask = ~y_train.isna()
        X_train = X_train[mask]
        y_train = y_train[mask]
        
        X_val = val_df['cleaned_text']
        y_val = val_df['sentiment'].map(sentiment_mapping)
        mask_val = ~y_val.isna()
        X_val = X_val[mask_val]
        y_val = y_val[mask_val]
        
        # Modeli eğit
        logging.info("Model eğitiliyor...")
        model, vectorizer = train_model(X_train, y_train)
        
        # Modeli değerlendir
        logging.info("Model değerlendiriliyor...")
        y_pred = model.predict(vectorizer.transform(X_val))
        
        # Sadece veri setinde bulunan etiketleri ve isimleri kullan
        unique_labels = sorted(np.unique(np.concatenate((y_val, y_pred))))
        target_names = [inverse_mapping[i] for i in unique_labels]
        
        # Sınıflandırma raporu
        logging.info("\nSınıflandırma Raporu:")
        logging.info(classification_report(y_val, y_pred, labels=unique_labels, target_names=target_names))
        
        # Confusion matrix
        logging.info("\nConfusion Matrix:")
        logging.info(confusion_matrix(y_val, y_pred, labels=unique_labels))
        
        # Modeli kaydet
        save_model(model, vectorizer)
        
        # Test
        test_text = "I feel bad"
        prediction = predict_sentiment(test_text)
        print(f"\nTest metni: {test_text}")
        print(f"Tahmin edilen duygu: {prediction}")
        
    except Exception as e:
        logging.error(f"Ana program hatası: {str(e)}")
        raise

if __name__ == "__main__":
    main()
