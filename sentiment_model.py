import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split
from transformers import EarlyStoppingCallback

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Türkçe BERT modeli
        self.model_name = "dbmdz/bert-base-turkish-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=4  # 4 sınıf: Positive, Negative, Neutral, Irrelevant
        )
        self.model.to(self.device)
        
        # Duygu etiketleri
        self.sentiment_mapping = {
            0: "Negative",
            1: "Neutral",
            2: "Positive",
            3: "Irrelevant"
        }
        
        # NLTK gerekli dosyaları
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # Küçük harfe çevirme
        text = text.lower()
        
        # URL'leri kaldırma
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Özel karakterleri kaldırma
        text = re.sub(r'[^\w\s]', '', text)
        
        # Fazla boşlukları kaldırma
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def prepare_data(self, train_data, val_data):
        # Etiketleri sayısallaştırma
        label_mapping = {
            'Negative': 0,
            'Neutral': 1,
            'Positive': 2,
            'Irrelevant': 3
        }
        
        # Veri setlerini hazırlama
        train_texts = train_data['text'].apply(self.clean_text).tolist()
        train_labels = train_data['sentiment'].map(label_mapping).tolist()
        
        val_texts = val_data['text'].apply(self.clean_text).tolist()
        val_labels = val_data['sentiment'].map(label_mapping).tolist()
        
        # Dataset oluşturma
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer)
        
        return train_dataset, val_dataset

    def train(self, train_dataset, val_dataset):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=5,  # Epoch sayısını artırdık
            per_device_train_batch_size=16,  # Batch size'ı artırdık
            per_device_eval_batch_size=16,
            warmup_steps=1000,  # Warmup steps'i artırdık
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            learning_rate=2e-5,  # Learning rate'i ayarladık
            fp16=True,  # Mixed precision training
        )
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.01
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[early_stopping]
        )
        
        trainer.train()
        
        # En iyi modeli kaydet
        self.model.save_pretrained('./best_model')
        self.tokenizer.save_pretrained('./best_model')

    def predict(self, text):
        # Metni temizle
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding='max_length'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Tahmin
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()
            confidence = torch.max(predictions).item()
        
        # Duygu etiketini ve güven skorunu döndür
        return self.sentiment_mapping[predicted_class], confidence

    def evaluate(self, val_data):
        predictions = []
        true_labels = []
        confidences = []
        
        for text, label in zip(val_data['text'], val_data['sentiment']):
            pred, conf = self.predict(text)
            predictions.append(pred)
            true_labels.append(label)
            confidences.append(conf)
        
        print("\nSınıflandırma Raporu:")
        print(classification_report(true_labels, predictions))
        
        # Ortalama güven skoru
        print(f"\nOrtalama Güven Skoru: {np.mean(confidences):.2f}")

def main():
    # Veri setlerini yükle
    print("Veri setleri yükleniyor...")
    train_data = pd.read_csv(cleaned_training.csv', header=None, 
                           names=['id', 'entity', 'sentiment', 'text'])
    val_data = pd.read_csv('twitter_validation.csv', header=None,
                         names=['id', 'entity', 'sentiment', 'text'])
    
    # Model oluştur
    print("Model yükleniyor...")
    analyzer = SentimentAnalyzer()
    
    # Veri setlerini hazırla
    print("Veri setleri hazırlanıyor...")
    train_dataset, val_dataset = analyzer.prepare_data(train_data, val_data)
    
    # Modeli eğit
    print("Model eğitiliyor...")
    analyzer.train(train_dataset, val_dataset)
    
    # Modeli değerlendir
    print("Model değerlendiriliyor...")
    analyzer.evaluate(val_data)
    
    # Test
    test_texts = [
        "Bu oyun gerçekten harika, çok eğlenceli!",
        "Bugün hava bulutlu ve sıcaklık 20 derece.",
        "Bu kadar kötü bir hizmet beklemiyordum.",
        "Yarın diş doktoruna gideceğim.",
        "Ürün beklentilerimin üzerinde çıktı, çok memnun kaldım.",
        "Toplantı yarın saat 14:00'te başlayacak.",
        "Müşteri hizmetleri hiç yardımcı olmadı, çok kızgınım.",
        "Marketten ekmek ve süt aldım."
    ]
    
    print("\nTest Sonuçları:")
    for text in test_texts:
        sentiment, confidence = analyzer.predict(text)
        print(f"Metin: {text}")
        print(f"Tahmin edilen duygu: {sentiment}")
        print(f"Güven skoru: {confidence:.2f}\n")
    
    return analyzer

if __name__ == "__main__":
    main() 