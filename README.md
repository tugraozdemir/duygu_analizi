# Duygu Analizi Projesi

Bu proje, metinlerdeki duyguları (pozitif, negatif, nötr vb.) analiz etmek için geliştirilmiştir. Python ile yazılmıştır ve hem PyQt5 hem de CustomTkinter arayüzleriyle kullanılabilir.

## İçerik
- `sentiment_model.py`: Duygu analizi modeli ve eğitim kodları
- `sentiment_analysis.py`: Klasik makine öğrenmesi ile duygu analizi
- `sentiment_app.py`: PyQt5 ile hazırlanmış masaüstü uygulaması
- `sentiment_app_new.py`: CustomTkinter ile hazırlanmış alternatif masaüstü uygulaması
- `sentiment_model.pkl`: Eğitilmiş model dosyası
- `cleaned_training.csv`, `twitter_validation.csv`: Eğitim ve doğrulama veri setleri
- `requirements.txt`: Gerekli Python kütüphaneleri

## Kurulum
1. Gerekli kütüphaneleri yükleyin:
   ```sh
   pip install -r requirements.txt
   ```
2. (İsteğe bağlı) NLTK veri paketlerini indirin:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Kullanım
- PyQt5 arayüzü için:
  ```sh
  python sentiment_app.py
  ```
- CustomTkinter arayüzü için:
  ```sh
  python sentiment_app_new.py
  ```

## Notlar
- Modeli yeniden eğitmek için `sentiment_model.py` dosyasını kullanabilirsiniz.
- Kendi veri setinizle eğitmek için CSV dosyalarını güncelleyebilirsiniz.

## Lisans
Bu proje eğitim amaçlıdır. 