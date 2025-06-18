import customtkinter as ctk
from sentiment_model import SentimentAnalyzer
import threading
import time

class SentimentApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Pencere ayarları
        self.title("Modern Duygu Analizi")
        self.geometry("900x700")
        
        # Tema ayarları
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Model yükleme
        self.model = None
        self.loading = True
        
        # Arayüz oluşturma
        self.create_widgets()
        
        # Model yükleme thread'i
        threading.Thread(target=self.load_model, daemon=True).start()

    def create_widgets(self):
        # Ana frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Başlık
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Modern Duygu Analizi",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.title_label.pack(pady=20)
        
        # Metin girişi
        self.text_input = ctk.CTkTextbox(
            self.main_frame,
            height=200,
            font=ctk.CTkFont(size=14)
        )
        self.text_input.pack(fill="x", padx=20, pady=10)
        self.text_input.insert("1.0", "Duygu analizi yapılacak metni buraya girin...")
        
        # Butonlar
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.pack(fill="x", padx=20, pady=10)
        
        self.analyze_button = ctk.CTkButton(
            self.button_frame,
            text="Analiz Et",
            command=self.analyze_sentiment,
            state="disabled",
            height=40,
            font=ctk.CTkFont(size=16)
        )
        self.analyze_button.pack(side="left", padx=5)
        
        self.clear_button = ctk.CTkButton(
            self.button_frame,
            text="Temizle",
            command=self.clear_text,
            height=40,
            font=ctk.CTkFont(size=16)
        )
        self.clear_button.pack(side="left", padx=5)
        
        # Sonuç alanı
        self.result_frame = ctk.CTkFrame(self.main_frame)
        self.result_frame.pack(fill="x", padx=20, pady=10)
        
        self.result_label = ctk.CTkLabel(
            self.result_frame,
            text="Analiz Sonucu:",
            font=ctk.CTkFont(size=18)
        )
        self.result_label.pack(pady=10)
        
        self.result_text = ctk.CTkLabel(
            self.result_frame,
            text="",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.result_text.pack(pady=5)
        
        # Güven skoru
        self.confidence_label = ctk.CTkLabel(
            self.result_frame,
            text="Güven Skoru:",
            font=ctk.CTkFont(size=16)
        )
        self.confidence_label.pack(pady=5)
        
        self.confidence_text = ctk.CTkLabel(
            self.result_frame,
            text="",
            font=ctk.CTkFont(size=20)
        )
        self.confidence_text.pack(pady=5)
        
        # Örnek metinler
        self.examples_frame = ctk.CTkFrame(self.main_frame)
        self.examples_frame.pack(fill="x", padx=20, pady=10)
        
        self.examples_label = ctk.CTkLabel(
            self.examples_frame,
            text="Örnek Metinler:",
            font=ctk.CTkFont(size=16)
        )
        self.examples_label.pack(pady=5)
        
        examples = [
            "Bu oyun gerçekten harika, çok eğlenceli!",
            "Bugün hava bulutlu ve sıcaklık 20 derece.",
            "Bu kadar kötü bir hizmet beklemiyordum.",
            "Yarın diş doktoruna gideceğim."
        ]
        
        for example in examples:
            example_button = ctk.CTkButton(
                self.examples_frame,
                text=example,
                command=lambda t=example: self.use_example(t),
                height=30,
                font=ctk.CTkFont(size=12)
            )
            example_button.pack(pady=2)
        
        # Yükleniyor göstergesi
        self.loading_label = ctk.CTkLabel(
            self.main_frame,
            text="Model yükleniyor...",
            font=ctk.CTkFont(size=14)
        )
        self.loading_label.pack(pady=10)

    def load_model(self):
        try:
            self.model = SentimentAnalyzer()
            self.loading = False
            self.analyze_button.configure(state="normal")
            self.loading_label.configure(text="Model hazır!")
        except Exception as e:
            self.loading_label.configure(text=f"Hata: {str(e)}")

    def analyze_sentiment(self):
        if self.loading:
            return
        
        text = self.text_input.get("1.0", "end-1c")
        if not text.strip():
            self.result_text.configure(text="Lütfen bir metin girin!")
            return
        
        try:
            sentiment, confidence = self.model.predict(text)
            self.result_text.configure(text=sentiment)
            
            # Renk ayarları
            colors = {
                "Positive": "#4CAF50",  # Yeşil
                "Negative": "#f44336",  # Kırmızı
                "Neutral": "#2196F3",   # Mavi
                "Irrelevant": "#9E9E9E" # Gri
            }
            self.result_text.configure(text_color=colors.get(sentiment, "#FFFFFF"))
            
            # Güven skoru
            self.confidence_text.configure(
                text=f"{confidence:.2%}",
                text_color=colors.get(sentiment, "#FFFFFF")
            )
            
        except Exception as e:
            self.result_text.configure(text=f"Hata: {str(e)}")

    def clear_text(self):
        self.text_input.delete("1.0", "end")
        self.result_text.configure(text="")
        self.confidence_text.configure(text="")

    def use_example(self, text):
        self.text_input.delete("1.0", "end")
        self.text_input.insert("1.0", text)
        self.analyze_sentiment()

def main():
    app = SentimentApp()
    app.mainloop()

if __name__ == "__main__":
    main() 