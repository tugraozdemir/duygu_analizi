import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTextEdit, QPushButton, QLabel, 
                           QComboBox, QMessageBox, QGraphicsDropShadowEffect, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QPixmap, QClipboard
import sentiment_analysis
import random

# Emoji eşleştirmesi
SENTIMENT_EMOJIS = {
    'Positive': '😊',
    'Negative': '😞',
    'Neutral': '😐',
    'Irrelevant': '❓'
}

def tokenize(text):
    return text.split()

class SentimentAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Duygu Analizi Uygulaması")
        self.setGeometry(100, 100, 800, 600)
        self.history = []
        self.dark_mode = False
        self.loading = False
        
        # Ana widget ve layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        outer_layout = QVBoxLayout(main_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Üst bar: yardım ve karanlık mod
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(10)
        top_bar.addStretch(1)
        self.dark_mode_toggle = QCheckBox("Karanlık Mod")
        self.dark_mode_toggle.stateChanged.connect(self.toggle_dark_mode)
        top_bar.addWidget(self.dark_mode_toggle)
        help_btn = QPushButton("?")
        help_btn.setFixedSize(28, 28)
        help_btn.setStyleSheet("font-weight: bold; font-size: 18px; border-radius: 14px; background: #e0e0e0;")
        help_btn.clicked.connect(self.show_help)
        top_bar.addWidget(help_btn)
        outer_layout.addLayout(top_bar)

        # Ortalanmış kart paneli
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(40, 40, 40, 40)
        center_layout.setSpacing(25)
        center_widget.setObjectName("centerCard")
        # Drop shadow effect for card
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 40))
        center_widget.setGraphicsEffect(shadow)
        outer_layout.addStretch(1)
        outer_layout.addWidget(center_widget, alignment=Qt.AlignCenter)
        outer_layout.addStretch(1)

        # Başlık
        title = QLabel("Duygu Analizi Uygulaması")
        title.setFont(QFont('Segoe UI', 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("mainTitle")
        center_layout.addWidget(title)

        # Alt başlık
        subtitle = QLabel("Bir metnin duygusunu anında analiz edin!")
        subtitle.setFont(QFont('Segoe UI', 12))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666; margin-bottom: 10px;")
        center_layout.addWidget(subtitle)

        # Metin girişi alanı
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Duygu analizi yapılacak metni buraya girin...")
        self.text_input.setMinimumHeight(120)
        self.text_input.setObjectName("textInput")
        center_layout.addWidget(self.text_input)

        # Buton alanı
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        self.analyze_button = QPushButton("Analiz Et")
        self.analyze_button.setObjectName("analyzeButton")
        self.analyze_button.clicked.connect(self.analyze_sentiment)
        self.clear_button = QPushButton("Temizle")
        self.clear_button.setObjectName("clearButton")
        self.clear_button.clicked.connect(self.clear_text)
        # Yükleniyor etiketi
        self.loading_label = QLabel()
        self.loading_label.setFixedWidth(24)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.hide()
        button_layout.addWidget(self.analyze_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.loading_label)
        center_layout.addLayout(button_layout)

        # Sonuç alanı (kart şeklinde)
        result_card = QWidget()
        result_card.setObjectName("resultCard")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(20, 15, 20, 15)
        result_layout.setSpacing(8)
        result_label = QLabel("Analiz Sonucu:")
        result_label.setFont(QFont('Segoe UI', 11, QFont.Bold))
        result_label.setStyleSheet("color: #555;")
        # Sonuç + emoji
        self.result_text = QLabel("")
        self.result_text.setFont(QFont('Segoe UI', 16, QFont.Bold))
        self.result_text.setAlignment(Qt.AlignCenter)
        self.result_text.setObjectName("resultText")
        # Rastgele cümle butonu
        self.random_sentence_button = QPushButton("Rastgele Cümle")
        self.random_sentence_button.setObjectName("randomSentenceButton")
        self.random_sentence_button.setMinimumWidth(170)
        self.random_sentence_button.setMaximumWidth(220)
        self.random_sentence_button.setStyleSheet("font-size: 15px; padding-left: 10px; padding-right: 10px;")
        self.random_sentence_button.clicked.connect(self.insert_random_sentence)
        # Sonuç ve rastgele cümle yatay
        result_hbox = QHBoxLayout()
        result_hbox.addWidget(self.result_text, 1)
        result_hbox.addWidget(self.random_sentence_button, 0)
        result_layout.addWidget(result_label)
        result_layout.addLayout(result_hbox)
        center_layout.addWidget(result_card)

        # Geçmiş analizler
        self.history_label = QLabel("Son Analizler:")
        self.history_label.setFont(QFont('Segoe UI', 10, QFont.Bold))
        self.history_label.setStyleSheet("color: #888;")
        self.history_label.hide()
        self.history_list = QLabel("")
        self.history_list.setFont(QFont('Segoe UI', 10))
        self.history_list.setStyleSheet("color: #666;")
        self.history_list.setWordWrap(True)
        center_layout.addWidget(self.history_label)
        center_layout.addWidget(self.history_list)

        # Stil ayarları (daha modern)
        self.setStyleSheet(self.get_stylesheet())
        
        # Duygu renkleri
        self.sentiment_colors = {
            'Positive': '#4CAF50',  # Yeşil
            'Negative': '#f44336',  # Kırmızı
            'Neutral': '#2196F3',   # Mavi
            'Irrelevant': '#9E9E9E' # Gri
        }

    def get_stylesheet(self):
        if self.dark_mode:
            return """
            QMainWindow { background-color: #23272e; }
            #centerCard { background: #23272e; border-radius: 18px; min-width: 400px; max-width: 520px; }
            #mainTitle { color: #fff; margin-bottom: 0px; }
            #textInput { border: 2px solid #444; border-radius: 8px; padding: 10px; background-color: #2c313a; color: #eee; font-size: 15px; }
            #analyzeButton, #clearButton, #randomSentenceButton { background-color: #4F8EF7; color: white; border: none; padding: 12px 28px; border-radius: 8px; font-size: 15px; font-weight: 600; }
            #analyzeButton:hover { background-color: #357ae8; }
            #clearButton { background-color: #444; color: #eee; }
            #clearButton:hover { background-color: #666; }
            #randomSentenceButton { background-color: #2196F3; color: #fff; }
            #randomSentenceButton:hover { background-color: #1976D2; }
            #resultCard { background: #23272e; border-radius: 12px; border: 1.5px solid #444; margin-top: 10px; }
            #resultText { margin-top: 4px; font-size: 18px; color: #fff; }
            QLabel { color: #eee; }
            """
        else:
            return """
            QMainWindow { background-color: #e9ecef; }
            #centerCard { background: #fff; border-radius: 18px; min-width: 400px; max-width: 520px; }
            #mainTitle { color: #222; margin-bottom: 0px; }
            #textInput { border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px; background-color: #fafbfc; font-size: 15px; }
            #analyzeButton, #clearButton, #randomSentenceButton { background-color: #4F8EF7; color: white; border: none; padding: 12px 28px; border-radius: 8px; font-size: 15px; font-weight: 600; }
            #analyzeButton:hover { background-color: #357ae8; }
            #clearButton { background-color: #e0e0e0; color: #333; }
            #clearButton:hover { background-color: #bdbdbd; }
            #randomSentenceButton { background-color: #2196F3; color: #fff; }
            #randomSentenceButton:hover { background-color: #1976D2; }
            #resultCard { background: #f7fafd; border-radius: 12px; border: 1.5px solid #e3e8ee; margin-top: 10px; }
            #resultText { margin-top: 4px; font-size: 18px; }
            QLabel { color: #333; }
            """

    def show_help(self):
        QMessageBox.information(self, "Yardım", "\n- Metni girin ve 'Analiz Et' butonuna tıklayın.\n- Sonuç kutusunda analiz sonucu ve duygunun emojisi gösterilir.\n- 'Rastgele Cümle' butonuna tıklayarak otomatik olarak örnek bir İngilizce duygu cümlesi ekleyebilirsiniz.\n- Son analizler aşağıda listelenir.\n- Karanlık mod için sağ üstteki anahtarı kullanabilirsiniz.")

    def toggle_dark_mode(self):
        self.dark_mode = self.dark_mode_toggle.isChecked()
        self.setStyleSheet(self.get_stylesheet())

    def analyze_sentiment(self):
        text = self.text_input.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "Uyarı", "Lütfen analiz edilecek bir metin girin!")
            return
        # Yükleniyor animasyonu başlat
        self.set_loading(True)
        QTimer.singleShot(900, lambda: self._do_analysis(text))

    def _do_analysis(self, text):
        try:
            sentiment = sentiment_analysis.predict_sentiment(text)
            emoji = SENTIMENT_EMOJIS.get(sentiment, '')
            self.result_text.setText(f"{emoji} {sentiment}")
            self.result_text.setStyleSheet(f"color: {self.sentiment_colors[sentiment]};")
            # Geçmişe ekle
            self.history.insert(0, f"{emoji} {sentiment}: {text[:40]}{'...' if len(text)>40 else ''}")
            self.history = self.history[:5]
            self.update_history()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Analiz sırasında bir hata oluştu: {str(e)}")
        self.set_loading(False)

    def set_loading(self, loading):
        self.loading = loading
        if loading:
            self.loading_label.setText("⏳")
            self.loading_label.show()
            self.analyze_button.setEnabled(False)
        else:
            self.loading_label.hide()
            self.analyze_button.setEnabled(True)

    def insert_random_sentence(self):
        sentences = [
            "I am so happy today!",
            "This is the worst day ever.",
            "I feel nothing about this.",
            "What a wonderful surprise!",
            "I'm really disappointed.",
            "Life is beautiful.",
            "I don't care about this news.",
            "She is very excited for the trip.",
            "He was sad after the movie.",
            "Everything is just okay."
        ]
        self.text_input.setText(random.choice(sentences))

    def update_history(self):
        if self.history:
            self.history_label.show()
            self.history_list.show()
            self.history_list.setText("\n".join(self.history))
        else:
            self.history_label.hide()
            self.history_list.hide()

    def clear_text(self):
        self.text_input.clear()
        self.result_text.setText("")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SentimentAnalyzerApp()
    window.show()
    sys.exit(app.exec_()) 