# 🧠 AI Trader – LSTM Based Stock Prediction

Bu proje, **LSTM (Long Short-Term Memory)** modeli kullanarak hisse senedi fiyat tahmini yapar.  
Backend **FastAPI**, frontend ise **Streamlit** ile geliştirilmiştir.

---

## 📁 Proje Yapısı

```text
ai-trader-lstm/
├── app/                 # FastAPI backend
│   └── main.py
├── ml/                  # Model eğitimi ve veri işlemleri
│   ├── train.py
│   └── data.py
├── frontend/            # Streamlit frontend
│   └── streamlit_app.py
├── requirements.txt
├── README.md
└── .venv/
alıştırma Sırası (Tam Akış)

Aşağıdaki adımlar sırayla ve eksiksiz uygulanmalıdır.

1️⃣ Ortam ve Paket Kurulumu
Virtual environment oluştur ve aktif et
python3 -m venv .venv
source .venv/bin/activate

Gerekli paketleri yükle
pip install -r requirements.txt


📌 Not: Tüm komutlar (.venv) aktifken çalıştırılmalıdır.

2️⃣ Model Eğitimi

LSTM modelini eğitmek için:

python -m ml.train


Bu adımda:

Veri çekilir

Ön işleme yapılır

LSTM modeli eğitilir

Model dosyası diske kaydedilir

3️⃣ Backend (FastAPI) Başlatma

Yeni bir terminal aç (Terminal-1)
Aynı proje klasöründe ve venv aktifken:

source .venv/bin/activate
uvicorn app.main:app --reload --port 8000


Backend başarıyla ayağa kalktığında:

API: http://127.0.0.1:8000

Swagger Docs: http://127.0.0.1:8000/docs

4️⃣ Frontend (Streamlit) Başlatma

Yeni bir terminal aç (Terminal-2)
Aynı proje klasöründe:

source .venv/bin/activate
streamlit run frontend/streamlit_app.py


Tarayıcı otomatik açılmazsa:

👉 http://localhost:8501

🧩 Mimari Akış
Streamlit (Frontend)
        |
        | HTTP (REST)
        v
FastAPI (Backend)
        |
        v
LSTM Model (Eğitilmiş)

⚠️ Önemli Notlar

Backend ve frontend aynı anda ama farklı terminallerde çalıştırılmalıdır

source .venv/bin/activate her terminal için ayrı ayrı yapılmalıdır

Backend kapalıyken frontend çalışsa bile tahmin alınamaz

Kullanılan portlar:

FastAPI → 8000

Streamlit → 8501

🛠 Olası Sorunlar ve Çözümleri
Problem	Çözüm
streamlit: command not found	pip install streamlit
Veri çekilemiyor	yfinance ve curl_cffi sürümlerini güncelle
API bağlantı hatası	Backend çalışıyor mu (/docs) kontrol et
Model bulunamadı	Önce python -m ml.train çalıştır


👤 Author
Ulaş Gültekin
AI • Backend • DevOps • ML Engineering