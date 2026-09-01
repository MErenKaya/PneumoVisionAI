# 🫁 PneumoVisionAI

<p align="center">
  <strong>AI-Powered Pneumonia Detection & Explainable Chest X-Ray Analysis</strong>
</p>

<p align="center">
  <em>Detect. Explain. Visualize.</em>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a>
</p>

---

## 🧠 About the Project

**PneumoVisionAI** is an AI-powered medical imaging platform designed to analyze **Chest X-Ray** images and assist in the detection of **Pneumonia** using deep learning.

The system does not only provide a prediction. It also uses **Grad-CAM** to generate visual heatmaps showing which regions of the X-Ray contributed to the model's decision.

This allows the project to address not only:

> **"What did the model predict?"**

but also:

> **"Why did the model make this prediction?"**

> ⚠️ **Medical Disclaimer:** PneumoVisionAI is developed for educational and research purposes. It is not intended to replace professional medical diagnosis, clinical judgment, or treatment.

---

## ✨ Features

### 🩻 Chest X-Ray Analysis

Analyzes Chest X-Ray images using a deep learning model to estimate the likelihood of pneumonia.

### 🔥 Explainable AI — Grad-CAM

Uses **Grad-CAM** to visualize the areas of the X-Ray that contributed most to the model's prediction.

### 🎯 Confidence Threshold

Predictions are evaluated based on configurable confidence thresholds.

Possible outcomes include:

* 🟢 **PNEUMONIA**
* 🔵 **NORMAL**
* 🟡 **INCONCLUSIVE**

This helps prevent low-confidence predictions from being presented as definitive results.

### ⚡ Microservice Architecture

The AI workload is separated from the web application and handled by an independent **FastAPI microservice**.

Benefits include:

* Independent AI service development
* Easier model replacement
* Separation of Python and .NET environments
* Better scalability
* Clear separation of responsibilities

### 🔐 Authentication & User Management

User authentication and authorization are handled using **ASP.NET Core Identity**.

### 🗄️ Database Integration

**Entity Framework Core** and **SQL Server** are used to manage application data and analysis records.

### 🔄 Resilient Database Connections

EF Core's `EnableRetryOnFailure` is used to automatically retry transient database connection failures.

---

# 🏗️ Architecture

PneumoVisionAI consists of two primary layers:

```text
                         ┌──────────────────────┐
                         │       Browser        │
                         │    Web Interface     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     ASP.NET Core     │
                         │         MVC          │
                         │                      │
                         │ • Authentication     │
                         │ • MVC Controllers    │
                         │ • Razor Views        │
                         │ • Database           │
                         └──────────┬───────────┘
                                    │
                         REST / HTTP│
                      multipart/form-data
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │      AI Service      │
                         │                      │
                         │ • Image Processing   │
                         │ • PyTorch Inference  │
                         │ • Prediction         │
                         │ • Grad-CAM           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Deep Learning      │
                         │       Model          │
                         └──────────────────────┘
```

---

# 📁 Project Structure

```text
PneumoVisionAI/
│
├── WEB/
│   ├── Controllers/
│   │   └── MVC Controllers
│   │
│   ├── Views/
│   │   └── Razor Views
│   │
│   ├── Data/
│   │   └── Entity Framework Core
│   │
│   ├── Models/
│   │   └── Application Models
│   │
│   ├── Areas/
│   │   └── Identity
│   │
│   └── Program.cs
│
├── AI_Service/
│   ├── main.py
│   ├── requirements.txt
│   │
│   └── models/
│       └── Deep Learning Models
│
└── README.md
```

---

# 🛠️ Tech Stack

## 🌐 Web Application

| Technology                | Purpose                        |
| ------------------------- | ------------------------------ |
| **.NET 10**               | Application framework          |
| **ASP.NET Core MVC**      | Web application architecture   |
| **Entity Framework Core** | ORM / Database access          |
| **SQL Server**            | Relational database            |
| **ASP.NET Core Identity** | Authentication & authorization |
| **HttpClient**            | AI service communication       |

## 🤖 AI Service

| Technology       | Purpose                 |
| ---------------- | ----------------------- |
| **Python 3.11+** | AI service runtime      |
| **FastAPI**      | REST API                |
| **PyTorch**      | Deep learning inference |
| **OpenCV**       | Image processing        |
| **CLAHE**        | Image preprocessing     |
| **Grad-CAM**     | Explainable AI          |

---

# 🔬 AI Pipeline

When a Chest X-Ray is uploaded, it goes through the following pipeline:

```text
Chest X-Ray
     │
     ▼
Image Upload
     │
     ▼
Preprocessing
     │
     ├── Resize
     ├── Normalization
     └── CLAHE
     │
     ▼
PyTorch Model
     │
     ▼
Prediction
     │
     ├── Normal
     └── Pneumonia
     │
     ▼
Confidence Evaluation
     │
     ▼
Grad-CAM
     │
     ▼
Heatmap Generation
     │
     ▼
Web Interface
```

---

# 🔥 Explainable AI

One of the core features of PneumoVisionAI is its use of **Grad-CAM** for model interpretability.

Instead of displaying only the final prediction, the system generates a visual heatmap highlighting the areas that had the greatest influence on the model's decision.

```text
Original X-Ray
       │
       ▼
 ┌─────────────┐
 │             │
 │   🫁  🫁     │
 │             │
 └─────────────┘
       │
       ▼
    Grad-CAM
       │
       ▼
 ┌─────────────┐
 │     🔥      │
 │   🔥 🫁 🔥   │
 │      🔥     │
 └─────────────┘
```

This provides a practical **Explainable AI (XAI)** layer for inspecting the model's visual decision-making process.

---

# 🎯 Prediction Logic

The raw model probability is not directly treated as a definitive diagnosis.

Instead, predictions are evaluated using configurable confidence thresholds:

```text
              Model Prediction
                     │
                     ▼
             ┌───────────────┐
             │  Confidence   │
             │   Evaluation  │
             └───────┬───────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      HIGH        MODERATE       LOW
        │            │            │
        ▼            ▼            ▼
   PNEUMONIA    INCONCLUSIVE     NORMAL
```

The threshold-based approach helps identify uncertain predictions instead of forcing every input into a binary classification.

---

# 🚀 Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/MErenKaya/PneumoVisionAI.git

cd PneumoVisionAI
```

---

## 2️⃣ Start the AI Service

Navigate to the AI service directory:

```bash
cd AI_Service
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
python -m uvicorn main:app --reload --port 8000
```

The FastAPI Swagger documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

---

## 3️⃣ Start the Web Application

Open a new terminal and navigate to the web application:

```bash
cd WEB
```

Apply the Entity Framework Core migrations:

```bash
dotnet ef database update
```

Start the ASP.NET Core application:

```bash
dotnet run
```

The application will be available on the HTTPS port shown in the terminal.

For example:

```text
https://localhost:7123
```

---

# 🔌 API

The AI service exposes a REST API for image analysis.

### `POST /predict`

Analyzes a Chest X-Ray image.

**Request**

```http
POST /predict
Content-Type: multipart/form-data
```

**Example Response**

```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.94,
  "heatmap": "..."
}
```

> The exact response structure may vary depending on the current implementation of `main.py`.

---

# 🧩 Why a Microservice Architecture?

The web application and AI service are intentionally separated.

```text
┌─────────────────────┐
│     ASP.NET MVC     │
│                     │
│ • Users             │
│ • Authentication    │
│ • Web UI            │
│ • Database          │
└──────────┬──────────┘
           │
       REST API
           │
           ▼
┌─────────────────────┐
│       FastAPI       │
│                     │
│ • AI Inference      │
│ • Image Processing  │
│ • Grad-CAM          │
│ • Model Management  │
└─────────────────────┘
```

This architecture provides:

* **Separation of Concerns**
* Independent deployment
* Independent technology stacks
* Easier model updates
* Better scalability
* Cleaner project organization

---

# 🖥️ Application Flow

```text
1. User Login
      │
      ▼
2. Upload Chest X-Ray
      │
      ▼
3. ASP.NET MVC
      │
      ▼
4. FastAPI /predict
      │
      ▼
5. Image Preprocessing
      │
      ▼
6. PyTorch Inference
      │
      ▼
7. Confidence Evaluation
      │
      ▼
8. Grad-CAM Generation
      │
      ▼
9. Prediction + Heatmap
      │
      ▼
10. Web Interface
```

---

# 📊 Project Status

| Component                 | Status |
| ------------------------- | :----: |
| ASP.NET Core MVC          |    ✅   |
| Authentication            |    ✅   |
| SQL Server                |    ✅   |
| Entity Framework Core     |    ✅   |
| FastAPI                   |    ✅   |
| PyTorch Inference         |    ✅   |
| Image Preprocessing       |    ✅   |
| CLAHE                     |    ✅   |
| Grad-CAM                  |    ✅   |
| Confidence Threshold      |    ✅   |
| Microservice Architecture |    ✅   |

---

# 🗺️ Roadmap

Potential future improvements:

* [ ] Docker / Docker Compose support
* [ ] Model performance dashboard
* [ ] ROC-AUC visualization
* [ ] Precision / Recall / F1 metrics
* [ ] Multiple pneumonia classes
* [ ] Model versioning
* [ ] Automated model monitoring
* [ ] Cloud deployment
* [ ] Improved XAI visualizations
* [ ] Automated testing
* [ ] CI/CD pipeline

---

# 👨‍💻 Developer

### Mahmut Eren Kaya

<a href="https://github.com/MErenKaya">
  <img src="https://img.shields.io/badge/GitHub-MErenKaya-181717?style=for-the-badge&logo=github" />
</a>

---

# 📄 License

This project was developed for **educational and research purposes**.

See the `LICENSE` file for more information.

---

<p align="center">
  <strong>Made with ❤️ using .NET & Python</strong>
</p>
