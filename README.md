# 🫁 PneumoVisionAI: Explainable Medical Imaging Pipeline

[![.NET](https://img.shields.io/badge/.NET-10.0-512BD4.svg)](https://dotnet.microsoft.com/)
[![ASP.NET Core](https://img.shields.io/badge/ASP.NET_Core-MVC-blue.svg)](https://dotnet.microsoft.com/apps/aspnet)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Microservice-009688.svg)](https://fastapi.tiangolo.com/)

An AI-powered medical imaging platform designed to analyze Chest X-Rays for pneumonia detection. Built on a strict microservice architecture, it seamlessly integrates a secure **ASP.NET Core MVC** web application with a **Python/FastAPI** deep learning backend, utilizing **PyTorch** and **Grad-CAM** for Explainable AI (XAI).

> ⚠️ **Medical Disclaimer:** Developed for educational and research purposes. Not intended to replace professional medical diagnosis or clinical judgment.

---

## 🚀 Key Features

- **Explainable AI (Grad-CAM):** Moves beyond "black-box" predictions. The system generates visual heatmaps highlighting the exact pulmonary regions that contributed to the deep learning model's decision.
- **Microservice Architecture:** Complete decoupling of the web application (C#) from the AI inference engine (Python). This ensures independent scalability, easier model swapping, and robust environment isolation.
- **Adaptive Confidence Thresholds:** Evaluates raw PyTorch probabilities to classify results as *PNEUMONIA*, *NORMAL*, or *INCONCLUSIVE*, preventing the system from presenting low-confidence predictions as definitive results.
- **Advanced Preprocessing:** Utilizes CLAHE (Contrast Limited Adaptive Histogram Equalization) via OpenCV to normalize and enhance Chest X-Ray contrast prior to inference.
- **Secure Web Platform:** Full-stack implementation featuring ASP.NET Core Identity for authentication and Entity Framework Core (SQL Server) with resilient connection handling (`EnableRetryOnFailure`) for robust data management.

---

## 📊 Explainability & Visual Results

Instead of simply returning a binary classification, the AI microservice processes the X-Ray and returns a Grad-CAM heatmap overlay, allowing users to inspect the visual decision-making process.

- **Left:** Original Chest X-Ray Input | **Right:** Grad-CAM Decision Heatmap
![Grad-CAM Visualization](heatmap_result.jpg)

---

## 🛠️ Tech Stack & Architecture

**Web Application (Frontend & Backend):**
- C#, .NET 10, ASP.NET Core MVC
- Entity Framework Core & SQL Server
- ASP.NET Core Identity (Authentication)

**AI Microservice (Inference Engine):**
- Python 3.11+, FastAPI (RESTful API via `multipart/form-data`)
- PyTorch (Model Inference)
- OpenCV & Grad-CAM (Preprocessing & XAI)

---

## ⚙️ Getting Started

### Prerequisites
- .NET 10 SDK & SQL Server
- Python 3.11+

### Build & Run

**1. Start the AI Microservice:**
```bash
cd AI_Service
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
