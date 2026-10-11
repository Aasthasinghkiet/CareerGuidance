# 🌷 Arogyini — Career & Livelihood Guidance Chatbot

Arogyini is an AI-powered career and livelihood guidance chatbot designed to help women explore career paths, develop skills, discover self-employment ideas, and make informed decisions based on their education, experience, interests, and circumstances.

Built with Python, Streamlit, Retrieval-Augmented Generation (RAG), and Google's Gemini model, Arogyini combines conversational guidance with structured job-matching functionality.

## ✨ Features

- **Personalized career guidance** based on education, experience, skills, and interests.
- **Inclusive recommendations** for people with no formal education, school qualifications, vocational training, degrees, or postgraduate qualifications.
- **Job matching** using structured job data, with company, location, skills, experience, and salary information when available.
- **Non-job livelihood ideas**, including small businesses, independent services, and home-based activities.
- **Career-break support** and guidance for returning to work.
- **Government and international career pathways.**
- **Training and skill-development guidance.**
- **RAG-based responses** grounded in a curated career-knowledge collection.

## 🛠️ Technology Stack

- Python 3.10
- Streamlit — user interface
- LangChain — language-model and retrieval integration
- ChromaDB — vector database
- Sentence Transformers — text embeddings
- Google Gemini — language model
- Pandas — structured job-data processing

## 📁 Project Structure

```text
CareerGuidance/
├── app/
│   ├── rag_chat.py
│   └── job_matcher.py
├── data/
│   ├── raw/
│   └── chroma_db/
├── scripts/
│   └── ingest_data.py
├── streamlit_app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/arogyini-career-guidance.git
cd arogyini-career-guidance
```

Replace `YOUR-GITHUB-USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment

Windows CMD:

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

Obtain an API key through [Google AI Studio](https://aistudio.google.com/).

**Security:** Never commit your `.env` file or publish your API key.

### 5. Prepare the knowledge base

Ensure the required career datasets are available in `data/raw/`, then run the project's ingestion script:

```bash
python scripts/ingest_data.py
```

This prepares the ChromaDB knowledge base used by the RAG pipeline.

### 6. Start the chatbot

```bash
python -m streamlit run streamlit_app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## 🚀 Deployment

Arogyini is designed to run as a Streamlit application and can be deployed using [Streamlit Community Cloud](https://share.streamlit.io/).

Deployment requires the project's dependencies, the Gemini API key configured as a secret, and a reliable way to make the ChromaDB knowledge base available to the deployed application.

## 🔐 Data and Reliability

- Recommendations depend on the quality and coverage of the available datasets.
- A suggested role does not necessarily mean that an employer is currently hiring.
- Eligibility, compensation, application details, and official recruitment requirements should be checked with the relevant organization.
- Never expose API keys or other secrets in source control.

## 🎯 Project Goal

Arogyini aims to make career and livelihood guidance more accessible, practical, and inclusive, helping women explore options that fit their individual goals and circumstances.

## 📌 Project Status

Under development. Features, data sources, and deployment configuration may change as the project evolves.

## 📄 License

No license has been selected yet. All rights remain with the respective copyright holders unless a license is added.
