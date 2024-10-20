# CSV-RAG 🧠📊

![License](https://img.shields.io/badge/license-MIT-blue.svg)

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)

![FastAPI](https://img.shields.io/badge/FastAPI-0.95.1-blue.svg)

![LangChain](https://img.shields.io/badge/LangChain-0.0.232-blue.svg)

![OpenAI GPT-4](https://img.shields.io/badge/OpenAI-GPT--4-blue.svg)

## 🚀 Introduction

**CSV-RAG** is a sleek Retrieval-Augmented Generation (RAG) system that transforms your CSV data into an interactive Q&A tool. Powered by [LangChain](https://langchain.com/), [FastAPI](https://fastapi.tiangolo.com/), and [OpenAI GPT-4](https://openai.com/product/gpt-4), it allows you to effortlessly query data about French schools in natural language.

## ✨ Features

- **Smart CSV Handling**: Automatically manages various encodings and delimiters.
- **AI-Powered Insights**: Generates embeddings with OpenAI for precise data retrieval.
- **Lightning-Fast Search**: Utilizes FAISS for quick and relevant responses.
- **User-Friendly API**: Easy-to-use endpoints built with FastAPI.
- **Detailed Logging**: Keeps track of all operations for seamless debugging.

## 🛠 Installation

### Prerequisites

- **Python 3.9+**
- **Git**
- **OpenAI API Key**

### Steps

1. **Clone the Repo**

   ```bash
   git clone https://github.com/Ezz-di/csv-augmented-qa.git
   cd csv-augmented-qa
   ```

2. **Set Up Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**

   Create a `.env` file and add your OpenAI API key:

   ```env
   OPENAI_API_KEY=your-openai-api-key
   ```

## 🎯 Usage

### Run the Application

Start the server with:

```bash
python app.py
```

Or using Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Access the API at [http://localhost:8000](http://localhost:8000).

### API Endpoints

#### 1. `/rag` - Q&A

- **Method**: `POST`
- **Description**: Ask questions about French schools.
- **Example**:

  **Request:**

  ```json
  {
    "question": "Quel est l'identifiant de l'ECOLE PRIMAIRE PUBLIQUE REOTIER - ST CLEMENT SUR DURANCE?",
    "chat_history": []
  }
  ```

  **Response:**

  ```json
  {
    "answer": "L'identifiant de l'ECOLE PRIMAIRE PUBLIQUE REOTIER - ST CLÉMENT SUR DURANCE est 0050650E."
  }
  ```

#### 2. `/rag-chain` - Advanced Q&A

- **Method**: `POST`
- **Description**: Engage in contextual conversations.
- **Example**:

  **Request:**

  ```json
  {
    "question": "Quelle école a le meilleur taux de réussite au baccalauréat?",
    "chat_history": ["Quelle école a le meilleur taux de réussite au baccalauréat?"]
  }
  ```

  **Response:**

  ```json
  {
    "answer": "L'École primaire publique la Diamanterie a le meilleur taux de réussite au baccalauréat avec 98%."
  }
  ```

### 🧪 Testing with `curl`

**Example:**

```bash
curl -X POST "http://localhost:8000/rag" \
     -H "Content-Type: application/json" \
     -d '{"question": "Quel est le téléphone de l\'Ecole maternelle Sospel?"}'
```

**Response:**

```json
{
  "answer": "Le téléphone de l'Ecole maternelle Sospel est 0493272587."
}
```

## 🙏 Acknowledgements

- [LangChain](https://langchain.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenAI](https://openai.com/)

---
## 📫 Contact

Questions? [Open an issue](https://github.com/Ezz-di/csv-augmented-qa/issues) or email [izedine14021@gmail.com](mailto:izedine14021@gmail.com).

© 2024 Eze Eddine