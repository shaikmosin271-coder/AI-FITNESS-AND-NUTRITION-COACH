# 🏋️‍♂️ AI Fitness & Nutrition Coach: RAG Q&A Assistant

An end-to-end Retrieval-Augmented Generation (RAG) LLM application powered by **Google Gemini**, **LangChain**, and **FAISS**. 

This project transforms a standard Q&A setup into a specialized AI Fitness and Nutrition Coach. Users can query workout splits, exercise techniques, target muscle groups, macro splits, and dietary plans through an intuitive **Streamlit** chat interface.

---

## 📌 Features & Highlights

- **Custom Fitness Knowledge Base**: Grounded on a structured dataset (`fitness_data.csv`) covering exercises, routine schedules, macro nutrition, and specific fitness goals.
- **Strict Anti-Hallucination Prompting**: Custom prompt engineering forces the model to rely *only* on context retrieved from the database. If a question is outside the scope, it responds safely with *"I don't know."*
- **Local Vector Database**: Fast similarity lookup using **FAISS** without relying on third-party cloud vector stores.
- **Modern LLM Integration**: Uses Google's low-latency Gemini models via `langchain-google-genai`.

---

## 🛠️ Core Technologies

| Component | Technology / Framework |
| :--- | :--- |
| **Frontend UI** | Streamlit |
| **LLM Orchestration** | LangChain & LangChain Community |
| **Large Language Model** | Google Gemini (`gemini-1.5-flash` / `gemini-2.5-flash`) |
| **Embeddings** | HuggingFace Instructor (`hkunlp/instructor-large`) |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |

---

## 📁 Project Structure

```text
3_project_codebasics_q_and_a/
├── .env                 # Secret keys (GOOGLE_API_KEY)
├── fitness_data.csv     # Knowledge base (type, name, details, information)
├── langchain_helper.py  # Indexing, embedding setup, and RAG retrieval chain
├── main.py              # Streamlit application UI
└── requirements.txt     # Python dependencies