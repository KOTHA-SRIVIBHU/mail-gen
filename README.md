

---

# AI Email Generator using Generative AI (Gemini + LangChain + Flask)

This is a Flask-based web application that generates professional and personalized emails using Google's Gemini AI (`gemini-2.0-flash-001`) through LangChain. It uses email templates which are stored locally and retrieves context using vector search (FAISS) to generate accurate and well-structured emails.

#  [Live Link](https://mail-gen-ai.onrender.com/) 
sometimes it will take 50 seconds to load the appication

## Features

* Uses Google Gemini (`gemini-2.0-flash-001`) via `langchain-google-genai`
* Context-aware generation using FAISS vector store
* Editable and copyable email subject & body
* Simple UI with BootStrap Framework

---

## Tech Stack

* **Frontend**: HTML, CSS (Bootstrap)
* **Backend**: Python, Flask
* **AI/LLM**: Google Gemini via LangChain
* **Embeddings**: `models/embedding-001` via `GoogleGenerativeAIEmbeddings`
* **Vector Store**: FAISS
* **Environment Management**: `python-dotenv`

---

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-mail-generator.git
cd ai-mail-generator
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up `.env` file

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key
PORT=10000
```

> **Note**: Ensure your Google API Key has access to Gemini and Embedding models.

---

## 📂 File Structure

```
├── app.py                    # Main Flask application
├── templates/
│   └── index.html            # UI template
├── email_templates.txt       # Predefined email templates (optional)
├── .env                      # Environment variables
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Running the App

```bash
python app.py
```

The app will be live at:
🔗 `http://localhost:10000/`

---

## Example Usage

1. Provide:

   * Your **prompt**
   * Desired **tone** (e.g., Formal, Informal,Harsh)
   * Expected **length** (Short, Medium, Long) or (100,200,300) in words 
   * **Recipient** name

2. Submit the form to generate the email.

3. You can:

   * Edit the subject/body directly
   * Copy them with the click of a button

![Alt Text](screenshots\i.png)

![Alt Text](screenshots\ii.png)

---

## Behind the Scenes

* `email_templates.txt` is split into chunks using `CharacterTextSplitter`.
* FAISS indexes these chunks for semantic retrieval.
* On form submission:

  * Relevant templates are retrieved using the prompt.
  * Gemini LLM generates a personalized email using a `PromptTemplate`.
* Output is rendered with a clean Bootstrap interface.

---

## Dependencies

```
Flask
python-dotenv
langchain
langchain-google-genai
langchain-community
faiss-cpu
```

> Install using: `pip install -r requirements.txt`


## License

This project is for educational/demo purposes.


* [LangChain](https://www.langchain.com/)
* [Google Generative AI (Gemini)](https://ai.google.dev/)
* [Bootstrap](https://getbootstrap.com/)
* [FAISS](https://github.com/facebookresearch/faiss)

---