from flask import Flask, render_template, request
from langchain_google_genai import ChatGoogleGenerativeAI
# We are importing HuggingFaceEmbeddings to run embeddings locally
# This avoids the Google AI embedding quota error on startup.
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
# LLMChain is deprecated, we will use LCEL (prompt | llm) instead
# from langchain.chains import LLMChain 
from langchain_core.output_parsers import StrOutputParser # Import the output parser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
import os
from dotenv import load_dotenv

# Load environment variables (like your API key) from a .env file
load_dotenv()

app = Flask(__name__)

# Set your Google API key from the environment
# Make sure this is in your .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in .env file.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize the Chat LLM (this still uses the Google API and has its own quota)
# --- CHANGED: Use 'gemini-pro' as a stable alternative to 'gemini-1.5-flash' ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)

# --- CHANGED SECTION ---
# Use a local embedding model from HuggingFace
# This runs on your machine and avoids the API quota error.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# --- END OF CHANGED SECTION ---


def load_email_templates():
    """Loads email templates from a local file."""
    try:
        with open("email_templates.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        print("Error: email_templates.txt not found. Please create this file.")
        # Return a default as a fallback, though the file is better
        return """
        Formal Business Inquiry:
        Subject: Business Inquiry from [Your Name]
        Dear [Recipient Name],
        I hope this message finds you well. I am writing to inquire about [specific topic]. 
        Could you please provide more information regarding [specific question]?
        Thank you for your time and consideration.
        Best regards,
        [Your Name]
        
        Job Application:
        Subject: Application for [Position Name] - [Your Name]
        Dear Hiring Manager,
        I am writing to express my interest in the [Position Name] position at [Company Name]. 
        With my experience in [relevant experience], I believe I would be a strong candidate for this role.
        Please find my resume attached for your review.
        Sincerely,
        [Your Name]
        """


def create_vector_store():
    """Creates the FAISS vector store from the email templates."""
    template_text = load_email_templates()
    if not template_text:
        return None

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(template_text)
    
    if not chunks:
        print("Warning: No text chunks were created from templates.")
        return None

    # This now uses the local HuggingFace model for embeddings
    return FAISS.from_texts(chunks, embedding=embeddings)


# Create the vector store on startup
vector_store = create_vector_store()
if vector_store:
    retriever = vector_store.as_retriever(search_kwargs={"k": 2}) # Reduced k to 2
else:
    print("Error: Vector store could not be initialized.")
    # Create a dummy retriever to avoid crashing the app
    class DummyRetriever:
        def get_relevant_documents(self, query):
            return []
    retriever = DummyRetriever()


# Define the prompt template for the LLM
email_prompt = PromptTemplate(
    input_variables=["context", "prompt", "recipient", "tone", "length"],
    template="""
    You are a professional email assistant. Use the following email templates as reference:
    {context}

    Generate an email based on the following request:
    {prompt}
    
    Recipient: {recipient}
    
    Tone of the email: {tone}

    Length of the email: {length}

    Structure your response as follows:
    
    Subject: [Email Subject Here]
    
    Body:
    [Email Body Here]
    
    Important:
    - Include appropriate greeting and closing
    - Use proper tone mentioned in the tone of the email and proper formatting
    - Personalize for the recipient
    - Keep in mind the length of the email
    - DO NOT include anything before "Subject:"
    """
)

# Initialize the LLM chain
# --- CHANGED: Replaced deprecated LLMChain with LCEL syntax ---
email_chain = email_prompt | llm | StrOutputParser()

@app.route('/', methods=['GET', 'POST'])
def index():
    email_subject = None
    email_body = None
    error_message = None

    if request.method == 'POST':
        try:
            prompt_text = request.form['prompt']
            recipient = request.form['recipient']
            tone = request.form['tone']
            length = request.form['length']
            
            # Get relevant templates from the local vector store
            # --- CHANGED: Use .invoke() instead of .get_relevant_documents() ---
            relevant_templates = retriever.invoke(prompt_text)
            context = "\n\n".join([doc.page_content for doc in relevant_templates])
            
            # Run the chain to generate the email
            # This is the part that uses the Google LLM API
            # --- CHANGED: Use .invoke() with a dictionary instead of .run() ---
            input_data = {
                "context": context,
                "prompt": prompt_text,
                "recipient": recipient,
                "tone": tone,
                "length": length
            }
            result = email_chain.invoke(input_data)
            
            # Parse the result from the LLM
            if "Subject:" in result and "Body:" in result:
                subject_split = result.split("Subject:", 1)
                body_split = subject_split[1].split("Body:", 1)
                
                email_subject = body_split[0].strip()
                email_body = body_split[1].strip()
            else:
                # Fallback if the model didn't follow the format
                print(f"Warning: Model output format unexpected: {result}")
                email_subject = "Generated Email (Format Error)"
                email_body = result.strip()
                
        except Exception as e:
            # This will catch the 429 quota error from the LLM
            print(f"Error generating email: {e}")
            error_message = f"Error generating email: {e}. You might be sending requests too quickly. Please wait a moment and try again."

    return render_template('index.html', 
                           email_subject=email_subject, 
                           email_body=email_body, 
                           error_message=error_message)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) # Using 5000 as a common dev port
    app.run(host="0.0.0.0", port=port, debug=True) # Added debug=True

