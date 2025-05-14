import chromadb
import pandas as pd
import gradio as gr
from langchain_openai import OpenAIEmbeddings  # Updated import

# Load CSV with correct encoding
csv_file = "Software Questions.csv"
df = pd.read_csv(csv_file, encoding="ISO-8859-1")  # Change encoding if needed

# Ensure required columns exist
required_columns = {"Question Number", "Question", "Answer", "Category", "Difficulty"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV file must contain columns: {required_columns}")

# Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="faang_interview_questions")

# Initialize OpenAI Embeddings (Use your API key)
openai_api_key = " # Replace with actual key"
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Store Questions in ChromaDB
for _, row in df.iterrows():
    q_id = str(row["Question Number"])
    question_text = row["Question"]
    question_embedding = embeddings.embed_query(question_text)  # Generate embedding

    metadata = {
        "Answer": row["Answer"],
        "Category": row["Category"],
        "Difficulty": row["Difficulty"]
    }

    collection.add(ids=[q_id], documents=[question_text], embeddings=[question_embedding], metadatas=[metadata])

print("✅ Data stored successfully in ChromaDB!")

def retrieve_answer(user_query):
    query_embedding = embeddings.embed_query(user_query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1  # Get the most relevant question
    )

    if results["documents"]:
        best_match = results["documents"][0][0]
        answer = results["metadatas"][0][0]["Answer"]
        category = results["metadatas"][0][0]["Category"]
        difficulty = results["metadatas"][0][0]["Difficulty"]

        return f"""
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff;">
    <h3>🔹 Most Relevant Question:</h3>
    <p style="font-size: 16px;"><b>{best_match}</b></p>

    <h4>🔹 Category: <span style="color: #007bff;">{category}</span></h4>
    <h4>🔹 Difficulty: <span style="color: #ff5733;">{difficulty}</span></h4>

    <h3>🔹 Answer:</h3>
    <p style="font-size: 16px; line-height: 1.5;">💡 {answer}</p>
</div>
"""
    else:
        return "<b style='color:red;'>❌ No relevant answer found in the database. Try rephrasing your question!</b>"

# Gradio Interface with enhanced styling
iface = gr.Interface(
    fn=retrieve_answer,
    inputs=gr.Textbox(placeholder="Ask your interview question here...", label="💬 Interview Question"),
    outputs=gr.HTML(label="📢 Response"),  # Changed to HTML for better styling
    title="💡 Interview Question Bot",
    description="🔍 Ask any interview-related question and get structured answers instantly!",
    theme="default"
)

# Launch the Gradio App
iface.launch(share=True)



