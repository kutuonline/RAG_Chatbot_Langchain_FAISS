**RAG Chatbot Using Langchain, Grog, and FAISS vector database**

The chatbot can provide information based on uploaded PDF files stored in a vector database. You can ask any question as long as the information is in the file, and the machine will answer your question accordingly. There is a chat history feature on the chatbot.


**How to use:**
- Install all libraries in requirements.txt
- Create a folder with the name content to store source PDF files
- Copy rag_chatbot.py
- Run in your terminal or streamlit

If you see "**Please upload a PDF to create or load the FAISS index.**" on screen, that means there is no PDF file uploaded or saved yet. You can upload any PDF file. But if you see "**Loaded existing FAISS index.**" on screen, you can ask any question directly on the chat screen.


**Note:**
- Currently, the vector database can only store one file, additional files cannot be added.
- If you want to add additional information, you can add it to the previous PDF file and then re-upload it. Please delete the faiss_index folder first before re-upload.

File **.env** --> place your API key here to develop chatbot in local machine.
If you develop and run in local machine, on the statement os.environ["GROQ_API_KEY"] = "**GROQ_API_KEY**" and client = Groq(api_key=**GROQ_API_KEY**), replace GROQ_API_KEY with your key. If you are using an API key other than groq, just adjust it accordingly.

This chatbot uses the llama model and FAISS as a vector database. Perform similarity search when user asks a question.


**Refference:** 
- https://manojkumar19.medium.com/creating-an-ai-powered-chatbot-with-retrieval-augmented-generation-rag-using-faiss-langchain-a8ed5587f08c 
