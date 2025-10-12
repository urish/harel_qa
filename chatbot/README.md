# Setup

1. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install the required packages:

   ```bash
    pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your API keys:

   ```env
   GOOGLE_API_KEY=your_google_api_key
   ```

   Optionally, you can also enable LangSmith logging by adding:

   ```env
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=your_langsmith_api_key
   LANGSMITH_PROJECT=apex-harel
   ```

4. Run milvus db
   Note: https://milvus.io/docs/install_standalone-docker.md

   ```bash
   # to start a milvus db instance
   bash standalone_embed.sh start
   ```

4. Run the script to process the documents and set up the vector store:

   Indexing whole documents 
   ```bash
    python index_docs.py
   ```

   Indexing documents with page numbers (WIP!!)
   ```bash
   python index_docs_to_local_pages.py --input-dir ../data/data-original/ --output-dir ../data/data-processe-with-pages/d --no-ocr

   python push_docs_to_milvus.py --input-dir ../data/data-processe-with-pages --collection-name documents
   ```
   
5. Start the chatbot:

   ```bash
     python chatbot.py
   ```

You can also run the web server to access the chatbot via a web interface:

```bash
 python chatbot/chatbot.py --serve
```

Then open your browser and navigate to `http://localhost:8000` to interact with the chatbot.