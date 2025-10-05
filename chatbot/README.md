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
   ```

4. Run the script to process the documents and set up the vector store:

   ```bash
    python index_docs.py
   ```

5. Start the chatbot:

   ```bash
     python chatbot.py
   ```
