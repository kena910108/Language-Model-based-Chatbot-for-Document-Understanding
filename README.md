# Language Model-based Chatbot for Document Understanding

## Project Overview
- This project utilizes Django as the web framework. It incorporates Microsoft's Phi-1.5, a small language model, to run efficiently on laptops. The project employs the BM25 algorithm for document retrieval, which enables the chatbot to utilize documents as context for generating responses.

#### Installation
- set up a virtual environment:
  ```bash
  python -m venv LMChatbot
- activate the environment:
  ```bash
  source LMChatbot/bin/activate
- Install libraries:
  ```bash
  pip install -r requirements.txt

#### Building and Running the Project

- Running the redis server
  ```bash
  redis-server
- Start tailwind
  ```bash
  python manage.py tailwind start
- Run the Django
  ```bash
  python manage.py runserver
- Launching the app by pointing your browser to http://127.0.0.1:8080


### PDF Document 
- The PDF input is set to a specific file named 'example.pdf' at the root directory.

### Issues
- The response time can be as long as several minutes due to the computationally intensive nature of the inference process.
- If the text block size in the PDF is large, it may exceed the maximum token length.