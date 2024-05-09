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
  

### Language Model Setting
- Level of Initial Number of Meshes: This control sets the initial quantity of meshes generated. Note that applying further subdivisions will increase the total number of meshes.
- Value Noise Height Setting: This determines the foundational height of the terrain, independent of any modifications made by the displacement map.
- Maximum Displacement Height: Increasing this value can give the terrain a rockier appearance; however, it may also introduce more visual artifacts. Applying Loop Subdivision can help smooth out the terrain when artifacts become apparent.
- Loop Subdivision: Loop Subdivision: This feature smooths the shape of the terrain. Each application quadruples the number of meshes.
- Note: Applying subdivision twice can help enhancing the quality.

### Issues
- This project has been tested on Safari. When using Chrome, shadow mapping may exhibit issues due to limited precision support for WebGL depth maps.

### PDF Document 
1. Diffusion map: `/static/assets/skinning/diff.png`
2. Normal map: `/static/assets/skinning/normal.png`
3. Displacement map: `/static/assets/skinning/dis.png`
4. Feel free to experiment with other textures for the terrain. The default textures are sourced from [Poly Haven](https://polyhaven.com/).