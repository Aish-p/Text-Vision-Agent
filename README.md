# Text-Vision-Agent 🚀🖼️

The Text Vision Agent is an AI-powered assistant that can generate images from a text prompt and describe the generated images. It combines image generation and vision-based language models to create a seamless process of generating and interpreting images. The agent utilizes **LangChain, ChatOllama models,** and a custom pipeline for image processing.

## Features 🌟
  * Image Generation 🖌️: Generate stunning images from text descriptions.
  * Image Description 📝: Receive detailed descriptions of generated images.

## Tools Used 🔧
This agent leverages several state-of-the-art tools and models:

**Langchain** 🔗: A framework for developing AI applications. It helps with the orchestration of the tools and agents.
  * ChatOllama: A wrapper around the Ollama LLM for conversational AI tasks.
  * Initialize_agent: A function to configure and set up the agent to process tasks.

**FluxPipeline** 🔥: A model from Diffusers used for generating images from text prompts. It uses deep learning to produce high-quality images.

**Pillow (PIL)** 🖼️: A Python Imaging Library used to manipulate and process images, including converting images to Base64 encoding for easier sharing and storage.

**Base64** 🔐: For encoding and decoding images, allowing the agent to handle image data in a text-friendly format.

**Torch** 🔋: PyTorch framework used for deep learning, which powers the FluxPipeline.

**Langchain Core** 🧠: A set of essential utilities for managing input-output parsing, tools, and agent management.

## Installation 🛠️
1. **Clone the Repository**
2. **Install Requirements**
 * Install the necessary dependencies by running:
   ```
   pip install -r requirements.txt
   ```
3. **Set up Ollama**
  * Ensure that you have Ollama running. You can download and install Ollama from [here](https://ollama.com/download)
  * Pull the required models by running the following commands:
    ```
    ollama pull llama3.2:latest
    ollama pull llama3.2-vision:latest
    ```
4. **Verify Installation**
  * Once the dependencies are installed and the models are pulled, you should be good to go! To verify everything is set up correctly, run the following command:
    ```
    python text_vision_agent.py
    ```
    You should see the welcome message and be able to interact with the agent.
    
## How It Works 🔄

1. **Step 1: Generate Image** 🌄
  * The user provides a textual prompt (e.g., “A serene beach at sunset”).
  * The **FluxPipeline** generates the image from the prompt using deep learning models powered by **Torch**. These models convert the text into pixels, forming an image that corresponds to the user's description.

2. **Step 2: Describe Image** 🖋️
  * The generated image is passed through the **ChatOllama** Vision model. The model interprets the content of the image and generates a descriptive text. For example, it might generate a description like: "A beautiful beach with the sun setting behind the waves, and palm trees lining the shore."

3. **Step 3: Agent Interaction & Execution** 💬
  * The **initialize_agent** function ties all these tools together. It configures the **LangChain** agent to execute tasks step-by-step, starting with generating the image, then interpreting and describing it. The agent follows an approach known as **Zero-Shot Reacting**, where it can make decisions dynamically based on the available tools without predefined responses.

## Demo 🔍
Here are some screenshots from the Text-Vision Agent in action:

### User Input  
📝 **Prompt:**  
*"An ancient Greek soldier holding a sword and shield stands in the foreground. Behind him, horses are visible, and in the background, a snow-capped mountain rises."*  

### Screenshots

<div align="center">
  <p><strong>Generated Image</strong></p>
  <img src="/screenshots/An_ancient_G.png" alt="Generated Image" width="500">
</div>
<br>

<div align="center">
  <p><strong>Description of Image</strong></p>
  <img src="/screenshots/description.PNG" alt="Description of Image" width="700">
</div>
<br>

## Usage 🧑‍💻
Once the agent is set up and running, you can interact with it as follows:

1.**Launch the Agent**:
  * Start the agent by running the script
  ```
  python text_vision_agent.py
  ```

2. **Enter a Topic**:
  * Enter a topic (e.g., 'person standing on a mountain peak at sunrise') when prompted.
  * The agent will generate an image based on this topic and provide a description.

3. **Get the Result**:
  * You’ll receive the generated image and a description of it.
