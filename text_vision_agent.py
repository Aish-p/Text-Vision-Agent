from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_core.output_parsers import StrOutputParser
from diffusers import FluxPipeline
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage
import torch
from io import BytesIO
from PIL import Image
import base64
from colorama import Fore, init

# Initialize colorama
init(autoreset=True)

def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG") 
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def prompt_func(data):
    """
    Prepare the message for the LLM by combining image and text
    """
    image = data['image']
    text = data['text']

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]


@tool 
def make_image(topic: str):
    """
    Generate an image from the text prompt.
    The image name will depend on the topic provided by the user.
    """
    prompt = f"Create an image of a {topic}."
    
    # Generate image using FluxPipeline
    pipe = FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-schnell', torch_dtype=torch.bfloat16).to('cuda')
    image = pipe(prompt, height=1024, width=512).images[0]

    image_name = f"./{topic.replace(' ', '_')}.jpeg"
    image.save(image_name)
    return image_name


@tool 
def describe_image(data):
    "Describe the generated image."
    image = Image.open(data)
    data = convert_to_base64(image)
    llm_vision = ChatOllama(model='llama3.2-vision:latest', temperature=0)
    chain = prompt_func | llm_vision | StrOutputParser()
    response = chain.invoke({'text': 'Describe the image.', 'image': data})
    
    return response


def main():
    memory = ConversationBufferWindowMemory(return_messages=True, memory_key='chat_history', input_key='input', k=5)
    
    # Define the agent prompt template
    prompt_temp = """ 
    You are an intelligent assistant with access to the following tools: {tools}.
    You must answer the question. 
    you follow the steps.
    Question: [The question you must answer]
    Thought: [Your thoughts about what to do next]
    Action: [The action to take, one of: {tool_names}]
    Action Input: [The input to the action]

    Begin!
    Question: {input}
    Thought: {agent_scratchpad}
    """
    prompt = PromptTemplate(template=prompt_temp, input_variables=['input', 'tools', 'agent_scratchpad'])
    
    # Initialize the LLM model
    llm = ChatOllama(model='llama3.2:latest', temperature=0)
    
    # List of tools to be used in the agent
    tools = [make_image, describe_image]
    
    # Initialize the agent
    agent = initialize_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        memory=memory,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=2,
        verbose=True
    )
    
    print(Fore.LIGHTYELLOW_EX + "Welcome to the Text-Vision-Agent! Start by entering your query.")

    # Get the user's topic
    query = input("\nEnter your topic (or 'exit' to quit): ")
        
    if query.lower() == 'exit':
        print("Goodbye!")
        return
        
    # Invoke the agent to process the query
    response = agent.invoke({'input': query})


if __name__ == '__main__':
    main()
