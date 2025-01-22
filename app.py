import gradio as gr
from openai import OpenAI
import os

api_key = os.getenv("NVIDIA_API_KEY")
MODEL_ID = "meta/llama-3.1-405b-instruct"

# Check if the API key is found
if api_key is None:
    raise ValueError("NVIDIA_API_KEY environment variable not found.")
else:
    # Initialize the OpenAI client
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

def get_ai_response(prompt):
    """Generates a response from an AI model

    Args:
    prompt: The prompt to send to the AI model.

    Returns:
    response from the AI model.
    """
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a programming assistant focused on providing \
                accurate, clear, and concise answers to technical questions. \
                Your goal is to help users solve programming problems efficiently, \
                explain concepts clearly, and provide examples when appropriate. \
                Use a professional yet approachable tone. Use explicit markdown \
                format for code for all codes in the output."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        completion = client.chat.completions.create(
            model=MODEL_ID,
            temperature=0.5,  # Adjust temperature for creativity
            top_p=1,
            max_tokens=1024,
            messages=messages,
            stream=False
        )

        model_response = completion.choices[0].message.content
        return model_response

    except Exception as e:
        return f"Error handling AI response: {e}"

def main(platform, task):
    detailed_task = ""

    if task == "Get NVIDIA API Key":
        detailed_task = f"""
Search the web for information how to obtain an API key from Nvidia NGC and 
give detailed instruction on to how to setup a huggingface space to host 
a {platform} app that uses the Nvidia API. """
        
    elif task == "Code the Program on the select platform":
        detailed_task = f"""
Create a {platform} app that gives the user an intuitive interface using a text 
area to prompt the user for an input prompt.  Provide a button that sends the 
input to the AI model.  The app displays the response to the page. 
Give me the full python code for this app."""
        
    elif task == "Deploy and test the App":
        detailed_task = f"""
Give detailed instruction on how to deploy and test a {platform} app on Hugging Face.
        """

    response = get_ai_response(detailed_task)
    return detailed_task, response

# Gradio Interface
interface = gr.Interface(
    fn=main,
    inputs=[
        gr.Dropdown(["Streamlit", "Gradio"], label="Choose the platform:"),
        gr.Dropdown(["Get NVIDIA API Key", "Code the Program on the select platform", "Deploy and test the App"], label="Select a task:")
    ],
    outputs=[
        gr.Textbox(label="Generated AI Prompt:"),
        gr.Markdown(label="AI Response:")
    ],

    title="Create an AI App using the Nvidia AI Model"
)
layout="vertical",  # Ensures the layout is in a single column
interface.launch()
