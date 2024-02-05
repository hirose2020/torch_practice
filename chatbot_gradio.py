import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def user(message, history):
    return "", history + [[message, None]]

def bot(history):
    user_message = history[-1][0]
    new_user_input_ids = tokenizer.encode(
        user_message + tokenizer.eos_token, return_tensors="pt"
        )

    bot_input_ids = torch.cat([torch.LongTensor([]), new_user_input_ids], dim=-1)
    response = model.generate(
        bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id).tolist()

    response = tokenizer.decode(response[0]).split("<|endoftext|>")
    response = [
        (response[i], response[i+1]) for i in range(0, len(response) - 1, 2)
        ]
    history[-1] = response[0]
    return history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
        )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(
    server_name="0.0.0.0"
    )

        
              
    
        
