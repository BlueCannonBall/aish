#!/usr/bin/env python3

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import readline, colorama, os, requests, json
from colorama import Back

RESET   = "\033[0m"
BLACK   = "\033[30m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"
BOLDBLACK   = "\033[1m\033[30m"
BOLDRED     = "\033[1m\033[31m"
BOLDGREEN   = "\033[1m\033[32m"
BOLDYELLOW  = "\033[1m\033[33m"
BOLDBLUE    = "\033[1m\033[34m"
BOLDMAGENTA = "\033[1m\033[35m"
BOLDCYAN    = "\033[1m\033[36m"
BOLDWHITE   = "\033[1m\033[37m"

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

mode = "dialogpt"
modes_friendly = {
    "dialogpt": "DialoGPT",
    "gpt2": "GPT-2",
    "bart": "BART"
}

colorama.init()

print("All commands are prefixed with '!'. For help, run '!help'.\n")

while True:
    prompt = input(f"{MAGENTA}{Back.MAGENTA}{BOLDWHITE} {modes_friendly[mode]} ≻{RESET}{MAGENTA}{RESET} ")

    if prompt.startswith("!"):
        args = prompt[1:].strip().split(' ')
        command = args[0].lower()
        del args[0]

        if command == "set":
            if args[0] in modes_friendly.keys():
                mode = args[0]
            else:
                print("That mode is invalid.")

        elif command == "ls":
            print("dialogpt     Conversational dialouge")
            print("gpt2         Text generation")
            print("bart         Summarization")

        elif command == "help":
            print("Commands:")
            print("    !help    Show this help screen")
            print("    !ls      List modes")
            print("    !set     Set mode")
            print("    !clear   Clear the console")
            print("    !quit    Exit")
            print("    !exit    Same as !quit")

        elif command == "clear":
            os.system("clear")
            continue

        elif command in ["quit", "exit"]:
            quit()

    else:
        if mode == "gpt2":
            inputs = gpt2_tokenizer.encode(prompt, return_tensors='pt')
            outputs = gpt2_model.generate(inputs, min_length=100, max_length=300, temperature=0.1, do_sample=True, pad_token_id=gpt2_tokenizer.eos_token_id, repetition_penalty=5.1)
            text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(text)

        elif mode == "dialogpt":
            bot_input_ids = dialogpt_tokenizer.encode(prompt + dialogpt_tokenizer.eos_token, return_tensors='pt')
            chat_ids = dialogpt_model.generate(bot_input_ids, max_length=250, pad_token_id=dialogpt_tokenizer.eos_token_id, do_sample=True)
            text = dialogpt_tokenizer.decode(chat_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            print(text)
        
        elif mode == "bart":
            req = requests.post("https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
                data=json.dumps({"inputs":prompt}),
                headers={
                    "Origin": "https://huggingface.co",
                    "Content-Type": "text/plain;charset=UTF-8",
                    "Referer": "https://huggingface.co/",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Connection": "keep-alive"
                }
            )
            text = json.loads(req.text)[0]["summary_text"]
            print(text)
    print("")
