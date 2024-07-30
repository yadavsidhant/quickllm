from transformers import AutoTokenizer

def chat_with_model(model, message):
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    
    input_ids = tokenizer.encode(message, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def start_chat_interface(model):
    print("Welcome to QuickLLM Chat! Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        response = chat_with_model(model, user_input)
        print(f"Model: {response}")