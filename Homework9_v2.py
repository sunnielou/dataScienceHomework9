from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print("Model loaded.")

def generate_response(prompt, max_length=55, temperature=0.5, top_p=0.2):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def filter_response(prompt, response):
    if response.lower().startswith(prompt.lower()):
        response = response[len(prompt):].strip()
    return response

# Function to handle /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hi! I am your AI assistant with TinyLlama. How can I help you?")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    print(f"Message: {user_message}")
    
    try:
        prompt = f"{user_message}"
        ai_response = generate_response(prompt)
        filtered_response = filter_response(user_message, ai_response)
        print(f"Response: {filtered_response}")
    except Exception as e:
        filtered_response = "Sorry, there was an error."
        print(f"Error: {e}")
    await update.message.reply_text(filtered_response)



# Bot setup
def main():
    TELEGRAM_TOKEN = '7228272158:AAFRA5EqnIK-PJa2OwZPcABLhuSOyjciRBs'

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot running!")
    application.run_polling()

if __name__ == "__main__":
    main()
