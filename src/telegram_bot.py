from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackContext
from telegram import Update
from pydub import AudioSegment
import os
from project_dirs import DATA_DIR
import whisper

whisper_model = whisper.load_model("base")

# COMMANDS
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        """Ciao! Sono un bot che risponde alle domande sul documento 
        'BANDO CONneSSi CONtributi per lo Sviluppo di Strategie digitali 
        per i mercati globali. Anno 2024. Fammi una domanda.""")
    
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        """Scrivi una domanda sul documento
        'BANDO CONneSSi CONtributi per lo Sviluppo di Strategie digitali 
        per i mercati globali. Anno 2024. """)
    
async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        """.""")

# RESPONSES
def process_question(
    question: str,
    retriever_faq, 
    faq_similarity_threshold,
    faq_df,
    query_engine,
    qa_relevance_guard,
    provenance_v1_guard,
    
) -> str:
    """
    Main processing function
    """
    # Check similarity to sample questions (FAQ)
    # Use FAQ answer if the user question is similar (faq_similarity_threshold: 0.9)
    ######################
    
    most_silmilar_node = retriever_faq.retrieve(question)[0]
    if most_silmilar_node.score > faq_similarity_threshold:
        node_text = most_silmilar_node.text
        answer = faq_df.loc[faq_df.Domanda==node_text, "Risposta"].values[0]
        logging.info("Returning stored answer to the question %s", node_text)
    else:
        answer = 'Informazione non trovata'
        # Query with reranking
        ######################
        raw_answer, sourse_nodes = query_engine.query(question)
        if len(sourse_nodes) != 0:
            # Apply guardrails ai
            ######################
            raw_llm_output, validated_output, *rest = qa_relevance_guard.parse(
            llm_output=raw_answer, metadata={'question':question}
            )
            if validated_output:
                raw_llm_output, validated_output, *rest = provenance_v1_guard.parse(
                    llm_output=raw_answer, 
                    metadata={'query_function':partial(query_function, sources=[i.text for i in source_nodes])}
                )
                if validated_output:
                    answer = validated_output
    return answer

async def handle_message(
    update: Update, 
    context: ContextTypes.DEFAULT_TYPE,
    # retriever_faq, 
    # faq_similarity_threshold,
    # faq_df,
    # query_engine,
    # qa_relevance_guard,
    # provenance_v1_guard,
):
    message_type: str = update.message.chat.type
    text: str = update.message.text
    
    # logging.info("User %s in %s: '%s'", update.message.chat.id, message_type, text)
    print("User {update.message.chat.id} in {message_type}: {text}")
    
    response: str = process_question(
                        text,
                        retriever_faq, 
                        faq_similarity_threshold,
                        faq_df,
                        query_engine,
                        qa_relevance_guard,
                        provenance_v1_guard,
                    )
    print('Bot:', response)
    await update.message.reply_text(response)

def convert_ogg_to_mp3(ogg_filepath, file_id):
    mp3_filepath = os.path.join(DATA_DIR, f"{file_id}.mp3")
    audio = AudioSegment.from_file(ogg_filepath, format="ogg")
    audio.export(mp3_filepath, format="mp3")
    return mp3_filepath

def convert_speech_to_text(audio_filepath, model):
    data = model.transcribe(audio_filepath)
    return data["text"]
    
async def handle_voice_message(
    update: Update, 
    context: CallbackContext,
):   
    message_type = update.message.chat.type
    # voice = update.message.voice
    file_id = update.message.voice.file_id
    print(f"file_id {file_id}")
    # new_file = await context.bot.get_file(update.message.voice.file_id)
    new_file = await context.bot.get_file(file_id)
    print(f"new_file {new_file}")  
    await new_file.download_to_drive(f"{file_id}.ogg")

    mp3_filepath = convert_ogg_to_mp3(f"{file_id}.ogg", file_id)
    extracted_text = convert_speech_to_text(mp3_filepath, whisper_model)

    response: str = process_question(
                    extracted_text,
                    retriever_faq, 
                    faq_similarity_threshold,
                    faq_df,
                    query_engine,
                    qa_relevance_guard,
                    provenance_v1_guard,
                )
    print('Bot:', response)
    await update.message.reply_text(response)
    os.remove(file_path)
    os.remove(mp3_filepath)
    
# ERRORS    
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Update {update} caused error {context.error}")

    
def run_bot(token):
    # whisper_model = whisper.load_model("base")
    print('Starting bot')
    app = Application.builder().token(token).build()

    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', start_command))
    # app.add_handler(CommandHandler('custom', start_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

    # Errors
    app.add_error_handler(error)

    app.run_polling(poll_interval=5)
