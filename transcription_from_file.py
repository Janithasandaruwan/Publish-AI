from deepgram import Deepgram
import json, os
import random
import string
from pydub import AudioSegment
import io
import textwrap
from chapter_split import split_chapters
import nltk
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from difflib import Differ
import difflib

dg = Deepgram('086b8753f4516ed5447fcd9973957df514837f87')
MIMETYPE = 'wav'
params = {
    "punctuate": True,
    "model": 'general',
    "tier": 'nova'
}

book_dict_before_edited = {}
grammar_edited_book = {}
tone_edited_book = {}

nltk.download('punkt')
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# Function for reset all the data
def reset_system_audio_file():
    book_dict_before_edited.clear()
    grammar_edited_book.clear()
    tone_edited_book.clear()
    delete_files_in_directory('/tmp/gradio/wave_files/')
    delete_files_in_directory('/tmp/gradio/')

# Delete files in a directory
def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

# Function for convert audio files to wav format
def convert_to_wav(input_audio, output_path):
    audio = AudioSegment.from_file(io.BytesIO(input_audio))
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")


# Function for Deepgram and save transcription files as json files
def deepgram_transcription(audio_files):
    for audio_file in audio_files:
        file_name = os.path.basename(audio_file).split(".")[0]
        with open(audio_file, "rb") as f:
            source = {"buffer": f, "mimetype": 'audio/' + MIMETYPE}
            data = dg.transcription.sync_prerecorded(source, params)
            result = {"text": data['results']['channels'][0]['alternatives'][0]['transcript']}
            random_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            json_file_path = f"/tmp/gradio/wave_files/{file_name}_{random_code}.json"

            with open(json_file_path, "w") as transcript:
                json.dump(result, transcript)

    return "/tmp/gradio/wave_files/"


# Create transcription json files
def generate_transcript(scripts_directory):
    file_count = 1
    transcript_text_list = []
    chapter_dict = {}

    for filename in os.listdir(scripts_directory):
        file_path = os.path.join(scripts_directory, filename)
        if os.path.isfile(file_path) and filename.endswith(".json"):
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                wrapped_text = textwrap.fill(data['text'], width=80)
                chapter_dict[f'transcrption_file_{file_count}'] = wrapped_text
                file_count += 1
    return chapter_dict


# Function for generate whole book dict
def generate_whole_book(transcript_dict):
    whole_book_dict = {}

    for keys, texts in transcript_dict.items():
        splited_chapters = split_chapters(texts)
        whole_book_dict.update(splited_chapters)

    return display_whole_book(whole_book_dict)


# Display whole book
def display_whole_book(whole_book_dict):
    book_before_edited = ""

    def custom_sort_key(key):
        if key == 'Introduction':
            return 0
        else:
            return int(key.split(' ')[-1])

    sorted_book_dict = dict(sorted(whole_book_dict.items(), key=lambda x: custom_sort_key(x[0])))


    global book_dict_before_edited
    book_dict_before_edited = sorted_book_dict

    for keys, text in sorted_book_dict.items():
        book_before_edited += f"--------------------{keys}--------------------"
        book_before_edited += "\n"
        book_before_edited += text
        book_before_edited += "\n"
  
    return book_before_edited


# Convert all audio files into wave files
def transcribe_audio_files(audio_files):
    converted_audio_paths = []

    for audio_file in audio_files:
        file_path = audio_file.name
        print(file_path)
        audio_file_name = os.path.basename(file_path).split(".")[0]
        folder_path = "/tmp/gradio/wave_files"

        # Create directory for wave files if not exists
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
            except:
                pass

        # Replace spaces with underscores in the filename
        audio_file_name_cleaned = audio_file_name.replace(" ", "_")
        # Generate a random code
        random_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

        # Create the converted audio file path with the random code
        converted_audio_path = os.path.join(folder_path,
                                            os.path.splitext(audio_file_name_cleaned)[0] + f"_{random_code}.wav")
        converted_audio_paths.append(converted_audio_path)

        # Read the audio file from Google Drive into memory
        with open(file_path, "rb") as f:
            audio_path = f.read()

        # Convert the audio from Google Drive to WAV format
        convert_to_wav(audio_path, converted_audio_path)
        print("Audio conversion to WAV format complete:", audio_file.name)

    wave_audio_paths = deepgram_transcription(converted_audio_paths)
    transcript_dict = generate_transcript(wave_audio_paths)
    whole_book_dict = generate_whole_book(transcript_dict)
    return whole_book_dict


# Function for langchain
def langchain_function(prompt):
    chat_model = ChatOpenAI(temperature=0, openai_api_key="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",
                            model_name='gpt-3.5-turbo')
    prompt = prompt
    messages = [HumanMessage(content=prompt)]
    edited_ouput = chat_model.predict_messages(messages)
    wrapped_edited_text = textwrap.fill(edited_ouput.content, width=80)
    return wrapped_edited_text

# Function for show text difference
def diff_texts(text1, text2):
    d = Differ()
    diff = list(d.compare(text1.split(), text2.split()))

    diff_text = []

    for word in diff:
        if word.startswith("- "):
            word = word[2:]
            diff_text.append(f"<del style='color: red;'>{word}</del>")
        elif word.startswith("+ "):
            word = word[2:]
            diff_text.append(f"<ins style='color: green;'>{word}</ins>")
        elif word.startswith("  "):
            diff_text.append(word[2:])
        else:
            continue

    final_text = " ".join(diff_text)
    return final_text
    
# Function for check Grammar and Spellings
def check_grammar_spellings_file():
    grammar_edited_book_now = {}
    for key, text in book_dict_before_edited.items():
        try:
            grammer_edited = grammar_prompt(text)
        except:
            pass
            grammer_edited = text
        grammar_edited_book_now[key] = grammer_edited

    global grammar_edited_book
    grammar_edited_book = grammar_edited_book_now
    return display_whole_book(grammar_edited_book_now)


# Grammar prompt
def grammar_prompt(sentences):
    sentences.strip()  
    prompt = """I want you act as a proofreader. I will provide you below text paragraphs and I would like you to review them for any spelling, grammar, or punctuation errors. Once you have finished reviewing the text, provide me with any necessary corrections or suggestions to improve the text."""
    prompt += "\n"
    prompt += sentences
    output_text = langchain_function(prompt)
    process_edited_text = textwrap.fill(output_text, width=80)
    process_edited_text.strip()
    return process_edited_text

# Function for correct the Book tone
def check_tone_of_the_book_file(tone):
    tone_edited_book_now = {}
    for key, text in grammar_edited_book.items():
        try:
            tone_edited = tone_prompt(text, tone)
        except:
            pass
            tone_edited = text
        tone_edited_book_now[key] = tone_edited

    global tone_edited_book
    tone_edited_book = tone_edited_book_now
    return display_whole_book(tone_edited_book_now)

# Tone prompt
def tone_prompt(text, tone):
    new_tone = tone.lower()
    text.strip()
    prompt = f"""You are a book reader. Read the given chapter text below and detect the tone of the text. Your job is to convert this existing tone to a new {new_tone} tone. But don't summarize any sentence in the chapter text."""
    prompt += "\n"
    prompt += text
    output_text = langchain_function(prompt)
    process_edited_text = textwrap.fill(output_text, width=80)
    process_edited_text.strip()
    return process_edited_text









