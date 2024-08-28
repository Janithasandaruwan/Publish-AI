import numpy as np
import gradio as gr
import shutil
import tempfile
from difflib import Differ
from transcription_from_file import transcribe_audio_files, diff_texts, check_grammar_spellings_file, check_tone_of_the_book_file, reset_system_audio_file
from transcription_from_microphone import read_reocrd_audio_file, check_grammar_spellings_audio, check_tone_of_the_book_audio, reset_system_microphone_file

with gr.Blocks() as application:

    gr.Markdown("""
    # Publish-AI Book Production
    Select Your Audio Input Method
    """)

    with gr.Tab("Upload Audio Files"):
        audio_files = gr.Files(file_count="multiple")
        transcription_button = gr.Button("Start Transcription")
        transcription_text = gr.Textbox(label="Transcription Text")

        with gr.Row():
            grammer_button1 = gr.Button("Correct Grammar And Spelling")
            show_diff1 = gr.Button("Show Differences")
        with gr.Row():
            grammer_text1 = gr.Textbox(label="Grammar And Spelling Corrected Text")
            show_grammer_diff1 = gr.outputs.HTML(label="Grammar And Spelling Differences")

        with gr.Row():
            tone_type1 = gr.Dropdown(
                    ["Formal", "Informal", "Optimistic", "Pessimistic", "Joyful", "Sad", "Hypocritical", "Fearful", "Hopeful", "Humorous", "Serious"], label="Select A Tone", default="Formal", allow_custom_value=True)
            tone_button1 = gr.Button("Correct Tone")
            show_diff112 = gr.Button("Show Differences")
        with gr.Row():
            tone_text1 = gr.Textbox(label="Tone Checked Text")
            show_tone_diff1 = gr.outputs.HTML(label="Tone Differences")

        reset_button1 = gr.Button("RESET")

    with gr.Tab("Record Audio Files"):
        gr.Markdown("""
        # Guidelines
        Read these key words before and after your speech
        """)
        gr.Markdown("""
        [Start Introduction] Your Voice
        """)
        gr.Markdown("""
        [Chapter One] Your Voice [End Chapter]
        """)
        audio_record = gr.Audio(source="microphone", type="filepath")
        process_button = gr.Button("Start Transcription")
        disply_text = gr.Textbox(label="Transcription Text")

        with gr.Row():
            grammer_button2 = gr.Button("Correct Grammar And Spelling")
            show_diff2 = gr.Button("Show Differences")
        with gr.Row():
            grammer_text2 = gr.Textbox(label="Grammar And Spelling Checked Text")
            show_grammer_diff2 = gr.outputs.HTML(label="Grammar And Spelling Differences")
                  

        with gr.Row():
            tone_type2 = gr.Dropdown(
                    ["Formal", "Informal", "Optimistic", "Pessimistic", "Joyful", "Sad", "Hypocritical", "Fearful", "Hopeful", "Humorous", "Serious"], label="Select A Tone", default="Formal", allow_custom_value=True)
            tone_button2 = gr.Button("Correct Tone")
            show_diff22 = gr.Button("Show Differences")
        with gr.Row():
            tone_text2 = gr.Textbox(label="Tone Checked Text")
            show_tone_diff2 = gr.outputs.HTML(label="Tone Differences")

        reset_button2 = gr.Button("RESET")

    # Transcription from file
    transcription_button.click(transcribe_audio_files, inputs=audio_files, outputs=transcription_text)
    # Grammar
    grammer_button1.click(check_grammar_spellings_file, outputs=grammer_text1)
    show_diff1.click(diff_texts, inputs=[transcription_text, grammer_text1], outputs=show_grammer_diff1)
    # Tone
    tone_button1.click(check_tone_of_the_book_file, inputs=tone_type1, outputs=tone_text1)
    show_diff112.click(diff_texts, inputs=[grammer_text1, tone_text1], outputs=show_tone_diff1)
    reset_button1.click(reset_system_audio_file)
    

    # Transcription from audio
    process_button.click(read_reocrd_audio_file, inputs=audio_record, outputs=disply_text)
    # Grammar
    grammer_button2.click(check_grammar_spellings_audio, outputs=grammer_text2)
    show_diff2.click(diff_texts, inputs=[disply_text, grammer_text2], outputs=show_grammer_diff2)
    # Tone
    tone_button2.click(check_tone_of_the_book_audio, inputs=tone_type2, outputs=tone_text2)
    show_diff22.click(diff_texts, inputs=[grammer_text2, tone_text2], outputs=show_tone_diff2)
    reset_button2.click(reset_system_microphone_file)


    
if __name__ == "__main__":
    application.launch(share=True)