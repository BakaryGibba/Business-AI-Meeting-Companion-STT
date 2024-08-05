import torch
from transformers import pipeline
import gradio as gr
import logging

# Set up logging
logging.basicConfig(filename="transcription_errors.log", level=logging.ERROR)

# Function to transcribe audio using the OpenAI Whisper model
def transcript_audio(audio_file):
    try:
        # Initialize the speech recognition pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",  # Try using a different model variant
            chunk_length_s=10,  # Reduce chunk length
        )
        
        # Transcribe the audio file
        result = pipe(audio_file, return_timestamps=True)
        return result["text"]
        
    except IndexError as e:
        # Log detailed information about the error
        error_message = f"IndexError: {e}\n"
        error_message += f"Audio file path: {audio_file}\n"
        error_message += "Please ensure the audio file is correctly formatted and try again."
        logging.error(error_message)
        return error_message
    except ValueError as e:
        # Handle potential ValueErrors
        error_message = f"ValueError: {e}\n"
        error_message += f"Audio file path: {audio_file}\n"
        error_message += "Please ensure the audio file is correctly formatted and try again."
        logging.error(error_message)
        return error_message
    except Exception as e:
        # Handle any other exceptions
        logging.error(f"Unexpected error: {e}")
        return str(e)

# Set up Gradio interface
audio_input = gr.Audio(type="filepath")  # Audio input
output_text = gr.Textbox()  # Text output

# Create the Gradio interface with the function, inputs, and outputs
iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input, outputs=output_text,
    title="Audio Transcription App",
    description="Upload the audio file"
)

# Launch the Gradio app
iface.launch(debug=True)
