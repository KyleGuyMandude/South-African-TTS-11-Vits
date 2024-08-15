import argparse
import json
import os
import torch
from TTS.utils.synthesizer import Synthesizer
from MLFlowUtils.download_model import download_validate_model

class TTSModel:
    """
    TTS Model Class, for use with the Vits based models.
    """

    def __init__(self, local_path=None, logger=None):
        self.local_path = local_path
        self.logger = logger

    def download_model(self, source):
        return download_validate_model(
            source=source, 
            download_path=self.local_path, 
            model_name=self.model_name, 
            max_retries=3, 
            logger=self.logger
        )

    def load_model(self):
        config_path = os.path.join(self.local_path, "config.json")
        speaker_file = f"./{os.path.join(self.local_path, 'speakers.pth')}"
        language_file = f"./{os.path.join(self.local_path, 'language_ids.json')}"

        if os.path.isfile(language_file):
            with open(config_path, "r") as config:
                data = json.load(config)
            data["model_args"]["language_ids_file"] = language_file
            data["language_ids_file"] = language_file
            with open(config_path, "w") as config:
                json.dump(data, config, indent=4)
        else:
            self.language_file = None
            language_file = None

        if os.path.isfile(speaker_file):
            with open(config_path, "r") as config:
                data = json.load(config)
            data["model_args"]["speakers_file"] = speaker_file
            data["speakers_file"] = speaker_file
            with open(config_path, "w") as config:
                json.dump(data, config, indent=4)
            speakers = torch.load(os.path.join(self.local_path, "speakers.pth"))
            self.speakers = [key for key, value in speakers.items()]
        else:
            self.speakers = None
            speaker_file = None

        self.model = Synthesizer(
            os.path.join(self.local_path, "model.pth"),
            config_path,
            speaker_file,
            language_file,
            None,
            None,
            False,
        )

    def predict(self, text, speaker=None, language=None, save_file=True, file_path="default.wav"):
        if self.speakers is not None:
            self.outputs = self.model.tts(text, speaker, language)
        else:
            self.outputs = self.model.tts(text, None, language)

        if save_file:
            self.file_path = self.post_processing(file_path)
            return self.outputs, self.file_path

        return self.outputs, None

    def post_processing(self, filepath):
        if not filepath.endswith("wav"):
            filepath = f"{filepath}.wav"
        self.model.save_wav(self.outputs, filepath)
        return filepath

def main():
    parser = argparse.ArgumentParser(description="Text-to-Speech CLI using a VITS-based model.")
    parser.add_argument("--text", type=str, required=True, help="Input text to be synthesized.")
    parser.add_argument("--local_path", type=str, required=True, help="Path to the local model files.")
    parser.add_argument("--speaker", type=str, help="The speaker voice to use.")
    parser.add_argument("--language", type=str, help="The language the text is in.")
    parser.add_argument("--save_file", type=bool, default=True, help="Whether to save the output as a WAV file.")
    parser.add_argument("--file_path", type=str, default="output.wav", help="File path to save the generated audio.")

    args = parser.parse_args()

    tts_model = TTSModel(local_path=args.local_path)
    tts_model.load_model()

    outputs, file_path = tts_model.predict(
        text=args.text,
        speaker=args.speaker,
        language=args.language,
        save_file=args.save_file,
        file_path=args.file_path
    )

    if file_path:
        print(f"Generated audio saved at: {file_path}")
    else:
        print("Audio generated.")

if __name__ == "__main__":
    main()
