import whisper 
import json
import os

model = whisper.load_model("large-v2")

audios = os.listdir("audios")

for audio in audios: 
    if("_" in audio):
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
        result = model.transcribe(audio = f"audios/{audio}",  
                              language="hi",
                              task="translate",
                              word_timestamps=False )
        
        chunks = []
        for segment in result["segments"]:
            chunks.append({"number": number, "title":title, "start": segment["start"], "end": segment["end"], "text": segment["text"]})
        
        chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

        with open(f"jsons/{audio}.json", "w") as f:
            json.dump(chunks_with_metadata,f)

#whisper is a library for automatic speech recognition (ASR) that can transcribe audio files into text. In this code, we are using the whisper library to transcribe audio files from the "audios" folder and save the transcriptions as JSON files in the "jsons" folder.