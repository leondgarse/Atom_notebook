given a video input, it could be 2 hrs lone:
- extract as audio firstly.
- start `server.py`
- feed audio data into server to extract transcription

ser ver can be started by `uvicorn "server:app" --host 0.0.0.0 --port 8000 --reload`
