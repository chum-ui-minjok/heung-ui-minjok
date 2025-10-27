import time
import whisper
import edge_tts
import asyncio
import os
import torch

# GPU 사용 가능 여부 출력
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# 음성 명령 키워드 리스트
PLAY_MUSIC_KEYWORDS = ["노래 틀", "노래를 틀", "음악 틀"]
START_EXERCISE_KEYWORDS = ["체조 시작", "운동할래"]
PAUSE_KEYWORDS = ["그만", "멈춰", "일시정지"]
RESUME_KEYWORDS = ["계속", "다시 틀어줘"]
NEXT_SONG_KEYWORDS = ["다음"]
STOP_ALL_KEYWORDS = ["종료", "끝"]
EMERGENCY_KEYWORDS = ["도와줘", "살려줘"]

# 키워드에 따른 명령 분기 함수
def command_match(text):
    text = text.lower()
    if any(k in text for k in PLAY_MUSIC_KEYWORDS):
        return "PLAY_MUSIC"
    elif any(k in text for k in START_EXERCISE_KEYWORDS):
        return "START_EXERCISE"
    elif any(k in text for k in PAUSE_KEYWORDS):
        return "PAUSE"
    elif any(k in text for k in RESUME_KEYWORDS):
        return "RESUME"
    elif any(k in text for k in NEXT_SONG_KEYWORDS):
        return "NEXT_SONG"
    elif any(k in text for k in STOP_ALL_KEYWORDS):
        return "STOP_ALL"
    elif any(k in text for k in EMERGENCY_KEYWORDS):
        return "EMERGENCY"
    else:
        return "UNKNOWN"

# 명령어에 따른 피드백 메시지 생성
def generate_feedback(command_type):
    feedbacks = {
        "PLAY_MUSIC": "노래를 재생할게요.",
        "START_EXERCISE": "체조 모드로 전환합니다.",
        "PAUSE": "재생을 일시정지할게요.",
        "RESUME": "노래를 계속 재생할게요.",
        "NEXT_SONG": "다음 노래로 넘어갈게요.",
        "STOP_ALL": "모든 재생을 종료합니다.",
        "EMERGENCY": "긴급 상황입니다. 도움을 요청합니다.",
        "UNKNOWN": "명령을 이해하지 못했습니다. 다시 말씀해 주세요."
    }
    return feedbacks.get(command_type, "명령을 실행했습니다.")

# Edge TTS로 음성 합성 및 재생
async def tts_and_play(text, voice="ko-KR-InJoonNeural", filename="output.mp3"):
    tts = edge_tts.Communicate(text, voice)
    await tts.save(filename)
    os.system(f"start {filename}")

# Whisper 음성 인식 모델 불러오기 (GPU 자동 감지 가능)
model = whisper.load_model("medium")

# 음성 파일 경로
audio_path = r"C:\Users\SSAFY\Documents\소리 녹음\test3_voice.m4a"

# 음성 인식 및 처리 시작
start = time.time()
result = model.transcribe(audio_path)
end = time.time()

print(f"Processing time: {end - start} seconds")
print("Recognized text:", result["text"])

# 명령 분기
command_type = command_match(result["text"])
print("Command type:", command_type)

# 피드백 생성 및 음성 재생
feedback_text = generate_feedback(command_type)
asyncio.run(tts_and_play(feedback_text))
