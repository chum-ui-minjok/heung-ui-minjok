import subprocess
import time
import pvporcupine
from pvrecorder import PvRecorder

# --- 설정 값 ---
ACCESS_KEY = "------"
KEYWORD_PATH = "흥부야_ko_raspberry-pi_v3_0_0.ppn"
MODEL_PATH = "porcupine_params_ko.pv"     # 다운로드한 한국어 모델 파일명
MICROPHONE_INDEX = 2  # Jabra 마이크 인덱스
SENSITIVITY = 0.7

# --- 실행할 명령어 (DISPLAY=:0 으로 실제 모니터 지정) ---
BLANK_SCREEN_CMD = "DISPLAY=:0 xset s activate"
WAKE_SCREEN_CMD = "DISPLAY=:0 xset s reset"
BROWSER_CMD = "DISPLAY=:0 chromium-browser --kiosk http://k13a103.p.ssafy.io:8080/health"

def run_command(command):
    """명령어를 실행하는 함수"""
    try:
        subprocess.run(command, shell=True, check=True)
    except Exception as e:
        print(f"명령어 '{command}' 실행 중 오류: {e}")

def kill_process(process_name):
    """프로세스를 종료하는 함수"""
    try:
        subprocess.run(['pkill', '-f', process_name], check=False)
        print(f"'{process_name}' 프로세스를 종료 시도했습니다.")
    except Exception as e:
        print(f"프로세스 종료 중 오류: {e}")

# --- 메인 로직 ---
porcupine = None
recorder = None

try:
    # 1. 그래픽 환경(X-server)을 먼저 시작시킵니다.
    # 이 세션 위에서 xset과 chromium이 동작하게 됩니다.
    print("그래픽 세션을 시작합니다...")
    run_command("startx &")
    time.sleep(10) # X-server가 완전히 켜질 때까지 충분히 기다립니다.

    # 2. xset 초기 설정
    print("화면 보호기 설정을 초기화합니다...")
    run_command("DISPLAY=:0 xset s 30") # 30초 후 자동 꺼짐
    run_command("DISPLAY=:0 xset s blank")

    # 3. Porcupine 엔진 및 마이크 초기화
    porcupine = pvporcupine.create(
        access_key=ACCESS_KEY,
        keyword_paths=[KEYWORD_PATH],
        model_path=MODEL_PATH,
        sensitivities=[SENSITIVITY]
    )
    recorder = PvRecorder(
        device_index=MICROPHONE_INDEX,
        frame_length=porcupine.frame_length
    )
    recorder.start()

    print("시스템 준비 완료. '흥부야'를 기다립니다...")
    # 시작 시, 화면을 끕니다.
    run_command(BLANK_SCREEN_CMD)

    while True:
        pcm = recorder.read()
        result = porcupine.process(pcm)

        if result >= 0:
            print("'흥부야'가 감지되었습니다! 체조 화면으로 전환합니다...")
            
            # 화면을 깨우고 브라우저를 실행
            run_command(WAKE_SCREEN_CMD)
            kill_process('chromium-browser') # 혹시 켜져있을지 모를 브라우저 정리
            run_command(BROWSER_CMD + " &") # 백그라운드(&)로 실행
            
            # (여기에 나중에 "그만할래" 로직 추가)
            # "그만할래" 감지 시:
            # kill_process('chromium-browser')
            # run_command(BLANK_SCREEN_CMD)

except KeyboardInterrupt:
    print("프로그램을 종료합니다.")
finally:
    if recorder is not None:
        recorder.delete()
    if porcupine is not None:
        porcupine.delete()
    kill_process('chromium-browser')
    run_command("DISPLAY=:0 xset s default") # 화면 보호기 설정 원상복구
    kill_process('Xorg') # 그래픽 세션 전체 종료