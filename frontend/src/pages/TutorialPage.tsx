import { useEffect, useState, useRef, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { gameStartApi } from '@/api/game';
import { useGameStore } from '@/store/gameStore';
import { usePoseDetection } from '@/hooks/usePoseDetection';
import { POSE_LANDMARKS } from '@/types/pose';
import type { GameStartResponse } from '@/types/game';
import LoadingDots from '@/components/icons/LoadingDots';
import './TutorialPage.css';

type Step = 1 | 2 | 3;

function TutorialPage() {
  const nav = useNavigate();
  const location = useLocation();
  const setFromApi = useGameStore((s) => s.setFromApi);

  const [loading, setLoading] = useState(true);
  const [songId, setSongId] = useState<number | null>(null);
  const [step, setStep] = useState<Step>(1);
  const [isFinalMessage, setIsFinalMessage] = useState(false);
  const [showCheck, setShowCheck] = useState(false);

  // 포즈 감지 훅 (MediaPipe 사용)
  const {
    videoRef,
    isReady,
    isCameraOn,
    error,
    currentLandmarks,
    startCamera,
    stopCamera,
    startDetection,
    stopDetection,
  } = usePoseDetection();

  const displayVideoRef = useRef<HTMLVideoElement | null>(null);
  const isCameraReady = isCameraOn && isReady && !error;

  // 포즈 매칭 상태
  const [isPoseMatched, setIsPoseMatched] = useState(false);
  const matchStartTimeRef = useRef<number | null>(null);
  const MATCH_DURATION = 3000; // 3초 유지 필요

  // 타이머 관리 (단계 전환용)
  const timersRef = useRef<number[]>([]);
  const sequenceStartedRef = useRef(false);

  const isStep1 = step === 1;
  const cameraClass =
  step === 1
    ? 'camera-state-hidden'
    : step === 2
    ? 'camera-state-show-step2'
    : 'camera-state-show-step3';


  // 페이지 진입 시 게임 데이터 로드 (음성 명령 처리 포함)
  useEffect(() => {
    setLoading(true);

    const voiceCommandData = location.state as GameStartResponse | undefined;

    if (voiceCommandData?.gameInfo) {
      console.log('음성 명령으로 받은 게임 데이터를 store에 저장:', voiceCommandData);
      setFromApi(voiceCommandData);
      setSongId(voiceCommandData.gameInfo.songId);
      setLoading(false);
    } else {
      const initGameData = async () => {
        try {
          const res = await gameStartApi();
          setFromApi(res);
          setSongId(res.gameInfo.songId);
          console.log('게임 데이터 로드 완료:', res);
        } catch (e) {
          console.error('게임 데이터 로드 실패:', e);
        } finally {
          setLoading(false);
        }
      };
      initGameData();
    }
  }, [location.state, setFromApi]);

  // ===== 카메라 시작 / 정리 =====
  useEffect(() => {
    startCamera();

    return () => {
      stopCamera();
      timersRef.current.forEach((id) => clearTimeout(id));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // step 2에서 포즈 감지 시작
  useEffect(() => {
    if (step === 2 && isCameraReady) {
      startDetection();
    } else {
      stopDetection();
    }
  }, [step, isCameraReady, startDetection, stopDetection]);

  // 표시용 비디오에 stream 연결 (한 번만)
  useEffect(() => {
    if (displayVideoRef.current && videoRef.current && videoRef.current.srcObject) {
      if (displayVideoRef.current.srcObject !== videoRef.current.srcObject) {
        displayVideoRef.current.srcObject = videoRef.current.srcObject;
      }
    }
  }, [isCameraReady, step]);

  // ===== 단계 자동 진행 (step 1 → step 2만 타이머, 이후는 포즈 매칭 기반) =====
  useEffect(() => {
    if (loading || !songId || !isCameraReady) return;
    if (sequenceStartedRef.current) return;

    sequenceStartedRef.current = true;
    setStep(1);
    setIsFinalMessage(false);

    // step1: 3초 후 -> step 2 진입
    const t1 = window.setTimeout(() => setStep(2), 3000);

    timersRef.current = [t1];

    return () => {
      timersRef.current.forEach((id) => clearTimeout(id));
    };
  }, [loading, songId, isCameraReady, nav]);


  // ===== 포즈 매칭 체크 =====
  const checkPoseMatch = useCallback((landmarks: number[][]): boolean => {
    if (!displayVideoRef.current) return false;

    const width = displayVideoRef.current.clientWidth;
    const height = displayVideoRef.current.clientHeight;
    const centerX = width / 2;

    // 가이드 영역 정의 (silhouette.svg 기준)
    // SVG가 bottom: 20%에 배치되어 있고, 실루엣 위치에 맞춤
    // 머리 중심: 화면 상단에서 약 18%, 어깨: 약 32%
    const headY = height * 0.18;
    const headRadius = width * 0.12;
    const shoulderY = height * 0.32;
    const shoulderWidth = width * 0.50;

    // 랜드마크 추출 (정규화된 좌표 0~1 → 픽셀 좌표)
    const nose = landmarks[POSE_LANDMARKS.NOSE];
    const leftShoulder = landmarks[POSE_LANDMARKS.LEFT_SHOULDER];
    const rightShoulder = landmarks[POSE_LANDMARKS.RIGHT_SHOULDER];

    if (!nose || !leftShoulder || !rightShoulder) return false;

    // 거울 효과 적용 (카메라가 scaleX(-1)이므로 x 좌표 반전)
    const noseX = (1 - nose[0]) * width;
    const noseY = nose[1] * height;
    const leftShoulderX = (1 - leftShoulder[0]) * width;
    const leftShoulderY = leftShoulder[1] * height;
    const rightShoulderX = (1 - rightShoulder[0]) * width;
    const rightShoulderY = rightShoulder[1] * height;

    // 허용 오차 (더 넓게 설정)
    const tolerance = width * 0.15;

    // 머리(코) 위치 체크: 가이드 머리 영역 근처에 있는지
    const headDistFromCenter = Math.sqrt((noseX - centerX) ** 2 + (noseY - headY) ** 2);
    const isHeadInRange = headDistFromCenter < headRadius + tolerance;

    // 어깨 위치 체크: 가이드 어깨 라인 근처에 있는지
    const shoulderCenterX = (leftShoulderX + rightShoulderX) / 2;
    const shoulderCenterY = (leftShoulderY + rightShoulderY) / 2;
    const isShoulderXInRange = Math.abs(shoulderCenterX - centerX) < tolerance;
    const isShoulderYInRange = Math.abs(shoulderCenterY - shoulderY) < tolerance * 1.5;

    // 어깨 너비 체크: 너무 작거나 크지 않은지
    const userShoulderWidth = Math.abs(leftShoulderX - rightShoulderX);
    const isShoulderWidthOk = userShoulderWidth > shoulderWidth * 0.4 && userShoulderWidth < shoulderWidth * 1.6;

    return isHeadInRange && isShoulderXInRange && isShoulderYInRange && isShoulderWidthOk;
  }, []);

  // ===== 포즈 매칭 체크 (SVG 색상 변경용) =====
  useEffect(() => {
    if (step !== 2 || !displayVideoRef.current) return;

    // 포즈 매칭 여부 확인
    let matched = false;
    if (currentLandmarks && currentLandmarks.length > 0) {
      matched = checkPoseMatch(currentLandmarks);
    }

    // 매칭 상태 업데이트
    setIsPoseMatched(matched);
  }, [step, currentLandmarks, checkPoseMatch]);

  // ===== 포즈 매칭 3초 유지 시 step 3으로 전환 =====
  useEffect(() => {
    if (step !== 2) {
      matchStartTimeRef.current = null;
      return;
    }

    if (isPoseMatched) {
      if (matchStartTimeRef.current === null) {
        matchStartTimeRef.current = Date.now();
      } else {
        const elapsed = Date.now() - matchStartTimeRef.current;
        if (elapsed >= MATCH_DURATION) {
          // 체크 애니메이션 표시 후 step 3으로
          setShowCheck(true);
          const timer = window.setTimeout(() => {
            setShowCheck(false);
            setStep(3);
          }, 2000);
          timersRef.current.push(timer);
        }
      }
    } else {
      matchStartTimeRef.current = null;
    }
  }, [isPoseMatched, step]);

  // 3초 카운트를 위한 인터벌
  useEffect(() => {
    if (step !== 2 || !isPoseMatched) return;

    const interval = window.setInterval(() => {
      if (matchStartTimeRef.current !== null) {
        const elapsed = Date.now() - matchStartTimeRef.current;
        if (elapsed >= MATCH_DURATION && !showCheck) {
          setShowCheck(true);
          window.setTimeout(() => {
            setShowCheck(false);
            setStep(3);
          }, 2000);
        }
      }
    }, 100);

    return () => clearInterval(interval);
  }, [step, isPoseMatched, showCheck]);

  // ===== step 3 타이머 (기존 로직 유지) =====
  useEffect(() => {
    if (step !== 3) return;

    // step 3 진입 후 5초 뒤 체크 → 최종 멘트 → 게임 시작
    const t3a = window.setTimeout(() => setShowCheck(true), 3000);
    const t3b = window.setTimeout(() => {
      setShowCheck(false);
      setIsFinalMessage(true);
    }, 5000);

    const t4 = window.setTimeout(() => {
      nav(`/game/${songId}`);
    }, 7000);

    timersRef.current.push(t3a, t3b, t4);

    return () => {
      timersRef.current.forEach((id) => clearTimeout(id));
    };
  }, [step, nav, songId]);

  // ===== 건너뛰기 버튼 핸들러 =====
  const handleSkip = useCallback(() => {
    if (step === 2) {
      setShowCheck(true);
      window.setTimeout(() => {
        setShowCheck(false);
        setStep(3);
      }, 500);
    }
  }, [step]);

  // 단계별 문구
  const renderTitle = () => {
    if (loading || !isCameraReady) {
      return (
        <>
        {/* 카메라 준비 중 */}
          <LoadingDots className="tutorial-camera-loading"/>
        </>
      );
    }

    if (isFinalMessage) {
      return (
        <>
          좋아요!
          <br />
          이제 체조를 시작합니다!
        </>
      );
    }

    switch (step) {
      case 1:
        return (
          <>
            게임을 시작하기 전,
            <br />
            간단한 준비가 필요해요!
          </>
        );
      case 2:
        return (
          <>
            카메라에 상반신이
            <br />
            잘 나오도록 앉아주세요!
          </>
        );
      case 3:
        return (
          <>
            준비가 되면 머리 위로
            <br />
            동그라미를 만들어주세요!
          </>
        );
    }
  };

  return (
    <div className="tutorial-page">
      {/* 상단 단계 인디케이터 */}
      <div className="tutorial-steps">
        {[1, 2, 3].map((n) => {
          const isActive = step === n || (n === 3 && isFinalMessage);
          return (
            <div
              key={n}
              className={`tutorial-step-circle ${
                isActive
                  ? 'tutorial-step-circle--active'
                  : 'tutorial-step-circle--inactive'
              }`}
            >
              {n}
            </div>
          );
        })}
      </div>

      {/* 체크 애니메이션 오버레이 */}
      {showCheck && (
        <div className="step-check">
          <div className="step-check__outer">
            <div className="step-check__inner">
              <span className="step-check__mark">✓</span>
            </div>
          </div>
        </div>
      )}

      {/* 숨겨진 비디오 - 카메라 초기화용 (항상 렌더링) */}
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="tutorial-hidden-video"
      />

      {/* 고정 레이아웃: 왼쪽 카메라 슬롯 + 오른쪽 텍스트 */}
      <div
        className={`tutorial-layout ${
          isStep1 ? 'tutorial-layout--step1' : 'tutorial-layout--step23'
        }`}
      >
        {/* ⬇⬇ 1단계가 아닐 때만 카메라 영역 표시 */}
        {!isStep1 && (
          <div className={`tutorial-camera-wrapper ${cameraClass}`}>
            <div className="tutorial-camera-outer">
              <div className="tutorial-camera-frame">
                {!error && !loading && isCameraReady && (
                  <>
                    {/* 표시용 비디오 - 숨겨진 비디오의 stream 복제 */}
                    <video
                      autoPlay
                      muted
                      playsInline
                      className="tutorial-camera-video"
                      ref={displayVideoRef}
                    />
                    {/* step 2에서 가이드 SVG 오버레이 */}
                    {step === 2 && (
                      <svg
                        className={`tutorial-guide-svg ${isPoseMatched ? 'matched' : ''}`}
                        viewBox="230 50 320 680"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M 356.573 388.344 L 356.637 388.345 C 358.742 388.356 360.845 388.367 362.887 388.377 C 368.475 388.379 374.111 388.379 379.749 388.4 H 379.761 C 385.579 388.425 391.375 388.445 397.162 388.445 C 401.761 388.444 406.364 388.447 410.969 388.456 H 410.975 C 412.097 388.459 413.202 388.461 414.299 388.464 C 415.392 388.467 416.477 388.468 417.562 388.471 C 443.54 388.491 476.331 386.904 503.559 407.928 L 504.851 408.947 L 505.52 409.709 L 504.686 408.816 L 508.274 412.656 C 516.747 420.426 528.057 442.771 531.475 456.784 C 534.707 470.031 533.973 484.534 533.822 504.885 C 533.804 508.318 533.791 511.752 533.785 515.186 C 533.777 524.307 533.757 533.431 533.72 542.558 C 533.681 551.972 533.658 561.39 533.641 570.814 C 533.604 591.134 533.542 611.452 533.465 631.767 L 533.402 648.763 H 483.497 C 483.456 649.13 483.414 649.505 483.371 649.896 C 483.008 653.208 482.642 656.527 482.273 659.848 C 482.003 662.272 481.733 664.697 481.466 667.121 L 481.467 667.122 L 479.267 687.024 C 478.64 692.675 478.013 698.326 477.388 703.977 C 475.914 717.313 474.437 730.65 472.957 743.988 C 471.427 757.776 469.9 771.563 468.377 785.349 C 467.059 797.288 465.739 809.226 464.415 821.165 C 463.624 828.304 462.833 835.437 462.045 842.57 C 458.534 874.383 454.884 906.285 450.63 938.193 L 448.656 953 H 331.772 L 329.171 939.069 C 328.298 934.399 327.357 929.268 326.685 924.007 L 326.667 923.866 L 326.651 923.724 C 326.439 921.819 326.234 919.966 326.029 918.117 C 325.658 914.796 325.288 911.466 324.921 908.134 C 324.654 905.71 324.385 903.285 324.116 900.861 C 323.377 894.225 322.641 887.585 321.908 880.947 C 321.127 873.868 320.343 866.789 319.558 859.708 L 315.606 823.997 C 314.084 810.21 312.558 796.424 311.029 782.64 C 309.542 769.245 308.058 755.848 306.576 742.45 C 305.948 736.767 305.32 731.088 304.69 725.409 L 302.539 706.051 C 300.433 687.039 298.366 667.916 296.591 648.763 H 246.226 L 246.194 631.735 C 246.16 613.215 246.13 594.692 246.112 576.167 C 246.103 566.715 246.096 557.274 246.079 547.834 C 246.06 538.585 246.044 529.328 246.037 520.069 V 520.064 C 246.036 516.55 246.033 513.036 246.03 509.522 C 246.016 504.465 246.003 499.384 246.002 494.3 V 494.294 C 246.002 492.728 246.003 491.208 246.003 489.689 C 245.906 466.787 248.081 443.038 267.062 420.426 L 267.566 419.826 L 268.121 419.274 C 293.804 393.829 324.113 388.42 356.507 388.344 H 356.573 Z M 397.163 405.506 C 391.337 405.506 385.512 405.485 379.686 405.459 C 376.871 405.448 374.056 405.444 371.242 405.441 L 362.798 405.438 C 360.735 405.427 358.672 405.416 356.547 405.405 C 326.067 405.476 301.186 410.542 280.144 431.388 C 265.125 449.28 262.986 467.932 263.078 489.616 L 263.077 494.297 C 263.078 499.357 263.091 504.418 263.105 509.478 C 263.108 513.004 263.111 516.53 263.112 520.056 C 263.119 529.304 263.135 538.552 263.153 547.8 C 263.171 557.25 263.178 566.701 263.187 576.151 C 263.205 594.668 263.234 613.185 263.269 631.703 H 312.206 C 312.628 636.432 313.05 641.162 313.484 646.034 C 315.865 671.894 318.794 697.709 321.66 723.528 C 322.291 729.211 322.919 734.895 323.548 740.578 C 325.029 753.972 326.514 767.366 328 780.76 C 329.53 794.548 331.055 808.336 332.578 822.125 C 333.892 834.026 335.209 845.928 336.529 857.829 C 337.314 864.911 338.099 871.994 338.881 879.077 C 339.613 885.711 340.349 892.345 341.087 898.978 C 341.357 901.408 341.626 903.838 341.893 906.268 C 342.259 909.587 342.629 912.907 342.999 916.225 C 343.205 918.08 343.41 919.936 343.622 921.847 C 344.225 926.565 345.081 931.256 345.955 935.939 H 433.704 C 437.403 908.191 440.648 880.416 443.751 852.615 L 445.073 840.698 C 445.861 833.561 446.653 826.424 447.444 819.287 L 451.406 783.478 C 452.929 769.687 454.456 755.897 455.987 742.107 L 464.494 665.256 C 464.763 662.825 465.032 660.394 465.302 657.963 L 466.399 648.024 C 466.604 646.17 466.809 644.316 467.02 642.406 C 467.547 637.51 467.546 637.509 467.453 631.703 H 516.39 C 516.428 621.549 516.464 611.396 516.494 601.243 L 516.566 570.781 C 516.575 566.066 516.585 561.351 516.598 556.635 L 516.646 542.489 C 516.664 537.934 516.677 533.379 516.688 528.823 L 516.712 515.156 C 516.718 511.69 516.729 508.224 516.747 504.759 C 517.072 461.056 507.6 441.879 496.413 425.413 C 485.227 408.947 445.727 405.555 417.535 405.532 L 410.935 405.517 L 397.163 405.506 Z M 348.099 254.567 C 371.372 233.192 404.605 235.609 427.321 251.055 L 428.395 251.8 L 429.601 252.654 L 430.643 253.704 C 446.322 269.505 449.515 286.988 449.47 304.327 V 304.442 L 449.468 304.558 C 449.452 305.539 449.445 306.03 449.437 306.516 C 449.43 306.976 449.422 307.432 449.408 308.296 C 449.404 318.147 449.918 333.454 441.855 347.849 L 441.576 348.346 L 441.263 348.825 C 432.537 362.215 421.283 371.07 404.353 375.58 L 402.97 375.949 L 401.545 376.082 C 383.434 377.775 367.574 375.531 351.266 363.974 L 350.059 363.12 L 349.018 362.07 C 333.338 346.269 330.146 328.786 330.19 311.447 V 311.332 L 330.192 311.216 C 330.213 309.923 330.232 308.688 330.252 307.433 C 330.274 290.887 331.635 273.16 346.821 255.873 L 347.425 255.186 L 348.099 254.567 Z M 418.517 265.715 C 400.855 253.2 376.11 252.013 359.653 267.128 C 348.641 279.663 347.34 292.127 347.327 307.604 L 347.265 311.491 C 347.227 326.357 349.841 338.668 361.143 350.058 C 373.239 358.63 384.688 360.522 399.954 359.095 C 412.579 355.732 420.398 349.576 426.954 339.515 C 431.802 330.861 432.278 321.637 432.327 312.217 L 432.333 308.17 C 432.364 306.246 432.364 306.246 432.396 304.283 C 432.434 289.417 429.819 277.105 418.517 265.715 Z"
                          fill="currentColor"
                        />
                      </svg>
                    )}
                  </>
                )}
              </div>
            </div>
            {/* step 2에서 건너뛰기 버튼 */}
            {step === 2 && (
              <button className="tutorial-skip-btn" onClick={handleSkip}>
                건너뛰기
                <span className="skip-tooltip">건너뛰면 게임 점수가 정확하지 않을 수 있어요</span>
              </button>
            )}
          </div>
        )}

        {/* 텍스트 영역 */}
        <div
          className={`tutorial-title ${
            isStep1 ? 'tutorial-title--center' : 'tutorial-title--side'
          }`}
        >
          {error ? (
            <>
              카메라 연결에 문제가 있습니다.
              <br />
              담당자에게 알려 주세요.
            </>
          ) : (
            renderTitle()
          )}
        </div>
      </div>
    </div>
  );
}

export default TutorialPage;
