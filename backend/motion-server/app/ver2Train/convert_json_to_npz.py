"""
JSON으로 export된 MongoDB pose_training_data를
train_gcn_cnn.py에서 사용할 수 있는 .npz 파일로 변환하는 스크립트.

사용 예시:
    python convert_json_to_npz.py
    python convert_json_to_npz.py --min_judgment 1
    python convert_json_to_npz.py --output_dir ./pose_sequences
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# actionCode → 동작 폴더명 매핑
ACTION_CODE_TO_FOLDER = {
    1: "CLAP",       # 손 박수
    2: "ELBOW",      # 팔 치기
    4: "STRETCH",    # 팔 뻗기
    5: "TILT",       # 기우뚱
    6: "EXIT",       # 비상구
    7: "UNDERARM",   # 겨드랑이박수
    9: "STAY",       # 가만히 있음
}


@dataclass
class ConvertResult:
    mongo_id: str
    action: str
    saved_path: Path
    judgment: Optional[int]


def adjust_frames(frames: List, target_frames: int) -> Optional[List]:
    """
    프레임 수를 target_frames로 균등 샘플링하여 조정합니다.
    - numpy linspace를 사용하여 균등한 간격으로 프레임 선택
    - 예: 12프레임 -> 8프레임: indices [0, 1, 3, 5, 6, 8, 10, 11] 선택
    """
    current = len(frames)
    if current == target_frames:
        return frames

    if current < target_frames:
        # 프레임이 부족하면 None 반환 (스킵)
        return None

    # 균등 샘플링: linspace로 균등한 인덱스 선택
    indices = np.linspace(0, current - 1, target_frames, dtype=int)
    return [frames[i] for i in indices]


def convert_json_to_npz(
    json_path: Path,
    output_dir: Path,
    min_judgment: Optional[int] = None,
    max_judgment: Optional[int] = None,
    actions: Optional[List[str]] = None,
    frames_per_sample: int = 8,
    person_label: str = "GAME",
    overwrite: bool = False,
    adjust_frame_count: bool = True,
) -> List[ConvertResult]:
    """
    JSON 파일에서 pose_training_data를 읽어 npz 파일로 저장합니다.

    Args:
        json_path: MongoDB에서 export한 JSON 파일 경로
        output_dir: npz 파일을 저장할 디렉토리
        min_judgment: 최소 judgment 값 (이상)
        max_judgment: 최대 judgment 값 (이하)
        actions: 특정 동작만 export (예: ["CLAP", "ELBOW"])
        frames_per_sample: 시퀀스 프레임 수
        person_label: 데이터 출처 레이블 (폴더명)
        overwrite: 기존 파일 덮어쓰기
        adjust_frame_count: 프레임 수 자동 조정 (True면 잘라내거나 패딩)

    Returns:
        ConvertResult 리스트
    """
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON 파일 읽기
    print(f"\n{'='*70}")
    print(f"[*] JSON loading...")
    print(f"{'='*70}")
    print(f"Input: {json_path}")
    print(f"Output: {output_dir}")

    with open(json_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"[*] Total {len(documents)} documents loaded")
    print(f"{'='*70}\n")

    if not documents:
        print("[!] No data found.")
        return []

    # 필터 설정
    action_filter = None
    if actions:
        action_filter = {a.upper() for a in actions}
        valid_codes = [
            code for code, folder in ACTION_CODE_TO_FOLDER.items()
            if folder in action_filter
        ]
    else:
        valid_codes = list(ACTION_CODE_TO_FOLDER.keys())

    # 동작별 시퀀스 카운터
    sequence_counters: Dict[str, int] = defaultdict(int)
    results: List[ConvertResult] = []
    skipped = 0
    skip_reasons = defaultdict(int)

    for doc in documents:
        try:
            action_code = doc.get("actionCode")
            if action_code not in ACTION_CODE_TO_FOLDER:
                skip_reasons["unknown_action_code"] += 1
                skipped += 1
                continue

            if action_code not in valid_codes:
                skip_reasons["filtered_action"] += 1
                skipped += 1
                continue

            action_folder = ACTION_CODE_TO_FOLDER[action_code]
            pose_frames = doc.get("poseFrames", [])

            # judgment 필터
            judgment = doc.get("judgment")
            if min_judgment is not None and (judgment is None or judgment < min_judgment):
                skip_reasons["judgment_too_low"] += 1
                skipped += 1
                continue
            if max_judgment is not None and (judgment is None or judgment > max_judgment):
                skip_reasons["judgment_too_high"] += 1
                skipped += 1
                continue

            # 프레임 수 조정 또는 검증
            original_frame_count = len(pose_frames)
            if original_frame_count != frames_per_sample:
                if adjust_frame_count and original_frame_count >= frames_per_sample:
                    # 균등 샘플링으로 프레임 수 조정
                    pose_frames = adjust_frames(pose_frames, frames_per_sample)
                    if pose_frames is None:
                        skip_reasons["frame_adjust_failed"] += 1
                        skipped += 1
                        continue
                else:
                    skip_reasons["frame_count_too_few"] += 1
                    skipped += 1
                    continue

            # numpy 배열로 변환
            # 기존 학습 데이터 형식: (8, 22, 3) - 얼굴 제외 22 landmarks, xyz
            # 게임 데이터 원본: (8, 33, 2) - 33 landmarks, xy
            # 변환: 얼굴(0~10) 제외하고 z=0 추가하여 (8, 22, 3)으로 맞춤
            USED_LANDMARK_INDICES = list(range(11, 33))  # 22개
            try:
                landmarks_full = np.array(pose_frames, dtype=np.float32)  # (8, 33, 2)
                if landmarks_full.shape[1] != 33 or landmarks_full.shape[2] != 2:
                    skip_reasons["shape_mismatch"] += 1
                    skipped += 1
                    continue

                # 얼굴 제외 (11~32번 인덱스만 사용)
                landmarks_body = landmarks_full[:, USED_LANDMARK_INDICES, :]  # (8, 22, 2)

                # z=0 추가하여 (8, 22, 3)으로 변환
                z_zeros = np.zeros((frames_per_sample, 22, 1), dtype=np.float32)
                landmarks = np.concatenate([landmarks_body, z_zeros], axis=2)  # (8, 22, 3)

            except Exception as e:
                skip_reasons["numpy_conversion_error"] += 1
                skipped += 1
                continue

            # 시퀀스 ID 할당
            sequence_counters[action_folder] += 1
            seq_id = sequence_counters[action_folder]

            # 출력 경로 생성
            action_output_dir = output_dir / person_label / action_folder
            action_output_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{action_folder.lower()}_seq{seq_id:03d}.npz"
            output_path = action_output_dir / filename

            if output_path.exists() and not overwrite:
                skip_reasons["file_exists"] += 1
                skipped += 1
                continue

            # MongoDB ID 추출 (JSON export 형식 처리)
            mongo_id = doc.get("_id")
            if isinstance(mongo_id, dict):
                mongo_id = mongo_id.get("$oid", str(mongo_id))
            else:
                mongo_id = str(mongo_id)

            # userId, songId 추출 (JSON export 형식 처리)
            user_id = doc.get("userId")
            if isinstance(user_id, dict):
                user_id = user_id.get("$numberLong", user_id)

            song_id = doc.get("songId")
            if isinstance(song_id, dict):
                song_id = song_id.get("$numberLong", song_id)

            # 메타데이터 구성
            metadata = {
                "person": person_label,
                "action": action_folder,
                "sequence_id": seq_id,
                "frames_per_sample": frames_per_sample,
                "landmark_count": 33,
                "source": "mongodb_json",
                "mongo_id": mongo_id,
                "session_id": doc.get("sessionId"),
                "user_id": user_id,
                "song_id": song_id,
                "judgment": judgment,
                "target_probability": doc.get("targetProbability"),
                "verse": doc.get("verse"),
                "action_name": doc.get("actionName"),
                "original_frame_count": original_frame_count,
            }

            # npz 저장
            np.savez_compressed(output_path, landmarks=landmarks, metadata=json.dumps(metadata))

            results.append(
                ConvertResult(
                    mongo_id=mongo_id,
                    action=action_folder,
                    saved_path=output_path,
                    judgment=judgment,
                )
            )

            # 진행 상황 표시
            if len(results) % 100 == 0:
                print(f"  [+] {len(results)} sequences saved...")

        except Exception as e:
            print(f"[!] Exception: {e}")
            skip_reasons["exception"] += 1
            skipped += 1
            continue

    # 요약 출력
    if results:
        summary = defaultdict(int)
        judgment_summary = defaultdict(lambda: defaultdict(int))

        for result in results:
            summary[result.action] += 1
            if result.judgment is not None:
                judgment_summary[result.action][result.judgment] += 1

        print(f"\n{'='*70}")
        print("[*] Conversion Summary")
        print(f"{'='*70}")
        print(f"Total converted: {len(results)}")
        print(f"Skipped: {skipped}")

        if skip_reasons:
            print(f"\nSkip reasons:")
            for reason, count in sorted(skip_reasons.items()):
                print(f"  - {reason}: {count}")

        print(f"\nAction distribution:")
        for action, count in sorted(summary.items()):
            print(f"  - {action}: {count}")
            if action in judgment_summary:
                for j, jcount in sorted(judgment_summary[action].items()):
                    judgment_label = {0: "MISS", 1: "BAD", 2: "GOOD", 3: "PERFECT"}.get(j, f"J{j}")
                    print(f"      {judgment_label}: {jcount}")

        print(f"\nOutput dir: {output_dir}")
        print(f"{'='*70}\n")
    else:
        print(f"\n[!] No sequences saved. (Skipped: {skipped})")
        if skip_reasons:
            print(f"Skip reasons:")
            for reason, count in sorted(skip_reasons.items()):
                print(f"  - {reason}: {count}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JSON으로 export된 MongoDB 데이터를 npz 파일로 변환",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="heungbudb.pose_training_data.json",
        help="입력 JSON 파일 경로 (기본: heungbudb.pose_training_data.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pose_sequences",
        help="npz 파일 출력 디렉토리 (기본: ./pose_sequences)",
    )
    parser.add_argument(
        "--min_judgment",
        type=int,
        default=None,
        help="최소 judgment 값 (이상). 예: 1 (BAD 이상만 사용)",
    )
    parser.add_argument(
        "--max_judgment",
        type=int,
        default=None,
        help="최대 judgment 값 (이하). 예: 3 (PERFECT 이하만 사용)",
    )
    parser.add_argument(
        "--actions",
        nargs="*",
        default=None,
        help="특정 동작만 변환 (예: CLAP ELBOW STRETCH)",
    )
    parser.add_argument(
        "--frames_per_sample",
        type=int,
        default=8,
        help="시퀀스 프레임 수 (기본: 8)",
    )
    parser.add_argument(
        "--person_label",
        type=str,
        default="GAME",
        help="데이터 출처 레이블 (폴더명, 기본: GAME)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 파일 덮어쓰기",
    )
    parser.add_argument(
        "--no_adjust",
        action="store_true",
        help="프레임 수 자동 조정 비활성화 (기본: 활성화)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    convert_json_to_npz(
        json_path=Path(args.json_path),
        output_dir=Path(args.output_dir),
        min_judgment=args.min_judgment,
        max_judgment=args.max_judgment,
        actions=args.actions,
        frames_per_sample=args.frames_per_sample,
        person_label=args.person_label,
        overwrite=args.overwrite,
        adjust_frame_count=not args.no_adjust,
    )


if __name__ == "__main__":
    main()
