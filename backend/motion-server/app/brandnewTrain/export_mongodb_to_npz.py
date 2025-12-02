"""
MongoDB pose_training_data 컬렉션에서 데이터를 export하여
train_gcn_cnn.py에서 사용할 수 있는 .npz 파일로 변환하는 스크립트.

사용 예시:
    python export_mongodb_to_npz.py --output_dir ./pose_sequences_mongodb
    python export_mongodb_to_npz.py --output_dir ./pose_sequences_mongodb --mongo_uri "mongodb://localhost:27017"
    python export_mongodb_to_npz.py --output_dir ./pose_sequences_mongodb --min_judgment 1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pymongo import MongoClient


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

# 학습하지 않을 actionCode
SKIP_ACTION_CODES = {3, 8}  # 엉덩이 박수, 팔 모으기


@dataclass
class ExportResult:
    mongo_id: str
    action: str
    saved_path: Path
    judgment: Optional[int]


def export_mongodb_to_npz(
    output_dir: Path,
    mongo_uri: str = "mongodb://localhost:27017",
    db_name: str = "heungbuja",
    collection_name: str = "pose_training_data",
    min_judgment: Optional[int] = None,
    max_judgment: Optional[int] = None,
    actions: Optional[List[str]] = None,
    frames_per_sample: int = 8,
    person_label: str = "GAME",  # 게임 데이터 구분용
    overwrite: bool = False,
) -> List[ExportResult]:
    """
    MongoDB에서 pose_training_data를 읽어 npz 파일로 저장합니다.

    Args:
        output_dir: npz 파일을 저장할 디렉토리
        mongo_uri: MongoDB 연결 URI
        db_name: 데이터베이스 이름
        collection_name: 컬렉션 이름
        min_judgment: 최소 judgment 값 (이상)
        max_judgment: 최대 judgment 값 (이하)
        actions: 특정 동작만 export (예: ["CLAP", "ELBOW"])
        frames_per_sample: 시퀀스 프레임 수
        person_label: 데이터 출처 레이블 (폴더명)
        overwrite: 기존 파일 덮어쓰기

    Returns:
        ExportResult 리스트
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # MongoDB 연결
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # 쿼리 조건 구성
    query: Dict = {}

    # actionCode 필터
    valid_codes = list(ACTION_CODE_TO_FOLDER.keys())
    if actions:
        # 지정된 동작의 actionCode만 포함
        action_set = {a.upper() for a in actions}
        valid_codes = [
            code for code, folder in ACTION_CODE_TO_FOLDER.items()
            if folder in action_set
        ]
    query["actionCode"] = {"$in": valid_codes}

    # judgment 필터
    if min_judgment is not None or max_judgment is not None:
        judgment_filter = {}
        if min_judgment is not None:
            judgment_filter["$gte"] = min_judgment
        if max_judgment is not None:
            judgment_filter["$lte"] = max_judgment
        query["judgment"] = judgment_filter

    # 데이터 조회
    print(f"\n{'='*70}")
    print(f"🔍 MongoDB에서 pose_training_data 조회 중...")
    print(f"{'='*70}")
    print(f"MongoDB URI: {mongo_uri}")
    print(f"Database: {db_name}")
    print(f"Collection: {collection_name}")
    print(f"Query: {query}")
    print(f"{'='*70}\n")

    cursor = collection.find(query)
    documents = list(cursor)
    print(f"📊 총 {len(documents)}개 문서 조회됨")

    if not documents:
        print("⚠️  조회된 데이터가 없습니다.")
        return []

    # 동작별 시퀀스 카운터
    sequence_counters: Dict[str, int] = defaultdict(int)
    results: List[ExportResult] = []
    skipped = 0

    for doc in documents:
        try:
            action_code = doc.get("actionCode")
            if action_code not in ACTION_CODE_TO_FOLDER:
                print(f"⚠️  알 수 없는 actionCode: {action_code}, 건너뜀")
                skipped += 1
                continue

            action_folder = ACTION_CODE_TO_FOLDER[action_code]
            pose_frames = doc.get("poseFrames", [])

            # 프레임 수 검증
            if len(pose_frames) != frames_per_sample:
                print(
                    f"⚠️  프레임 수 불일치: {len(pose_frames)} (기대: {frames_per_sample}), "
                    f"ID: {doc.get('_id')}, 건너뜀"
                )
                skipped += 1
                continue

            # numpy 배열로 변환 (8, 33, 2)
            try:
                landmarks = np.array(pose_frames, dtype=np.float32)
                if landmarks.shape != (frames_per_sample, 33, 2):
                    print(
                        f"⚠️  Shape 불일치: {landmarks.shape} (기대: ({frames_per_sample}, 33, 2)), "
                        f"ID: {doc.get('_id')}, 건너뜀"
                    )
                    skipped += 1
                    continue
            except Exception as e:
                print(f"⚠️  numpy 변환 실패: {e}, ID: {doc.get('_id')}, 건너뜀")
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
                print(f"⚠️  파일 이미 존재 (덮어쓰기 비활성화): {output_path}")
                skipped += 1
                continue

            # 메타데이터 구성
            metadata = {
                "person": person_label,
                "action": action_folder,
                "sequence_id": seq_id,
                "frames_per_sample": frames_per_sample,
                "landmark_count": 33,
                "source": "mongodb",
                "mongo_id": str(doc.get("_id")),
                "session_id": doc.get("sessionId"),
                "user_id": doc.get("userId"),
                "song_id": doc.get("songId"),
                "judgment": doc.get("judgment"),
                "target_probability": doc.get("targetProbability"),
                "verse": doc.get("verse"),
            }

            # npz 저장
            np.savez_compressed(output_path, landmarks=landmarks, metadata=json.dumps(metadata))

            results.append(
                ExportResult(
                    mongo_id=str(doc.get("_id")),
                    action=action_folder,
                    saved_path=output_path,
                    judgment=doc.get("judgment"),
                )
            )

            # 진행 상황 표시
            if len(results) % 100 == 0:
                print(f"  ✓ {len(results)}개 시퀀스 저장 완료...")

        except Exception as e:
            print(f"⚠️  예외 발생: {e}, ID: {doc.get('_id')}, 건너뜀")
            skipped += 1
            continue

    # 연결 종료
    client.close()

    # 요약 출력
    if results:
        summary = defaultdict(int)
        judgment_summary = defaultdict(lambda: defaultdict(int))

        for result in results:
            summary[result.action] += 1
            if result.judgment is not None:
                judgment_summary[result.action][result.judgment] += 1

        print(f"\n{'='*70}")
        print("📊 Export 요약")
        print(f"{'='*70}")
        print(f"총 export: {len(results)}개")
        print(f"건너뜀: {skipped}개")
        print(f"\n동작별 분포:")
        for action, count in sorted(summary.items()):
            print(f"  - {action}: {count}개")
            if action in judgment_summary:
                for judgment, jcount in sorted(judgment_summary[action].items()):
                    print(f"      judgment={judgment}: {jcount}개")
        print(f"\n출력 폴더: {output_dir}")
        print(f"{'='*70}\n")
    else:
        print(f"\n⚠️  저장된 시퀀스가 없습니다. (건너뜀: {skipped}개)")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MongoDB pose_training_data를 npz 파일로 export",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pose_sequences_mongodb",
        help="npz 파일 출력 디렉토리 (기본: ./pose_sequences_mongodb)",
    )
    parser.add_argument(
        "--mongo_uri",
        type=str,
        default="mongodb://localhost:27017",
        help="MongoDB 연결 URI",
    )
    parser.add_argument(
        "--db_name",
        type=str,
        default="heungbuja",
        help="MongoDB 데이터베이스 이름 (기본: heungbuja)",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="pose_training_data",
        help="MongoDB 컬렉션 이름 (기본: pose_training_data)",
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
        help="특정 동작만 export (예: CLAP ELBOW STRETCH)",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    export_mongodb_to_npz(
        output_dir=Path(args.output_dir),
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        collection_name=args.collection_name,
        min_judgment=args.min_judgment,
        max_judgment=args.max_judgment,
        actions=args.actions,
        frames_per_sample=args.frames_per_sample,
        person_label=args.person_label,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
