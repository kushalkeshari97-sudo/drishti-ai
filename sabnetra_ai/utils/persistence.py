import os
import pickle
import numpy as np
from typing import List, Optional
from sabnetra_ai.core.matcher import SuspectProfile


def save_suspects(suspects: List[SuspectProfile], path: str = "suspect_embeddings"):
    os.makedirs(path, exist_ok=True)
    for s in suspects:
        fpath = os.path.join(path, f"{s.suspect_id}.pkl")
        data = {
            "suspect_id": s.suspect_id,
            "case_id": s.case_id,
            "face_emb": s.face_embedding,
            "body_emb": s.body_embedding,
            "clothing_emb": s.clothing_descriptor,
            "gait_emb": s.gait_descriptor,
            "metadata": s.metadata,
        }
        with open(fpath, "wb") as f:
            pickle.dump(data, f)


def load_suspects(path: str = "suspect_embeddings") -> List[SuspectProfile]:
    if not os.path.isdir(path):
        return []
    suspects = []
    for fname in os.listdir(path):
        if not fname.endswith(".pkl"):
            continue
        fpath = os.path.join(path, fname)
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        profile = SuspectProfile(
            suspect_id=data["suspect_id"],
            case_id=data.get("case_id", ""),
            face_emb=data.get("face_emb"),
            body_emb=data.get("body_emb"),
            clothing_emb=data.get("clothing_emb"),
            gait_emb=data.get("gait_emb"),
            metadata=data.get("metadata", {}),
        )
        suspects.append(profile)
    return suspects
