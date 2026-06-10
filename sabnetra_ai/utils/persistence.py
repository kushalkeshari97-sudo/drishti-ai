import os
import pickle
import numpy as np
from typing import List, Optional
from sabnetra_ai.core.matcher import SuspectProfile

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


def _get_key_path(path: str) -> str:
    return os.path.join(os.path.dirname(path.rstrip("/\\")) if "/" in path or "\\" in path else ".", ".suspect_key")


def _load_or_create_key(path: str) -> bytes:
    key_path = _get_key_path(path)
    if os.path.exists(key_path):
        with open(key_path, "rb") as f:
            return f.read()
    key = Fernet.generate_key()
    os.makedirs(os.path.dirname(key_path) or ".", exist_ok=True)
    with open(key_path, "wb") as f:
        f.write(key)
    return key


def save_suspects(suspects: List[SuspectProfile], path: str = "suspect_embeddings"):
    """Save a list of suspect profiles to disk with optional encryption.
    Args:
        suspects: List of SuspectProfile objects to persist.
        path: Directory path to save suspect files.
    """
    os.makedirs(path, exist_ok=True)
    cipher = Fernet(_load_or_create_key(path)) if HAS_CRYPTO else None
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
        raw = pickle.dumps(data)
        if cipher:
            raw = cipher.encrypt(raw)
        with open(fpath, "wb") as f:
            f.write(raw)


def load_suspects(path: str = "suspect_embeddings") -> List[SuspectProfile]:
    """Load suspect profiles from disk with optional decryption.
    Args:
        path: Directory path containing suspect files.
    Returns:
        List of SuspectProfile objects.
    """
    if not os.path.isdir(path):
        return []
    cipher = Fernet(_load_or_create_key(path)) if HAS_CRYPTO else None
    suspects = []
    for fname in os.listdir(path):
        if not fname.endswith(".pkl"):
            continue
        fpath = os.path.join(path, fname)
        with open(fpath, "rb") as f:
            raw = f.read()
        if cipher:
            try:
                raw = cipher.decrypt(raw)
            except Exception:
                continue
        data = pickle.loads(raw)
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
