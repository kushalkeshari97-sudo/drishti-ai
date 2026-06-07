import logging
import sys
import os

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("logs", "sabnetra.log")),
    ],
)

from sabnetra_ai.config import SabNetraConfig
from sabnetra_ai.pipeline.orchestrator import SabNetraPipeline
from sabnetra_ai.models.model_manager import ModelManager
from sabnetra_ai.enrollment.fir_module import FIRInputModule
from sabnetra_ai.core.matcher import MatchingEngine, SuspectProfile
from sabnetra_ai.core.state_engine import StateClassificationEngine, IdentityState
from sabnetra_ai.core.alert_system import AlertSystem, Alert

__version__ = "2.0.0"
__author__ = "SabNetra AI"
__description__ = "CCTV-first real-time surveillance intelligence system for suspect detection, tracking, and cross-camera identity recognition."


def create_pipeline(config: SabNetraConfig = None) -> SabNetraPipeline:
    return SabNetraPipeline(config or SabNetraConfig())
