"""API routes for managing Agent Voiceprint biometrics and acoustic profiles."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from asr_pro.api.deps import get_db
from asr_pro.db.models import AgentVoiceprint
from asr_pro.services.biometric_service import BiometricService

logger = logging.getLogger("asr_pro.api.routes.agents")
router = APIRouter(prefix="/agents", tags=["agents"])


class VoiceprintResponse(BaseModel):
    id: str
    agent_code: str
    agent_name: str
    created_at: Any
    embedding_dim: int = 128

    class Config:
        from_attributes = True


class EnrollVoiceprintRequest(BaseModel):
    agent_code: str = Field(..., description="Unique agent identifier code (e.g. AG-1001)")
    agent_name: str = Field(..., description="Full name of the contact center agent")


@router.get("/voiceprints", response_model=list[VoiceprintResponse])
def list_voiceprints(db: Session = Depends(get_db)) -> list[Any]:
    """List all enrolled agent acoustic voiceprints."""
    service = BiometricService(db_session=db)
    voiceprints = service.list_voiceprints()
    return voiceprints


@router.post("/voiceprints/enroll", status_code=status.HTTP_201_CREATED)
async def enroll_agent_voiceprint(
    agent_code: str,
    agent_name: str,
    file: UploadFile,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Enroll a new agent acoustic voiceprint from a reference audio file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Ses dosyası gereklidir.")
    
    import shutil
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        service = BiometricService(db_session=db)
        success = service.enroll_agent(
            agent_code=agent_code,
            agent_name=agent_name,
            audio_path_or_array=temp_path,
        )
        if not success:
            raise HTTPException(status_code=400, detail="Ses izi çıkarılamadı veya temsilci zaten mevcut.")
        
        return {
            "status": "success",
            "message": f"Temsilci '{agent_name}' ({agent_code}) ses izi başarıyla kaydedildi.",
            "embedding_dim": 128,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
