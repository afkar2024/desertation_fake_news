from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(prefix="/feedback", tags=["feedback"])

FEEDBACK_DIR = Path("processed_data")
FEEDBACK_FILE = FEEDBACK_DIR / "user_feedback.json"


class FeedbackItem(BaseModel):
    id: str = Field(default_factory=lambda: datetime.now().strftime("fb_%Y%m%d%H%M%S%f"))
    text: str
    model_prediction: Optional[int] = None  # 0 real, 1 fake
    model_confidence: Optional[float] = None
    user_label: Optional[int] = None  # 0 real, 1 fake
    is_correct: Optional[bool] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


def _load_all() -> List[Dict[str, Any]]:
    if FEEDBACK_FILE.exists():
        try:
            import json

            return json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_all(items: List[Dict[str, Any]]) -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    import json

    FEEDBACK_FILE.write_text(json.dumps(items, indent=2, default=str), encoding="utf-8")


@router.get("/", response_model=List[FeedbackItem])
async def list_feedback() -> List[FeedbackItem]:
    return [FeedbackItem(**it) for it in _load_all()]


@router.post("/", response_model=FeedbackItem)
async def create_feedback(item: FeedbackItem) -> FeedbackItem:
    data = _load_all()
    data.append(item.dict())
    _save_all(data)
    return item


@router.delete("/{item_id}")
async def delete_feedback(item_id: str) -> Dict[str, Any]:
    data = _load_all()
    new_data = [it for it in data if it.get("id") != item_id]
    if len(new_data) == len(data):
        raise HTTPException(status_code=404, detail="Feedback not found")
    _save_all(new_data)
    return {"deleted": item_id}


@router.post("/export")
async def export_feedback() -> Dict[str, Any]:
    """Export feedback to CSV for future training."""
    import csv

    items = _load_all()
    csv_path = FEEDBACK_DIR / "user_feedback.csv"
    fieldnames = [
        "id",
        "text",
        "model_prediction",
        "model_confidence",
        "user_label",
        "is_correct",
        "notes",
        "created_at",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for it in items:
            writer.writerow({k: it.get(k) for k in fieldnames})
    return {"csv_path": str(csv_path.resolve()), "count": len(items)}


