from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class AgentRun(BaseModel):
    client_id: str
    task: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, stopped
    steps_completed: int = 0
    max_steps: int
    memory: str = ""
    task_progress: str = ""
    future_plans: str = ""
    errors: Optional[str] = None
    config: Dict[str, Any]
    history_gif_url: Optional[str] = None  # URL for the agent_history.gif on S3
    recording_url: Optional[str] = None    # URL for the video (.webm) recording on S3
    agent_history: Optional[str] = None  # New field for agent history JSON
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ResearchRun(BaseModel):
    research_task: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: str = "running"
    iterations_completed: int = 0
    max_iterations: int
    results: Optional[str] = None
    errors: Optional[str] = None
    config: Dict[str, Any]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 