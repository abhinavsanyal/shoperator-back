from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class AgentRun(BaseModel):
    client_id: str
    agent_id: Optional[str] = None
    clerk_id: str
    task: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, stopped
    steps_completed: int = 0
    max_steps: int
    memory: Optional[str] = None
    task_progress: Optional[str] = None
    future_plans: Optional[str] = None
    errors: Optional[str] = None
    config: Dict[str, Any]
    history_gif_url: Optional[str] = None  # URL for the agent_history.gif on S3
    recording_url: Optional[str] = None    # URL for the video (.webm) recording on S3
    agent_history: Optional[Dict[str, Any]] = None  # Changed from Optional[str] to Optional[Dict[str, Any]]
    chat_history: List[str] = Field(default_factory=list)  # New field to store step-by-step chat logs
    
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

class User(BaseModel):
    clerk_id: str
    email: str
    first_name: Optional[str] = ""
    image_url: Optional[str] = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 