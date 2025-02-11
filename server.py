import os
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Set
import asyncio
import uvicorn
import logging
from src.utils.default_config_settings import default_config
from src.utils.utils import update_model_dropdown, get_latest_files, capture_screenshot
from datetime import datetime
import json

# Import the required functions from webui
from webui import (
    run_browser_agent,
    run_deep_search,
    stop_research_agent,
    stop_agent
)

# Initialize FastAPI app
app = FastAPI(title="Shoperator Agent API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables (same as in webui.py)
_global_browser = None
_global_browser_context = None
_global_agent_state = None

# Add these after the existing global variables
_current_agent_state = {
    "is_running": False,
    "current_step": 0,
    "max_steps": 0,
    "screenshot": None,
    "memory": "",
    "task_progress": "",
    "future_plans": "",
    "last_update": None,
    "task": None,
    "errors": None
}

# Initialize logging
logger = logging.getLogger(__name__)

# Add these imports at the top with the other imports
from fastapi import WebSocket
from typing import Dict, Set

# Add after the other global variables
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = set()
        self.active_connections[client_id].add(websocket)

    async def disconnect(self, websocket: WebSocket, client_id: str):
        if client_id in self.active_connections:
            self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]

    async def broadcast_to_client(self, client_id: str, message: str):
        if client_id in self.active_connections:
            dead_connections = set()
            for connection in self.active_connections[client_id]:
                try:
                    await connection.send_text(message)
                except:
                    dead_connections.add(connection)
            
            # Clean up dead connections
            for dead_connection in dead_connections:
                self.active_connections[client_id].remove(dead_connection)

# Add after FastAPI initialization
manager = ConnectionManager()

# Pydantic models for request validation
class AgentConfig(BaseModel):
    agent_type: str = "custom"
    llm_provider: str = "openai"
    llm_model_name: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_base_url: Optional[str] = "https://api.openai.com/v1"
    llm_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    use_own_browser: bool = False
    keep_browser_open: bool = False
    headless: bool = False
    disable_security: bool = False
    window_w: int = 1280
    window_h: int = 720
    save_recording_path: Optional[str] = "./tmp/record_videos"
    save_agent_history_path: Optional[str] = "./tmp/agent_history"
    save_trace_path: Optional[str] = "./tmp/traces"
    enable_recording: bool = True
    task: str  # This is the only required field without a default
    add_infos: Optional[str] = None
    max_steps: int = 20
    use_vision: bool = True
    max_actions_per_step: int = 3
    tool_calling_method: str = "function_call"

class DeepResearchConfig(BaseModel):
    research_task: str
    max_search_iteration: int
    max_query_per_iter: int
    llm_provider: str
    llm_model_name: str
    llm_temperature: float
    llm_base_url: Optional[str]
    llm_api_key: Optional[str]
    use_vision: bool
    use_own_browser: bool
    headless: bool

# API endpoints
@app.get("/config/default")
async def get_default_config():
    """Get default configuration"""
    return default_config()

@app.post("/agent/run")
async def run_agent(config: AgentConfig, background_tasks: BackgroundTasks):
    """Run browser agent with given configuration"""
    global _current_agent_state
    
    try:
        # Create a unique client_id for this session
        client_id = str(datetime.now().timestamp())
        
        # Create history directory if it doesn't exist
        if config.save_agent_history_path:
            os.makedirs(config.save_agent_history_path, exist_ok=True)
            
        # Initialize agent state
        _current_agent_state.update({
            "is_running": True,
            "current_step": 0,
            "max_steps": config.max_steps,
            "screenshot": None,
            "memory": "",
            "task_progress": "",
            "future_plans": "",
            "last_update": datetime.now().isoformat(),
            "task": config.task,
            "errors": None
        })

        # Run agent in background task to not block
        background_tasks.add_task(
            run_agent_with_status_updates,
            config,
            client_id
        )
        
        return {"message": "Agent started successfully", "client_id": client_id}
    except Exception as e:
        _current_agent_state["is_running"] = False
        _current_agent_state["errors"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

async def run_agent_with_status_updates(config: AgentConfig, client_id: str):
    """Run the agent while updating status"""
    global _current_agent_state, _global_browser_context
    
    try:
        # Create a websocket callback that handles both screenshots and status updates
        async def websocket_callback(message: str):
            try:
                # Parse the message to handle different types
                message_data = json.loads(message)
                if message_data.get("type") == "browser_screenshot":
                    # Add timestamp and client info to screenshot message
                    message_data["timestamp"] = datetime.now().isoformat()
                    message_data["client_id"] = client_id
                
                # Broadcast the message
                await manager.broadcast_to_client(client_id, json.dumps(message_data))
                logger.debug(f"Websocket message sent: {message_data['type']}")
            except Exception as e:
                logger.error(f"Error in websocket callback: {e}")

        # Create a custom callback that includes WebSocket streaming
        async def status_callback(step_number: int, memory: str, task_progress: str, future_plans: str, trace_file: Optional[str] = None, history_file: Optional[str] = None):
            await update_agent_status(step_number, memory, task_progress, future_plans, trace_file, history_file)
            
            # Prepare the message for streaming
            message = {
                "type": "agent_update",
                "step": step_number,
                "memory": str(memory) if memory is not None else "",
                "task_progress": str(task_progress) if task_progress is not None else "",
                "future_plans": str(future_plans) if future_plans is not None else "",
                "timestamp": datetime.now().isoformat()
            }
            
            # Broadcast to the specific client
            await manager.broadcast_to_client(client_id, json.dumps(message))

        # Use both callbacks in run_browser_agent
        result = await run_browser_agent(
            agent_type=config.agent_type,
            llm_provider=config.llm_provider,
            llm_model_name=config.llm_model_name,
            llm_temperature=config.llm_temperature,
            llm_base_url=config.llm_base_url,
            llm_api_key=config.llm_api_key,
            use_own_browser=config.use_own_browser,
            keep_browser_open=config.keep_browser_open,
            headless=config.headless,
            disable_security=config.disable_security,
            window_w=config.window_w,
            window_h=config.window_h,
            save_recording_path=config.save_recording_path,
            save_agent_history_path=config.save_agent_history_path,
            save_trace_path=config.save_trace_path,
            enable_recording=config.enable_recording,
            task=config.task,
            add_infos=config.add_infos,
            max_steps=config.max_steps,
            use_vision=config.use_vision,
            max_actions_per_step=config.max_actions_per_step,
            tool_calling_method=config.tool_calling_method,
            status_callback=status_callback if config.agent_type == "custom" else None,
            websocket_callback=websocket_callback
        )
        
        # Get the browser context from the result
        if isinstance(result, tuple) and len(result) >= 9:
            _, _, _, _, _, _, _, _, _, browser_context = result
            _global_browser_context = browser_context
        
        # Update final state
        _current_agent_state.update({
            "is_running": False,
            "current_step": config.max_steps,
            "task_progress": "Completed",
            "last_update": datetime.now().isoformat()
        })
        
        return result
    except Exception as e:
        error_message = {
            "type": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        await manager.broadcast_to_client(client_id, json.dumps(error_message))
        raise e

async def update_agent_status(step_number: int, memory: str, task_progress: str, future_plans: str, trace_file: Optional[str] = None, history_file: Optional[str] = None):
    """Callback to update agent status"""
    global _current_agent_state
    
    _current_agent_state.update({
        "current_step": step_number,
        "memory": memory,
        "task_progress": task_progress,
        "future_plans": future_plans,
        "trace_file": trace_file,
        "history_file": history_file,
        "last_update": datetime.now().isoformat()
    })

@app.post("/agent/stop")
async def stop_agent():
    """Stop the running agent"""
    global _current_agent_state
    
    try:
        result = await stop_agent()
        _current_agent_state.update({
            "is_running": False,
            "task_progress": "Stopped by user",
            "last_update": datetime.now().isoformat()
        })
        return {"message": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research/run")
async def run_research(config: DeepResearchConfig):
    """Run deep research with given configuration"""
    try:
        result = await run_deep_search(
            research_task=config.research_task,
            max_search_iteration_input=config.max_search_iteration,
            max_query_per_iter_input=config.max_query_per_iter,
            llm_provider=config.llm_provider,
            llm_model_name=config.llm_model_name,
            llm_temperature=config.llm_temperature,
            llm_base_url=config.llm_base_url,
            llm_api_key=config.llm_api_key,
            use_vision=config.use_vision,
            use_own_browser=config.use_own_browser,
            headless=config.headless
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research/stop")
async def stop_research():
    """Stop the running research"""
    try:
        result = await stop_research_agent()
        return {"message": result[0]}  # Convert tuple result to dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{provider}")
async def get_models(provider: str):
    """Get available models for a provider"""
    try:
        return update_model_dropdown(provider)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recordings")
async def get_recordings(recording_path: str):
    """Get list of available recordings"""
    try:
        recordings = get_latest_files(recording_path)
        return recordings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/status")
async def get_agent_status():
    """Get current agent status including screenshot and step information"""
    global _current_agent_state, _global_browser_context
    
    try:
        # Add debug logging
        logger.debug(f"Agent status requested. Is running: {_current_agent_state['is_running']}")
        logger.debug(f"Browser context exists: {_global_browser_context is not None}")

        # Only capture new screenshot if agent is running AND browser context exists
        if _current_agent_state["is_running"] and _global_browser_context:
            try:
                screenshot = await capture_screenshot(_global_browser_context)
                if screenshot:
                    _current_agent_state["screenshot"] = screenshot
                    _current_agent_state["last_update"] = datetime.now().isoformat()
            except Exception as screenshot_error:
                logger.error(f"Failed to capture screenshot: {screenshot_error}")
                # Don't fail the entire request if screenshot capture fails
                _current_agent_state["screenshot"] = None

        # Clean up browser context reference if agent is not running
        if not _current_agent_state["is_running"]:
            _global_browser_context = None

        return _current_agent_state
    except Exception as e:
        logger.error(f"Error in get_agent_status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent status: {str(e)}"
        )

# Add this new WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    logger.info(f"Client {client_id} connecting to WebSocket...")
    await manager.connect(websocket, client_id)
    logger.info(f"Client {client_id} connected successfully")
    try:
        while True:
            # Keep the connection alive and wait for client messages if needed
            data = await websocket.receive_text()
    except:
        logger.info(f"Client {client_id} disconnected from WebSocket")
        await manager.disconnect(websocket, client_id)

def main():
    """Run the FastAPI server"""
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=3030,
        reload=True
    )

if __name__ == "__main__":
    main()