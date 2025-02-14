import os
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, SecretStr
from typing import Optional, List, Dict, Any, Set
import asyncio
import uvicorn
import logging
from src.utils.default_config_settings import default_config
from src.utils.utils import update_model_dropdown, get_latest_files, capture_screenshot
from datetime import datetime
import json

from dotenv import load_dotenv
load_dotenv(override=True)

# Import the required functions from webui
from webui import (
    run_browser_agent,
    run_deep_search,
    stop_research_agent,
    stop_agent
)

# Import the database modules
from src.db.db import Database
from src.db.models import AgentRun, ResearchRun, User
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

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
    clerk_id: str  
    agent_type: str = "custom"
    llm_provider: str = "openai"
    llm_model_name: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_base_url: Optional[str] = "https://api.openai.com/v1"
    llm_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    use_own_browser: bool = False
    keep_browser_open: bool = False
    headless: bool = True
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

# Add after FastAPI initialization but before the routes
@app.on_event("startup")
async def startup_db_client():
    mongodb_url = os.getenv("MONGODB_URI")
    if not mongodb_url:
        raise ValueError("MONGODB_URI environment variable not set")

    await Database.connect_to_database(mongodb_url)

@app.on_event("shutdown")
async def shutdown_db_client():
    await Database.close_database_connection()

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
        # Validate task with guardrail
        validation = await guardrail(config.task)
        if not validation["pass"]:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid task",
                    "detail": validation["comment"]
                }
            )
        
        # Extract dynamic filters
        dynamic_filters = await get_prompt_dynamic_filters(config.task)
        
        # Create a unique client_id for this session
        client_id = str(datetime.now().timestamp())
        
        # Create history directory if it doesn't exist
        if config.save_agent_history_path:
            os.makedirs(config.save_agent_history_path, exist_ok=True)
        
        # Create agent run record with clerk_id from request body
        agent_run = AgentRun(
            client_id=client_id,
            clerk_id=config.clerk_id,  # Use clerk_id from config
            task=config.task,
            max_steps=config.max_steps,
            config=config.model_dump()
        )
        
        # Store in MongoDB
        db = Database.get_database()
        result = await db.agent_runs.insert_one(agent_run.model_dump())
            
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
            client_id,
            agent_run  # Pass the agent_run instance
        )
        
        return {
            "task": config.task,
            "dynamic_filters": dynamic_filters,
            "client_id": client_id,
            "message": "Agent started successfully",
            "status": "running",
            "run_id": str(result.inserted_id) 
        }
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to start agent",
                "detail": str(e)
            }
        )

async def run_agent_with_status_updates(config: AgentConfig, client_id: str, agent_run: AgentRun):
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

        async def status_callback(step_number: int, memory: str, task_progress: str, future_plans: str, client_id: str, trace_file: Optional[str] = None, history_file: Optional[str] = None):
            await update_agent_status(step_number, memory, task_progress, future_plans, trace_file, history_file)
            print(f"Updating agent status: {step_number}, {memory}")
            
            # Send websocket message when agent finishes
            message = {
                "type": "agent_finished",
                "data": "Agent finished its work",
                "timestamp": datetime.now().isoformat()
            }
            await manager.broadcast_to_client(client_id, json.dumps(message))

        # Create a partial function that includes the client_id
        async def status_callback_with_client(step_number: int, memory: str, task_progress: str, future_plans: str, trace_file: Optional[str] = None, history_file: Optional[str] = None):
            await status_callback(step_number, memory, task_progress, future_plans, client_id, trace_file, history_file)
        
        # Run the agent with the websocket callback
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
            status_callback=status_callback_with_client if config.agent_type == "custom" else None,
            websocket_callback=websocket_callback,
            agent_run=agent_run
        )
        
        # Unpack extended results when available (custom agent branch returns extra data)
        if isinstance(result, tuple) and len(result) >= 10:
            (
                _,
                _,
                _,
                _,
                _,
                _,
                browser_context,
                history_gif_url,
                recording_url,
                agent_history
            ) = result
            _global_browser_context = browser_context
            _current_agent_state["history_gif_url"] = history_gif_url
            _current_agent_state["recording_url"] = recording_url
            _current_agent_state["agent_history"] = agent_history
        elif isinstance(result, tuple) and len(result) >= 7:
            # Fallback for non-custom agents
            (_, _, _, _, _, _, browser_context) = result
            _global_browser_context = browser_context

        # Update final state
        _current_agent_state.update({
            "is_running": False,
            "task_progress": "Completed",
            "last_update": datetime.now().isoformat()
        })
        
        # Update the AgentRun record with extended fields (including agent_history)
        try:
            from src.db.models import AgentRun  # AgentRun schema with new fields
            db = Database.get_database()
            update_fields = {}
            update_fields["status"] = "completed"
            update_fields["end_time"] = datetime.now().isoformat()
            update_fields["steps_completed"] = _current_agent_state["current_step"]
            update_fields["memory"] = _current_agent_state["memory"]
            update_fields["task_progress"] = _current_agent_state["task_progress"]
            update_fields["future_plans"] = _current_agent_state["future_plans"]

            if _current_agent_state.get("history_gif_url"):
                print(f"Updating history_gif_url: {_current_agent_state['history_gif_url']}")
                update_fields["history_gif_url"] = _current_agent_state["history_gif_url"]
            if _current_agent_state.get("recording_url"):
                print(f"Updating recording_url: {_current_agent_state['recording_url']}")
                update_fields["recording_url"] = _current_agent_state["recording_url"]
            if _current_agent_state.get("agent_history"):
                agent_history_obj = _current_agent_state["agent_history"]
                # If the object has a .dict() method (e.g., a Pydantic model), convert it
                if hasattr(agent_history_obj, "dict"):
                    history_dict = agent_history_obj.dict()
                    # Remove the "state" field from each history item
                    if "history" in history_dict:
                        for history_item in history_dict["history"]:
                            if "state" in history_item:
                                del history_item["state"]
                    update_fields["agent_history"] = history_dict
                else:
                    # Handle the case where it's already a dict
                    history_dict = agent_history_obj
                    if "history" in history_dict:
                        for history_item in history_dict["history"]:
                            if "state" in history_item:
                                del history_item["state"]
                    update_fields["agent_history"] = history_dict
            if update_fields:
                await db.agent_runs.update_one(
                    {"client_id": client_id},
                    {"$set": update_fields}
                )
                # Generate UI if memory exists
                if _current_agent_state["memory"]:
                    # Initialize variables
                    completion_summary = None
                    result_content = None
                    
                    # Try to get completion summary and result content from agent history
                    try:
                        agent_history_obj = _current_agent_state["agent_history"]
                        if hasattr(agent_history_obj, "dict"):
                            history_dict = agent_history_obj.dict()
                        else:
                            history_dict = agent_history_obj
                            
                        # Get the last history item
                        if history_dict.get("history"):
                            last_history_item = history_dict["history"][-1]
                            
                            # Look for 'done' action in model output
                            if "model_output" in last_history_item and "action" in last_history_item["model_output"]:
                                for action_item in last_history_item["model_output"]["action"]:
                                    if isinstance(action_item, dict) and "done" in action_item:
                                        completion_summary = action_item["done"].get("text", "")
                                        break
                            
                            # Look for result content where is_done is true
                            if "result" in last_history_item:
                                for result_item in last_history_item["result"]:
                                    if isinstance(result_item, dict) and result_item.get("is_done"):
                                        result_content = result_item.get("extracted_content", "")
                                        break
                                        
                    except Exception as e:
                        logger.error(f"Error extracting completion summary or result content: {e}")
                        completion_summary = None
                        result_content = None
                    
                    # Safely combine all content parts with separators
                    content_parts = []
                    
                    if completion_summary:
                        content_parts.append(f"Summary:\n{completion_summary}")
                    
                    if _current_agent_state["memory"]:
                        content_parts.append(f"Memory:\n{_current_agent_state['memory']}")
                    
                    if result_content:
                        content_parts.append(f"Results:\n{result_content}")
                    
                    # Join all non-empty parts with double newlines
                    content_for_ui = "\n\n".join(filter(None, content_parts))
                    
                    # If all parts were empty, use a default message
                    if not content_for_ui:
                        content_for_ui = "No content available from the agent run."
                    
                    generated_ui = await generative_ui_builder(content_for_ui)
                    print("\nGenerated UI Output:")
                    print(generated_ui)
                    
                    # Await the database update
                    await db.agent_runs.update_one(
                        {"client_id": client_id},
                        {"$set": {"generated_ui": generated_ui}}
                    )

                # Only broadcast after all updates are complete
                message = {
                    "type": "agent_finished",
                    "data": "Agent finished its work",
                    "timestamp": datetime.now().isoformat()
                }
                await manager.broadcast_to_client(client_id, json.dumps(message))
                
        except Exception as e:
            logger.error(f"Failed to update AgentRun record with extended data: {e}")

        return result
    except Exception as e:
        update_fields = {}
        update_fields["status"] = "failed"
        update_fields["end_time"] = datetime.now().isoformat()
        update_fields["errors"] = str(e)
        await db.agent_runs.update_one(
            {"client_id": client_id},
            {"$set": update_fields}
        )
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
async def stop_agent_endpoint():  # Renamed to avoid naming conflict
    """Stop the running agent"""
    global _current_agent_state
    
    try:
        logger.info("Attempting to stop agent...")
        # Call the imported stop_agent function from webui
        result = await stop_agent()
        
        if result and isinstance(result, tuple):
            _current_agent_state.update({
                "is_running": False,
                "task_progress": "Stopped by user",
                "last_update": datetime.now().isoformat()
            })
            return {"message": "Agent stopped successfully"}
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid response from stop_agent"}
            )
            
    except Exception as e:
        logger.error(f"Error stopping agent: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to stop agent", "details": str(e)}
        )

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

@app.post("/clerk-webhook")
async def clerk_webhook(request: Request):
    """Handle Clerk webhook events for user management"""
    try:
        data = await request.json()
        
        if data['type'] in ['user.created', 'user.updated']:
            user_data = data['data']
            clerk_id = user_data['id']
            email = user_data['email_addresses'][0]['email_address']
            first_name = user_data.get('first_name', '')
            image_url = user_data.get('image_url', '')

            # Get database instance
            db = Database.get_database()
            
            # Upsert user data
            await db.users.update_one(
                {"clerk_id": clerk_id},
                {
                    "$set": {
                        "email": email,
                        "first_name": first_name,
                        "image_url": image_url,
                        "updated_at": datetime.now()
                    }
                },
                upsert=True
            )
            
            return {"message": "User upserted successfully"}
            
        return JSONResponse(
            status_code=400,
            content={"message": "Unhandled event type"}
        )
        
    except Exception as e:
        logger.error(f"Error in clerk webhook: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Webhook processing failed: {str(e)}"
        )

async def get_authenticated_user(clerk_id: str = Header(..., alias="x-clerk-user-id")) -> str:
    """Dependency to get authenticated clerk_id from request header"""
    if not clerk_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return clerk_id

async def guardrail(task: str) -> Dict[str, Any]:
    """
    Validates if the given task is related to e-commerce/online shopping.
    Returns a dict with pass status and explanation.
    """
    try:
        print(f"Starting guardrail validation for task: {task}")
        
        # Initialize the LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=SecretStr(os.getenv("OPENAI_API_KEY", ""))
        )
        print("LLM initialized successfully")
        
        # Create the prompt template with properly escaped JSON
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a task validator that determines if a given task is related to or requires e-commerce, online shopping, or product research, either directly or indirectly. 

            Consider tasks valid ('true') if they:
            1. Directly involve shopping or product research
            2. Require price comparisons or market research
            3. Need product specifications or reviews
            4. Involve building setups/kits that require purchasing items
            5. Research-based tasks that would benefit from e-commerce site data
            
            Examples of valid tasks:
            - "Find me a cheap laptop under $500"
            - "I want to build a gaming setup with $2000 budget"
            - "Research what camera equipment I need for wildlife photography"
            - "Compare prices of standing desks across different stores"
            
            Respond with 'true' if the task relates to these topics (directly or indirectly), and 'false' if it doesn't.
            Also provide a brief explanation of your decision.
            
            Format your response exactly as:
            {{"pass": boolean, "comment": "explanation"}}"""),
            ("user", "Task: {task}")
        ])
        print("Prompt template created")
        
        # Get the response
        chain = prompt | llm
        print("Chain created, invoking LLM...")
        response = await chain.ainvoke({"task": task})
        print(f"Raw LLM response: {response.content}")
        
        # Parse the response
        result = json.loads(response.content)
        print(f"Parsed result: {result}")
        return result
        
    except Exception as e:
        print(f"Error in guardrail validation: {str(e)}")
        logger.error(f"Error in guardrail validation: {str(e)}")
        return {
            "pass": False,
            "comment": f"Error validating task: {str(e)}"
        }

async def get_prompt_dynamic_filters(task: str) -> Dict[str, Any]:
    """
    Analyzes the task string and extracts potential filter variations.
    Returns a dictionary with key words and their alternative options.
    """
    try:
        print(f"Starting dynamic filter extraction for task: {task}")
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=SecretStr(os.getenv("OPENAI_API_KEY", ""))
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a filter extraction system that analyzes shopping-related tasks and identifies key terms that could have alternatives.
            For each key term (like website names, colors, price indicators, gender, etc.), provide 2-3 relevant alternatives.
            
            Format your response as a valid JSON object where:
            - Keys are the original terms found in the task
            - Values are arrays of alternative options
            - Ensure all strings are properly quoted
            - Use double quotes for all keys and string values
            - Ensure all arrays end with proper commas
            
            Example Task: "Go to Amazon and find me the cheapest Black shirt for men"
            Example Response:
            {{
                "Amazon": ["Walmart", "Target", "eBay"],
                "cheapest": ["most expensive", "mid-range", "premium"],
                "Black": ["White", "Blue", "Gray"],
                "men": ["women", "unisex", "kids"]
            }}
            
            Only include meaningful shopping-related terms that could serve as filters. Ignore common words or articles."""),
            ("user", "Task: {task}")
        ])
        
        chain = prompt | llm
        response = await chain.ainvoke({"task": task})
        
        # Add error checking for JSON parsing
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError as json_error:
            print(f"JSON parsing error: {json_error}")
            print(f"Raw response content: {response.content}")
            # Return empty dict on parsing error
            return {}
            
        print(f"Extracted filters: {result}")
        return result
        
    except Exception as e:
        print(f"Error in dynamic filter extraction: {str(e)}")
        logger.error(f"Error in dynamic filter extraction: {str(e)}")
        return {}

@app.get("/agent-runs/get/{clerk_id}")
async def get_agent_runs(clerk_id: str):
    """Get all agent runs for a specific clerk_id"""
    try:
        # Get database instance
        db = Database.get_database()
        
        # Query the database for all agent runs matching the clerk_id
        # Sort by start_time in descending order (newest first)
        cursor = db.agent_runs.find(
            {"clerk_id": clerk_id}
        ).sort("start_time", -1)
        
        # Convert cursor to list of AgentRun models
        agent_runs = []
        async for run in cursor:
            # Convert MongoDB _id to string
            run_id = str(run["_id"])
            run["_id"] = run_id
            agent_run = AgentRun(**run)
            # Create dictionary from model and explicitly add _id
            run_dict = agent_run.model_dump()
            run_dict["_id"] = run_id
            agent_runs.append(run_dict)
        
        return {
            "agent_runs": agent_runs,
            "total": len(agent_runs)
        }
        
    except Exception as e:
        logger.error(f"Error fetching agent runs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch agent runs: {str(e)}"
        )

@app.get("/agent-run/get/{agent_run_id}")
async def get_agent_run(agent_run_id: str):
    """Get a specific agent run by its ID"""
    try:
        from bson import ObjectId
        
        # Get database instance
        db = Database.get_database()
        
        # Try to convert string ID to ObjectId
        try:
            object_id = ObjectId(agent_run_id)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent run ID format: {str(e)}"
            )
        
        # Query the database for the specific agent run
        agent_run = await db.agent_runs.find_one({"_id": object_id})
        
        if not agent_run:
            raise HTTPException(
                status_code=404,
                detail=f"Agent run with ID {agent_run_id} not found"
            )
        
        # Convert MongoDB _id to string
        agent_run["_id"] = str(agent_run["_id"])
        
        # Convert to AgentRun model and return
        return AgentRun(**agent_run).model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching agent run: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch agent run: {str(e)}"
        )

async def generative_ui_builder(memory_text: str) -> str:
    """
    Transforms unstructured memory text into structured HTML+Tailwind UI
    """
    # Base template that includes necessary Tailwind CDN and basic layout
    base_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/@alpinejs/collapse@3.x.x/dist/cdn.min.js"></script>
        <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js"></script>
        <title>Shopping Analysis</title>
    </head>
    <body class="bg-gray-50">
        <div class="container mx-auto px-4 py-8">
            {{content}}
        </div>
    </body>
    </html>
    """

    system_prompt = """You are an expert UI developer who converts unstructured text about shopping and product research into simple, clean HTML with Tailwind styling.

    IMPORTANT: You must ONLY output the complete HTML code. Do not include any explanations, markdown code blocks, or additional text.
    Your entire response should be valid HTML that starts with <!DOCTYPE html> and ends with </html>.

    Content Analysis Guidelines:
    1. For product listings, comparisons, or specifications:
       - ALWAYS use <table> with proper structure
       - Include headers for each data column
       - Use consistent formatting for prices and metrics
    
    2. For narrative content or general findings:
       - Use article sections with clear headings
       - Break down into readable paragraphs
       - Use lists for sequential steps or bullet points
    
    3. For mixed content:
       - Lead with key findings in a summary section
       - Follow with detailed tables for product data
       - Use expandable sections for additional details

    Styling Guidelines:
    Tables (for product data):
    - Use 'border-collapse' with 'border-gray-200'
    - Add 'hover:bg-gray-50' for row interactions
    - Include 'px-6 py-4' for cell padding
    - Use 'whitespace-nowrap' for prices/metrics
    - Add 'sticky top-0 bg-white' for headers

    Text Content:
    - Use 'prose prose-lg' for narrative sections
    - Include 'text-gray-600' for descriptive text
    - Add 'font-semibold' for important points
    - Use 'space-y-4' for proper spacing

    Technical Requirements:
    1. Use only basic HTML (div, p, table, ul, etc.)
    2. No JavaScript or Alpine.js
    3. Use semantic HTML and accessible components
    4. Use the provided base template
    5. Keep it React-compatible (dangerouslySetInnerHTML)

    Remember: Output ONLY the HTML code. No explanations or markdown formatting."""

    user_prompt = """Convert this text into HTML+Tailwind UI. Remember to output ONLY the HTML code:

    {memory_text}"""

    try:
        llm = ChatOpenAI(
            model="o3-mini",
            api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
            # response_format={"type": "text"}  # Enforce text-only response
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])

        chain = prompt | llm
        response = await chain.ainvoke({
            "memory_text": memory_text
        })
        
        # Extract only the HTML content
        html_content = response.content.strip()
        
        # Verify the response starts with <!DOCTYPE html>
        if not html_content.startswith("<!DOCTYPE html>"):
            logger.warning("LLM response did not contain proper HTML. Using fallback template.")
            return base_template.replace("{{content}}", f"<pre class='whitespace-pre-wrap'>{memory_text}</pre>")
            
        return html_content

    except Exception as e:
        logger.error(f"Error in generative_ui_builder: {str(e)}")
        return f"<p class='text-red-500'>Error generating UI: {str(e)}</p>"

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