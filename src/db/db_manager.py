from src.db.models import AgentRun
from src.db.db import Database  # Reusing the existing Database connection from your codebase
import logging

logger = logging.getLogger(__name__)

async def update_agent_run_history(agent_id: str, agent_history: dict):
    """
    Update the AgentRun record in the database with the given agent_history.
    
    This function uses the asynchronous database connection provided by
    Database.get_database() (as seen in your server.py) and updates the record
    where "client_id" equals the given agent_id.
    
    Example for MongoDB (using Motor):
        db = Database.get_database()
        result = await db.agent_runs.update_one(
            {"client_id": agent_id},
            {"$set": {"agent_history": agent_history}},
            upsert=True
        )
        return result
    """
    try:
        db = Database.get_database()
        result = await db.agent_runs.update_one(
            {"client_id": agent_id},
            {"$set": {"agent_history": agent_history}},
            upsert=True
        )
        logger.info(f"Successfully updated AgentRun for client_id {agent_id}: {result.raw_result}")
        return result
    except Exception as e:
        logger.error(f"Error updating AgentRun for client_id {agent_id}: {e}")
        raise e 