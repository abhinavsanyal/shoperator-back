import os
import json
from src.db.db_manager import update_agent_run_history

def run_org_agent(agent, save_agent_history_path):
    history_file = os.path.join(save_agent_history_path, f"{agent.agent_id}.json") if save_agent_history_path else None
    if history_file:
        agent.save_history(history_file)
        # Update DB with the agent history JSON
        try:
            with open(history_file, "r") as f:
                agent_history_data = json.load(f)
            update_agent_run_history(agent_id=agent.agent_id, agent_history=agent_history_data)
        except Exception as e:
            logger.error(f"Failed to update DB with agent history for {agent.agent_id}: {e}") 