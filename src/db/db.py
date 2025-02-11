from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class Database:
    client: Optional[AsyncIOMotorClient] = None
    
    @classmethod
    async def connect_to_database(cls, mongodb_url: str):
        try:
            # Create a new client and connect to the server
            cls.client = AsyncIOMotorClient(
                mongodb_url,
                server_api=ServerApi('1')  # Use the latest stable API version
            )
            
            # Send a ping to confirm a successful connection
            await cls.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB!")
            
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise

    @classmethod
    async def close_database_connection(cls):
        if cls.client:
            cls.client.close()
            logger.info("MongoDB connection closed")

    @classmethod
    def get_database(cls):
        if not cls.client:
            raise ConnectionError("Database not initialized. Call connect_to_database first.")
        return cls.client.get_default_database() 