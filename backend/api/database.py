from motor.motor_asyncio import AsyncIOMotorClient
import os

# MongoDB Connection String
# Default to local MongoDB, but can be overridden by environment variable
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")

client = AsyncIOMotorClient(MONGODB_URL)
database = client.sentiment_db
predictions_collection = database.get_collection("predictions")

async def save_prediction(prediction_data: dict):
    try:
        await predictions_collection.insert_one(prediction_data)
    except Exception as e:
        print(f"⚠️ Warning: Could not save to MongoDB (History will not be updated): {e}")

async def get_all_history(limit: int = 50):
    try:
        cursor = predictions_collection.find().sort("timestamp", -1).limit(limit)
        history = []
        async for document in cursor:
            document["_id"] = str(document["_id"])  # Convert ObjectId to string
            history.append(document)
        return history
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch history from MongoDB: {e}")
        return []
