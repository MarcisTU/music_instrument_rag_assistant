import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import aio_pika
from fastapi import FastAPI, HTTPException, Depends
from loguru import logger
from src.db.service import TaskService
from src.modules.mq_connection_manager import RabbitMQManager


rmq_manager = RabbitMQManager(os.environ["RABBITMQ_URL"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    await rmq_manager.connect()
    yield
    await rmq_manager.close()


app = FastAPI(lifespan=lifespan)


async def get_rmq_channel() -> aio_pika.RobustChannel:
    return await rmq_manager.get_channel()


@app.get("/")
async def root():
    return {"message": "Hey!"}


@app.post("/api/v1/products", status_code=202)
async def create_product_search_task(
    user_query: str,
    callback_url: Optional[str],
    channel: aio_pika.RobustChannel = Depends(get_rmq_channel)
):
    task_uuid = str(uuid.uuid4())

    try:
        await TaskService.insert_task(
            user_query=user_query,
            callback_url=callback_url,
            task_uuid=task_uuid
        )
    except Exception as e:
        logger.error(f"Database insertion failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize task tracking in database.")


    task_payload = {
        "request_id": task_uuid,
        "worker_type": "product_rag_worker",  # Tells your BrokerWorker which group routing key to hit
        "user_query": user_query,
        "callback_url": callback_url,
        "metadata": {"generated_by": "fastapi_v1_products"}
    }

    try:
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(task_payload).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=os.environ["REQUEST_QUEUE"],
        )
        logger.info(f"Successfully published task {task_uuid} to {os.environ["REQUEST_QUEUE"]}")

    except Exception as e:
        logger.error(f"Failed to push message payload to RabbitMQ queue: {e}")
        raise HTTPException(status_code=500, detail="Task saved to db but failed to publish.")

    return {
        "status": "Accepted",
        "task_uuid": task_uuid,
        "message": "Your search query processing task has been queued."
    }
