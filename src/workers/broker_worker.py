import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime

import aio_pika
from aio_pika import ExchangeType, IncomingMessage
from loguru import logger

from src.db.service import TaskService
from src.models.schemas import TaskUpdate

logger.add("./logs/broker_worker.log", rotation="00:00", retention="7 days")


class BrokerWorker:
    def __init__(self):
        self.available_workers: dict[str, set[str]] = defaultdict(set)
        self.pending_requests: list[dict] = []

        self.connection = None
        self.channel = None

        self.request_queue = None
        self.response_queue = None
        self.callback_queue = None

    async def connect(self):
        self.connection = await aio_pika.connect_robust(os.environ["RABBITMQ_URL"])
        self.channel = await self.connection.channel()
        await self.channel.set_qos(prefetch_count=1)

        self.request_queue = await self.channel.declare_queue(os.environ["REQUEST_QUEUE"], durable=True)
        self.response_queue = await self.channel.declare_queue(os.environ["RESPONSE_QUEUE"], durable=True)
        self.callback_queue = await self.channel.declare_queue(os.environ["CALLBACK_QUEUE"], durable=True)

        self.exchange = await self.channel.declare_exchange("rag.direct", ExchangeType.DIRECT, durable=True)

        logger.info("RabbitMQ connected")

    async def print_stats(self):
        while True:
            await asyncio.sleep(5)

            available_worker_count = sum(len(workers) for workers in self.available_workers.values())
            logger.info(f"Available workers: {available_worker_count}, Pending requests: {len(self.pending_requests)}")

            if self.pending_requests:
                oldest = self.pending_requests[0]
                logger.info(f"Oldest pending request: {oldest['request_id']} | {oldest['queued_at']}")

    async def register_worker(self, worker_type: str, worker_id: str):
        if worker_id not in self.available_workers[worker_type]:
            self.available_workers[worker_type].add(worker_id)

            logger.info(f"Worker registered: {worker_type}/{worker_id}")

    async def unregister_worker(self, worker_type: str, worker_id: str):
        self.available_workers[worker_type].discard(worker_id)

        logger.info(f"Worker removed: {worker_type}/{worker_id}")

    async def dispatch_request(self, request: dict):
        worker_type = request["worker_type"]

        if not self.available_workers[worker_type]:
            logger.warning(f"No workers available for {worker_type}. Queueing request {request['request_id']}")
            self.pending_requests.append(request)
            return

        routing_key = f"worker.{worker_type}"

        await self.exchange.publish(
            aio_pika.Message(
                body=json.dumps(request).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=routing_key,
        )

        logger.info(f"Dispatched request {request['request_id']} to {worker_type}")

    async def process_pending_requests(self):
        if not self.pending_requests:
            return

        remaining = []
        for request in self.pending_requests:
            worker_type = request["worker_type"]

            if self.available_workers[worker_type]:
                await self.dispatch_request(request)
            else:
                remaining.append(request)

        self.pending_requests = remaining

    async def handle_api_request(self, message: IncomingMessage):
        async with message.process():
            try:
                payload = json.loads(message.body.decode())
                payload["queued_at"] = datetime.utcnow().isoformat()
                logger.info(f"Received API request: {payload['request_id']}")

                await self.dispatch_request(payload)

            except Exception as e:
                logger.exception(e)

    async def handle_worker_response(self, message: IncomingMessage):
        async with message.process():
            try:
                payload = json.loads(message.body.decode())
                worker_type = payload["worker_type"]
                worker_id = payload["worker_id"]
                task_uuid = payload["request_id"]

                # Update worker as available
                await self.register_worker(worker_type, worker_id)

                if payload["status"] == "shutdown":
                    # remove worker from tracking
                    await self.unregister_worker(worker_type, worker_id)
                    return
                elif payload["status"] == "heartbeat":
                    # Worker sent a heartbeat status message to be registered in broker
                    return

                logger.info(f"Received result response for task {task_uuid}")

                task_update = TaskUpdate(
                    status="ready",
                    llm_result_text=payload["result"]
                )
                await TaskService.update_task(
                    task_uuid=task_uuid,
                    update_data=task_update
                )

                if payload["callback_url"] is not None and len(payload["callback_url"].strip()) > 0:
                    # Forward to Callback Worker for sending results back to client
                    callback_payload = {
                        "request_id": task_uuid,
                        "result": payload["result"],
                        "callback_url": payload["callback_url"]
                    }

                    await self.channel.default_exchange.publish(
                        aio_pika.Message(
                            body=json.dumps(callback_payload).encode(),
                            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                        ),
                        routing_key=os.environ["CALLBACK_QUEUE"],
                    )
                    logger.info(f"Result for task {task_uuid} sent to callback worker")

                await self.process_pending_requests()

            except Exception as e:
                logger.exception(e)

    async def requeue_pending_on_exit(self):
        """Moves in-memory pending_requests back to RabbitMQ before exit."""
        if not self.pending_requests:
            return

        logger.info(f"Requeuing {len(self.pending_requests)} pending requests to RabbitMQ...")

        for request in self.pending_requests:
            try:
                await self.channel.default_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(request).encode(),
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    ),
                    routing_key=os.environ["REQUEST_QUEUE"],
                )
            except Exception as e:
                logger.error(f"Failed to requeue request {request.get('request_id')}: {e}")

        self.pending_requests.clear()

    async def run(self):
        await self.connect()
        asyncio.create_task(self.print_stats())

        # use tag to easily cancel consumption if needed
        request_consumer_tag = await self.request_queue.consume(self.handle_api_request)
        await self.response_queue.consume(self.handle_worker_response)

        logger.info("Broker worker started. Press Ctrl+C to exit.")

        try:
            # Keep the loop alive
            await asyncio.Future()
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Shutdown signal received...")
        finally:
            # Shutdown & Stop accepting new requests from RabbitMQ first
            if self.request_queue:
                await self.request_queue.cancel(request_consumer_tag)
                logger.info("Stopped consuming new requests.")

            await self.requeue_pending_on_exit()

            if self.connection:
                await self.connection.close()

            logger.info("Broker worker shut down gracefully.")


if __name__ == "__main__":
    broker = BrokerWorker()

    asyncio.run(broker.run())
