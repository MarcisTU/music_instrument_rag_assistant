import argparse
import asyncio
import json
import os
import signal
import uuid
from datetime import datetime, timezone

import aio_pika
from loguru import logger

from src.services.llm_service import LLMService


WORKER_TYPE = "llm"
WORKER_ID = f"worker-{uuid.uuid4().hex[:8]}"

logger.add(f"./logs/worker_{WORKER_ID}.log", rotation="00:00", retention="7 days")


class LLMWorker:
    def __init__(self, args):
        self.connection = None
        self.channel = None
        self.queue = None
        self.routing_key = f"worker.{WORKER_TYPE}"

        self.llm_service = LLMService(use_reranker=args.use_reranker)

    async def connect(self):
        self.connection = await aio_pika.connect_robust(os.environ["RABBITMQ_URL"])
        self.channel = await self.connection.channel()

        # Ensure we only take one task at a time
        await self.channel.set_qos(prefetch_count=1)

        # Declare the exchange (must match broker)
        self.exchange = await self.channel.declare_exchange(
            os.environ["EXCHANGE_NAME"], aio_pika.ExchangeType.DIRECT, durable=True
        )

        # Declare a private queue for this worker and bind it to the exchange
        # We use a non-durable, auto-delete queue so it vanishes when the worker dies
        self.queue = await self.channel.declare_queue(
            f"queue.{WORKER_ID}", auto_delete=True, exclusive=True
        )
        await self.queue.bind(self.exchange, routing_key=self.routing_key)

        logger.info(f"Worker {WORKER_ID} connected and bound to {self.routing_key}")

    async def send_status(self, status: str = "heartbeat", extra_data: dict = None):
        payload = {
            "worker_type": WORKER_TYPE,
            "worker_id": WORKER_ID,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra_data:
            payload.update(extra_data)

        try:
            await self.channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(payload).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=os.environ["RESPONSE_QUEUE"],
            )
        except Exception as e:
            logger.error(f"Failed to send status {status}: {e}")

    async def heartbeat_loop(self):
        while True:
            await self.send_status("heartbeat")
            await asyncio.sleep(30)  # Heartbeat every 30 seconds

    async def process_task(self, message: aio_pika.IncomingMessage):
        async with message.process():
            try:
                payload = json.loads(message.body.decode())
                request_id = payload["request_id"]

                logger.info(f"Processing task {request_id}...")

                result_text = await self.llm_service.inference(user_query=payload['user_query'])
                logger.info(f"Finished task {request_id}. \nResult: {result_text}")

                result_data = {
                    "request_id": request_id,
                    "result": result_text,
                    "callback_url": payload["callback_url"],
                    "metadata": payload["metadata"],
                    "status": "success"
                }

                # Send result back (this also acts as a registration for the next task)
                await self.send_status("success", result_data)
                logger.info(f"Task {request_id} completed and result sent.")

            except Exception as e:
                logger.exception(f"Error processing task: {e}")

    async def run(self):
        await self.connect()

        asyncio.create_task(self.heartbeat_loop())

        logger.info(f"Worker {WORKER_ID} waiting for tasks...")

        try:
            await self.queue.consume(self.process_task)
            await asyncio.Future()  # Run forever
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Worker shutting down...")
        finally:
            # Notify broker we are leaving
            await self.send_status("shutdown")
            if self.connection:
                await self.connection.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Worker args")
    parser.add_argument(
        "--use_reranker",
        default=False,
        help="Enable the use of reranking after retrieving initial similarity docs."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    worker = LLMWorker(
        args=args
    )

    # Create the main event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


    def handle_exit_signal():
        logger.warning("Received stop signal (SIGTERM/SIGINT). Initiating graceful shutdown...")
        main_task.cancel()


    # Register handlers for both Docker stop (SIGTERM) and Ctrl+C (SIGINT)
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig=sig, callback=handle_exit_signal)

    # Wrap worker in a task so we can cancel it from the signal handler
    main_task = loop.create_task(worker.run())

    try:
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        logger.info("Main worker task cancelled successfully.")
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()
            logger.info("Worker process completely stopped.")
