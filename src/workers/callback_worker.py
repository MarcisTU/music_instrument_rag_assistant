import asyncio
import json
import os
import signal
import sys
from datetime import datetime

import aio_pika
import httpx
from aio_pika import IncomingMessage
from loguru import logger

os.makedirs("./logs", exist_ok=True)
logger.add("./logs/callback_worker.log", rotation="00:00", retention="7 days")


class CallbackWorker:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.callback_queue = None

        # Share a single HTTPX Client instance across tasks for connection pooling
        self.http_client = httpx.AsyncClient(timeout=10.0)

    async def connect(self):
        rabbitmq_url = os.environ["RABBITMQ_URL"]
        queue_name = os.environ["CALLBACK_QUEUE"]

        logger.info("Connecting to RabbitMQ for Callback Worker...")
        self.connection = await aio_pika.connect_robust(rabbitmq_url)
        self.channel = await self.connection.channel()

        # Process up to 10 callbacks concurrently per worker instance
        await self.channel.set_qos(prefetch_count=10)

        self.callback_queue = await self.channel.declare_queue(queue_name, durable=True)
        logger.info(f"Callback Worker linked and listening on: {queue_name}")

    async def send_callback_with_retry(self, callback_url: str, payload: dict, max_tries: int = 3) -> bool:
        """Sends the HTTP POST callback, retrying up to max_tries with backoff."""
        request_id = payload["request_id"]

        for attempt in range(1, max_tries + 1):
            try:
                logger.info(f"Sending callback for {request_id} to {callback_url} (Attempt {attempt}/{max_tries})")

                response = await self.http_client.post(callback_url, json=payload)

                # Raise exception for 4xx/5xx status codes to trigger retry logic
                response.raise_for_status()

                logger.info(f"Successfully delivered callback for {request_id}. Status: {response.status_code}")
                return True

            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                logger.warning(f"Attempt {attempt} failed for task {request_id}: {exc}")

                if attempt < max_tries:
                    # Exponential backoff: Wait 2s, then 4s...
                    sleep_time = 2 ** attempt
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(f"Max retries reached. Callback delivery failed permanently for task {request_id}.")

        return False

    async def process_callback(self, message: IncomingMessage):
        asyncio.create_task(self._safe_execute_callback(message))

    async def _safe_execute_callback(self, message: IncomingMessage):
        async with message.process(requeue=False):
            try:
                payload = json.loads(message.body.decode())
                request_id = payload["request_id"]
                callback_url = payload["callback_url"]

                if not callback_url:
                    logger.error(f"Missing callback_url in message payload for task {request_id}. Dropping.")
                    return

                # Compile structured payload for client webhook consumers
                client_payload = {
                    "task_uuid": request_id,
                    "status": "completed",
                    "result": payload["result"],
                    "completed_at": datetime.utcnow().isoformat()
                }

                # Async call out to the endpoint
                success = await self.send_callback_with_retry(callback_url, client_payload)

                if not success:
                    logger.error(f"Task {request_id} failed to send callback.")

            except Exception as e:
                logger.exception(f"Fatal crash processing item from callback queue: {e}")

    async def run(self):
        await self.connect()

        await self.callback_queue.consume(self.process_callback)
        logger.info("Callback Worker engine running. Press Ctrl+C to terminate.")

        try:
            await asyncio.Future()  # Keeps loop alive infinitely
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Shutdown sequence activated...")
        finally:
            # Clean up network resources cleanly on exit
            await self.http_client.aclose()
            if self.connection:
                await self.connection.close()
            logger.info("Callback Worker disconnected safely.")


if __name__ == "__main__":
    if not os.environ["RABBITMQ_URL"] or not os.environ["CALLBACK_QUEUE"]:
        logger.warning("Error: RABBITMQ_URL and CALLBACK_QUEUE environment variables must be defined.")
        sys.exit(1)

    worker = CallbackWorker()

    # Create a dedicated event loop to capture OS signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


    def handle_exit_signal():
        logger.warning("Received stop signal (SIGTERM/SIGINT). Shutting down Callback Worker...")
        main_task.cancel()


    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig=sig, callback=handle_exit_signal)

    main_task = loop.create_task(worker.run())

    try:
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        logger.info("Main callback task cancelled via signal.")
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()
            logger.info("Callback process dead and buried.")
