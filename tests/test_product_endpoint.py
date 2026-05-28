import asyncio
import sys
import httpx
import uvicorn
from fastapi import FastAPI, Request
from loguru import logger

# Configuration for the test environment
API_URL = "http://localhost:8082"  # api deployed target url
TEST_HOST = "127.0.0.1"
TEST_PORT = 9001
CALLBACK_URL = f"http://{TEST_HOST}:{TEST_PORT}/test_callback"

# Configure clear logger formatting for terminal viewing
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


async def run_standalone_callback_test():
    callback_received_event = asyncio.Event()
    received_payload = {}

    # Setup a lightweight, temporary webhook server
    callback_app = FastAPI()

    @callback_app.post("/test_callback")
    async def handle_callback(request: Request):
        nonlocal received_payload
        try:
            payload = await request.json()
            logger.info(f"🎯 Local Webhook Listener received callback payload: {payload}")

            # Store payload data for validation step
            received_payload.update(payload)

            # Fire event to unblock the main thread execution
            callback_received_event.set()
            return {"status": "accepted"}
        except Exception as e:
            logger.error(f"Error reading callback payload: {e}")
            return {"status": "error"}

    # Spin up the webhook server background task using uvicorn
    config = uvicorn.Config(app=callback_app, host=TEST_HOST, port=TEST_PORT, log_level="warning")
    server = uvicorn.Server(config)

    server_task = asyncio.create_task(server.serve())
    logger.info(f"Local test webhook server listening at: {CALLBACK_URL}")

    # Give uvicorn a brief split-second window to bind safely to port 9000
    await asyncio.sleep(0.5)

    test_failed = False
    try:
        # Dispatch the initial request to your active api_worker app
        async with httpx.AsyncClient(base_url=API_URL, timeout=10.0) as client:
            # test_query = f"I’m looking for a versatile MIDI keyboard controller for music production and film scoring, preferably with 49 or 61 semi-weighted keys, velocity sensitivity, aftertouch, and assignable pads/knobs for DAW control."
            # test_query = "I’m looking for a modern electric guitar for progressive metal and hard rock, with a roasted maple neck, stainless steel frets, and active humbuckers like Fishman Fluence. I’d like models similar with floyd rose locking tremolo."
            test_query = "I’m looking for a matched pair of condenser microphones specifically for drum overhead recording in a studio setup. Preferably small-diaphragm condensers with a detailed high-end response, low self-noise, and good stereo imaging for capturing cymbals and room ambience in rock and fusion mixes."

            # This dictionary payload perfectly matches your backend's TaskSubmitRequest schema
            payload = {
                "user_query": test_query,
                "callback_url": CALLBACK_URL
            }

            logger.info(f"📤 Posting request JSON payload to api_worker ({API_URL}): {payload}")

            response = await client.post("/api/v1/task_submit", json=payload)

            if response.status_code != 202:
                logger.error(f"❌ Initial request rejected. Expected 202, got {response.status_code}: {response.text}")
                return

            response_data = response.json()
            task_uuid = response_data.get("task_uuid")
            logger.info(f"✅ Request accepted by API. Tracking Task UUID: {task_uuid}")

        logger.info("⏳ Webhook server sitting idle. Waiting for downstream worker cluster to deliver response...")

        try:
            await asyncio.wait_for(callback_received_event.wait(), timeout=60.0)

            # 5. Evaluate payload variables to confirm exact transaction tracking integrity
            print("\n" + "-" * 40 + "\n🔍 VALIDATING RECEIVED DATA\n" + "-" * 40)

            if received_payload["request_id"] == task_uuid:
                logger.info("✅ SUCCESS: Callback request_id matches original Task UUID perfectly.")
                logger.info(f"📦 Result string payload content: '{received_payload.get('result')}'")
            else:
                logger.error(
                    f"❌ MISMATCH ERROR: Received ID ({received_payload.get('request_id')}) does not match sent ID ({task_uuid})")
                test_failed = True

        except asyncio.TimeoutError:
            logger.error("❌ TIMEOUT ERROR: No callback hit our server.")
            test_failed = True

    finally:
        logger.info("Shutting down local webhook server...")
        server.should_exit = True
        await server_task
        logger.info("Test pipeline closed safely.")

        if test_failed:
            print("❌ TEST STATUS: FAILED")
            sys.exit(1)
        else:
            print("🎉 TEST STATUS: ALL CHECKS PASSED SUCCESSFULLY")
            sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(run_standalone_callback_test())
    except KeyboardInterrupt:
        print("\nTest manually aborted by user via Ctrl+C.")