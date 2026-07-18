import asyncio
import websockets

async def test_ws():
    uri = "ws://localhost:8000/ws/exotel/a18410ec-71a3-4aed-891c-862c9172d86e/test-call-123"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            # We don't need to send anything immediately, pipecat should start sending audio or events
            res = await websocket.recv()
            print(f"Received: {res}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_ws())
