from starlette.requests import Request
from starlette.responses import JSONResponse
from ray import serve
from ray.serve.handle import DeploymentHandle
import logging
import time
import asyncio

@serve.deployment(name="YoloIngress", num_replicas=1, ray_actor_options={"num_cpus": 0.1})
class YoloIngress:
    def __init__(self, decoder_handle: DeploymentHandle, detector_handle: DeploymentHandle):
        self.decoder = decoder_handle
        self.detector = detector_handle
        
        self.logger = logging.getLogger("ray.serve.ingress")
        self.logger.setLevel(logging.INFO)
        self.logger.info("Ingress initialized - Logger configurato correttamente")

    async def __call__(self, http_request: Request):
        self.logger.info(
            "Incoming request: method=%s content_type=%s",
            http_request.method,
            http_request.headers.get("content-type")
        )
        form_data = await http_request.form()
        
        frames = form_data.getlist("frames")
        if not frames:
            self.logger.warning("No frames found in form data. keys=%s", list(form_data.keys()))
            return JSONResponse({'error': 'No frames found'}, status_code=400)


        # Leggi tutti i file in parallelo
        read_tasks = [f.read() for f in frames]
        images_bytes = await asyncio.gather(*read_tasks)
        self.logger.info("Received %d frames", len(images_bytes))


        # ASYNCHRONOUS PIPELINE:
        # 1. Ask the decoder to decompress
        decode_refs = [await self.decoder.decode.remote(b) for b in images_bytes]

        # 2. Pass the reference of the decoded image DIRECTLY to the detector
        # Ray handles data passing in the background (Object Store)
        detect_refs = [await self.detector.detect.remote(ref) for ref in decode_refs]
        self.logger.info("Detection COMPLETED for all frames")

        results = []
        for idx, ref_result in enumerate(detect_refs):
            results.append({
                'frame_index': idx,
                'detections': ref_result
            })

        return JSONResponse(results)