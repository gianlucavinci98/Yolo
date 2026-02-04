from starlette.requests import Request
from starlette.responses import JSONResponse
from ray import serve
from ray.serve.handle import DeploymentHandle
import logging

@serve.deployment(name="YoloIngress")
class YoloIngress:
    def __init__(self, decoder_handle: DeploymentHandle, detector_handle: DeploymentHandle):
        self.decoder = decoder_handle
        self.detector = detector_handle
        self.logger = logging.getLogger("YoloIngress")
        logging.basicConfig(level=logging.INFO)

    async def __call__(self, http_request: Request):
        form_data = await http_request.form()

        self.logger.info("Content-Type: %s", http_request.headers.get("content-type"))
        self.logger.info("Form keys: %s", list(form_data.keys()))

        
        if 'frames' not in form_data:
            return JSONResponse({'error': 'No frames found'}, status_code=400)

        image_bytes = await form_data['frames'].read()

        # PIPELINE ASINCRONA:
        # 1. Chiedi al decoder di decomprimere
        ref_decoded = await self.decoder.decode.remote(image_bytes)

        # 2. Passa il riferimento dell'immagine decodificata DIRETTAMENTE al detector
        # Ray gestisce il passaggio dati in background (Object Store)
        ref_result = await self.detector.detect.remote(ref_decoded)

        return JSONResponse(ref_result)