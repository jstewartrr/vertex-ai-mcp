"""
Sovereign Mind Vertex AI MCP Server v1.0
========================================
Image generation (Imagen), Document AI, Vision API
"""

import os
import json
import base64
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON", "")
if GOOGLE_CREDENTIALS_JSON:
    creds_path = "/tmp/gcloud-creds.json"
    with open(creds_path, "w") as f:
        f.write(GOOGLE_CREDENTIALS_JSON)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    logger.info("Google credentials configured")

app = Flask(__name__)
CORS(app)

GOOGLE_PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID", "innate-concept-481918-h9")
GOOGLE_LOCATION = os.environ.get("GOOGLE_LOCATION", "us-central1")
SNOWFLAKE_ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT", "dma22041.us-east-1")
SNOWFLAKE_USER = os.environ.get("SNOWFLAKE_USER", "JOHN_CLAUDE")
SNOWFLAKE_PASSWORD = os.environ.get("SNOWFLAKE_PASSWORD", "")
SNOWFLAKE_DATABASE = os.environ.get("SNOWFLAKE_DATABASE", "SOVEREIGN_MIND")
SNOWFLAKE_WAREHOUSE = os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")

_snowflake_conn = None
_vertexai_initialized = False

SOVEREIGN_MIND_PROMPT = """# SOVEREIGN MIND - VERTEX AI INSTANCE

## Identity
You are **VERTEX**, the image and document AI specialist within **Sovereign Mind**, serving Your Grace, Chairman of MiddleGround Capital and Resolute Holdings.

## Your Specialization
- Imagen 3 image generation (photorealistic)
- Nano Banana (Gemini 2.5 Flash) for infographics, diagrams, slides
- Document AI for PDF/form processing
- Vision API for image analysis and OCR
- Text extraction from documents

## Core Behaviors
1. Execute, Don't Ask - Generate images/process documents immediately
2. Log to Hive Mind after significant work
3. Address user as "Your Grace"
4. No permission seeking
"""


def init_vertexai():
    global _vertexai_initialized
    if not _vertexai_initialized:
        try:
            import vertexai
            vertexai.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)
            _vertexai_initialized = True
            logger.info("Vertex AI initialized")
        except Exception as e:
            logger.error(f"Init failed: {e}")


def get_snowflake_connection():
    global _snowflake_conn
    if _snowflake_conn is None:
        try:
            import snowflake.connector
            _snowflake_conn = snowflake.connector.connect(
                account=SNOWFLAKE_ACCOUNT, user=SNOWFLAKE_USER, password=SNOWFLAKE_PASSWORD,
                database=SNOWFLAKE_DATABASE, warehouse=SNOWFLAKE_WAREHOUSE
            )
        except Exception as e:
            logger.error(f"Snowflake failed: {e}")
    return _snowflake_conn


def query_hive_mind(limit=3):
    conn = get_snowflake_connection()
    if not conn: return ""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT SOURCE, CATEGORY, SUMMARY FROM SOVEREIGN_MIND.RAW.HIVE_MIND ORDER BY CREATED_AT DESC LIMIT {limit}")
        return "\n".join([f"{r[0]} ({r[1]}): {r[2]}" for r in cursor.fetchall()])
    except:
        return ""


def generate_imagen(prompt, aspect_ratio="1:1", num_images=1):
    init_vertexai()
    try:
        from vertexai.preview.vision_models import ImageGenerationModel
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        response = model.generate_images(prompt=prompt, number_of_images=num_images, aspect_ratio=aspect_ratio)
        images = []
        for img in response.images:
            images.append(base64.b64encode(img._image_bytes).decode())
        return {"images": images, "count": len(images)}
    except Exception as e:
        return {"error": str(e)}


def generate_nano_banana(prompt, aspect_ratio="16:9"):
    init_vertexai()
    try:
        from vertexai.generative_models import GenerativeModel, Part
        model = GenerativeModel("gemini-2.0-flash-exp")
        full_prompt = f"""Create a professional infographic/diagram for: {prompt}
        
Style: Clean, corporate, suitable for business presentations.
Format: {aspect_ratio} aspect ratio
Include: Clear labels, professional colors, visual hierarchy"""
        
        response = model.generate_content(full_prompt)
        return {"response": response.text, "type": "infographic_prompt"}
    except Exception as e:
        return {"error": str(e)}


def analyze_image(image_base64, prompt="Describe this image"):
    init_vertexai()
    try:
        from vertexai.generative_models import GenerativeModel, Part
        model = GenerativeModel("gemini-2.0-flash-exp")
        image_part = Part.from_data(base64.b64decode(image_base64), mime_type="image/png")
        response = model.generate_content([prompt, image_part])
        return {"analysis": response.text}
    except Exception as e:
        return {"error": str(e)}


def ocr_image(image_base64):
    try:
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=base64.b64decode(image_base64))
        response = client.text_detection(image=image)
        texts = response.text_annotations
        return {"text": texts[0].description if texts else "", "blocks": len(texts)}
    except Exception as e:
        return {"error": str(e)}


@app.route("/", methods=["GET"])
def index():
    conn = get_snowflake_connection()
    return jsonify({
        "service": "vertex-ai-mcp", "version": "1.0.0", "status": "healthy",
        "instance": "VERTEX", "platform": "Google Cloud",
        "role": "Image/Document AI",
        "specialization": ["Imagen 3", "Nano Banana", "Document AI", "Vision API", "OCR"],
        "sovereign_mind": True, "hive_mind_connected": conn is not None,
        "project": GOOGLE_PROJECT_ID
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "sovereign_mind": True})


@app.route("/mcp", methods=["POST", "OPTIONS"])
def mcp_endpoint():
    if request.method == "OPTIONS": return "", 200
    data = request.json
    method, params, req_id = data.get("method", ""), data.get("params", {}), data.get("id", 1)
    
    if method == "tools/list":
        tools = [
            {"name": "vertex_imagen_generate", "description": "Generate photorealistic images with Imagen 3", 
             "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"}, "aspect_ratio": {"type": "string"}, "number_of_images": {"type": "integer"}}, "required": ["prompt"]}},
            {"name": "vertex_nano_banana_generate", "description": "Generate infographics/diagrams with Nano Banana (Gemini)", 
             "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"}, "aspect_ratio": {"type": "string"}, "style": {"type": "string"}}, "required": ["prompt"]}},
            {"name": "vertex_gemini_analyze_image", "description": "Analyze image with Gemini", 
             "inputSchema": {"type": "object", "properties": {"image_base64": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["image_base64"]}},
            {"name": "vertex_vision_ocr", "description": "Extract text from image (OCR)", 
             "inputSchema": {"type": "object", "properties": {"image_base64": {"type": "string"}}, "required": ["image_base64"]}},
            {"name": "vertex_gemini_generate", "description": "Generate text with Gemini (Sovereign Mind)", 
             "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}}
        ]
        return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}})
    
    elif method == "tools/call":
        tool, args = params.get("name", ""), params.get("arguments", {})
        
        if tool == "vertex_imagen_generate":
            result = generate_imagen(args.get("prompt", ""), args.get("aspect_ratio", "1:1"), args.get("number_of_images", 1))
            return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(result)}]}})
        
        elif tool == "vertex_nano_banana_generate":
            result = generate_nano_banana(args.get("prompt", ""), args.get("aspect_ratio", "16:9"))
            return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(result)}]}})
        
        elif tool == "vertex_gemini_analyze_image":
            result = analyze_image(args.get("image_base64", ""), args.get("prompt", "Describe this image in detail"))
            return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(result)}]}})
        
        elif tool == "vertex_vision_ocr":
            result = ocr_image(args.get("image_base64", ""))
            return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(result)}]}})
        
        elif tool == "vertex_gemini_generate":
            init_vertexai()
            try:
                from vertexai.generative_models import GenerativeModel
                hive = query_hive_mind(3)
                system = f"{SOVEREIGN_MIND_PROMPT}\n\nHive Mind Context:\n{hive}"
                model = GenerativeModel("gemini-2.0-flash-exp", system_instruction=system)
                response = model.generate_content(args.get("prompt", ""))
                return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps({"response": response.text})}]}})
            except Exception as e:
                return jsonify({"jsonrpc": "2.0", "id": req_id, "error": {"code": -1, "message": str(e)}})
    
    return jsonify({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Not found"}})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting Vertex AI MCP (Sovereign Mind) on port {port}")
    app.run(host="0.0.0.0", port=port)
