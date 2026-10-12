import os
import io
import base64
import urllib.parse
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from PIL import Image
from rembg import remove
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

app = FastAPI(
    title="AI Image Studio API",
    description="API para geração de imagens (Flux AI + Gemini Fallback) e remoção de fundo (rembg U2Net)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateImageRequest(BaseModel):
    prompt: str
    aspect_ratio: str = "1:1"  # 1:1, 3:4, 4:3, 9:16, 16:9
    engine: str = "flux"       # "flux" ou "gemini"

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "AI Image Studio API",
        "gemini_configured": bool(GEMINI_API_KEY),
        "engines_available": ["flux", "gemini"]
    }

async def generate_with_gemini(prompt: str, key: str):
    """Tenta gerar via Google GenAI / Imagen 3"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    result = client.models.generate_images(
        model='imagen-3.0-generate-002',
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="1:1",
            output_mime_type="image/jpeg"
        )
    )
    images_b64 = []
    for gen_img in result.generated_images:
        b64 = base64.b64encode(gen_img.image.image_bytes).decode('utf-8')
        images_b64.append(f"data:image/jpeg;base64,{b64}")
    return images_b64

async def generate_with_flux(prompt: str, aspect_ratio: str = "1:1"):
    """Gera com modelo Flux via Pollinations ultra rápido em alta definição"""
    dims = {
        "1:1": (1024, 1024),
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "4:3": (1024, 768),
        "3:4": (768, 1024)
    }
    width, height = dims.get(aspect_ratio, (1024, 1024))
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true&model=flux"

    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await client.get(url)
        if res.status_code != 200:
            raise Exception(f"Flux generator retornou status {res.status_code}")
        
        b64 = base64.b64encode(res.content).decode("utf-8")
        return [f"data:image/jpeg;base64,{b64}"]

@app.post("/api/generate")
async def generate_image(req: GenerateImageRequest):
    """
    Gera uma imagem a partir de um prompt.
    Usa o modelo de alta definição Flux com fallback inteligente para Gemini.
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="O prompt não pode ser vazio.")

    # Se o usuário escolheu expressamente Gemini ou por padrão tenta Flux
    if req.engine == "gemini" and GEMINI_API_KEY:
        try:
            images = await generate_with_gemini(req.prompt, GEMINI_API_KEY)
            return {"success": True, "prompt": req.prompt, "engine": "gemini", "images": images}
        except Exception as e:
            # Fallback transparente para Flux caso quota esteja esgotada
            print(f"Gemini quota excedida ou indisponível ({str(e)}). Usando Flux como fallback.")
    
    try:
        images = await generate_with_flux(req.prompt, req.aspect_ratio)
        return {
            "success": True,
            "prompt": req.prompt,
            "engine": "flux",
            "images": images
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar imagem: {str(e)}")

@app.post("/api/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    """
    Remove o fundo de uma imagem enviada usando a rede neural rembg (U2Net).
    Retorna o PNG com fundo transparente.
    """
    try:
        input_bytes = await file.read()
        input_image = Image.open(io.BytesIO(input_bytes))

        # Executa remoção de fundo com rembg
        output_image = remove(input_image)

        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        # Retorna a imagem em PNG com headers para download
        return Response(
            content=output_buffer.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": f'attachment; filename="no-bg-{file.filename.split(".")[0]}.png"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao remover fundo: {str(e)}")

@app.post("/api/remove-bg/b64")
async def remove_background_base64(file: UploadFile = File(...)):
    """
    Remove o fundo e devolve em Base64 para exibição imediata em apps móveis (Compose/Glide).
    """
    try:
        input_bytes = await file.read()
        input_image = Image.open(io.BytesIO(input_bytes))

        output_image = remove(input_image)

        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        b64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        return {
            "success": True,
            "image": f"data:image/png;base64,{b64}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao remover fundo: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
