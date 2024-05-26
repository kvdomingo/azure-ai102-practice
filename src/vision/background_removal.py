import asyncio

import aiofiles
from aiohttp import ClientSession
from loguru import logger

from src.settings import settings


async def background_removal():
    api_version = "2023-02-01-preview"
    mode = "backgroundRemoval"

    image_url = "https://github.com/MicrosoftLearning/mslearn-ai-vision/blob/main/Labfiles/01-analyze-images/Python/image-analysis/images/person.jpg?raw=true"
    body = {"url": image_url}
    headers = {
        "Ocp-Apim-Subscription-Key": settings.AZURE_AI_SUBSCRIPTION_KEY,
        "Content-Type": "application/json",
    }
    params = {
        "api-version": api_version,
        "mode": mode,
    }

    async with ClientSession(base_url=str(settings.AZURE_AI_ENDPOINT)) as session:
        res = await session.post(
            "/computervision/imageanalysis:segment",
            params=params,
            headers=headers,
            json=body,
        )
        if not res.ok:
            logger.error(await res.json())

        img = await res.content.read()

    async with aiofiles.open(
        settings.BASE_DIR / "src" / "vision" / "outputs" / "background_removed.png",
        "wb",
    ) as f:
        await f.write(img)

    return img


if __name__ == "__main__":
    asyncio.run(background_removal())
