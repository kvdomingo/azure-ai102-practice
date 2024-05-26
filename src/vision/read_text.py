import asyncio
from pprint import PrettyPrinter

import aiofiles
from azure.ai.vision.imageanalysis.aio import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from loguru import logger
from matplotlib import (
    patches,
    pyplot as plt,
)

from src.settings import settings


async def read_text():
    filepath = str(settings.BASE_DIR / "src" / "vision" / "text-images" / "Lincoln.jpg")
    async with aiofiles.open(filepath, "rb") as f:
        data = await f.read()

    client = ImageAnalysisClient(
        endpoint=settings.AZURE_AI_ENDPOINT,
        credential=AzureKeyCredential(settings.AZURE_AI_SUBSCRIPTION_KEY),
    )
    result = await client.analyze(
        image_data=data,
        visual_features=[VisualFeatures.READ],
        language="en",
    )
    await client.close()

    if result.read is None:
        logger.error("No text detected in the image")
        return

    img = plt.imread(filepath)
    fig = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100))
    ax = fig.add_subplot(111)
    ax.imshow(img)

    PrettyPrinter().pprint(result.read.as_dict())

    for line in result.read.blocks[0].lines:
        logger.info(line.text)
        poly = patches.Polygon(
            [(bp.x, bp.y) for bp in line.bounding_polygon],
            linewidth=3,
            edgecolor="g",
            facecolor="none",
        )
        ax.add_patch(poly)

    plt.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(
        str(settings.BASE_DIR / "src" / "vision" / "outputs" / "read_text.png"),
    )

    return result


if __name__ == "__main__":
    asyncio.run(read_text())
