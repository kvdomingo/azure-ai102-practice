import asyncio
import sys
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
from pydantic import BaseModel, FilePath

from src.settings import settings


class Args(BaseModel):
    path: FilePath


async def analyze_image(args: Args):
    async with aiofiles.open(args.path, "rb") as f:
        data = await f.read()

    client = ImageAnalysisClient(
        endpoint=settings.AZURE_AI_ENDPOINT,
        credential=AzureKeyCredential(settings.AZURE_AI_SUBSCRIPTION_KEY),
    )
    result = await client.analyze(
        image_data=data,
        visual_features=[
            VisualFeatures.TAGS,
            VisualFeatures.OBJECTS,
            VisualFeatures.CAPTION,
            VisualFeatures.DENSE_CAPTIONS,
            VisualFeatures.PEOPLE,
            VisualFeatures.SMART_CROPS,
            VisualFeatures.READ,
        ],
        gender_neutral_caption=True,
        language="en",
        smart_crops_aspect_ratios=[1.0, 4 / 3, 16 / 9, 16 / 10],
    )
    await client.close()

    PrettyPrinter().pprint(result.as_dict())

    if result.people is None:
        logger.info("No people detected in the image")
        return

    img = plt.imread(str(args.path))
    fig = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100))
    ax = fig.add_subplot(111)
    ax.imshow(img)

    for person in result.people.list:
        bb = person.bounding_box
        rect = patches.Rectangle(
            (bb.x, bb.y),
            bb.width,
            bb.height,
            linewidth=3,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(
        str(settings.BASE_DIR / "src" / "vision" / "outputs" / "detected_people.png"),
    )


if __name__ == "__main__":
    asyncio.run(analyze_image(Args(path=sys.argv[1])))
