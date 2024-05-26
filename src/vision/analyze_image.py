import asyncio
import sys
from pprint import PrettyPrinter

import aiofiles
from azure.ai.vision.imageanalysis.aio import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from pydantic import BaseModel, FilePath

from src.settings import settings


class Args(BaseModel):
    url: FilePath


async def main(args: Args):
    async with aiofiles.open(args.url, "rb") as f:
        data = await f.read()

    client = ImageAnalysisClient(
        endpoint=settings.AZURE_AI_ENDPOINT,
        credential=AzureKeyCredential(settings.AZURE_AI_SUBSCRIPTION_KEY),
    )

    result = await client.analyze(
        image_data=data,
        visual_features=[
            VisualFeatures.CAPTION,
            VisualFeatures.DENSE_CAPTIONS,
            VisualFeatures.READ,
        ],
        gender_neutral_caption=True,
        language="en",
    )

    await client.close()
    PrettyPrinter().pprint(result.as_dict())
    return result


if __name__ == "__main__":
    asyncio.run(main(Args(url=sys.argv[1])))
