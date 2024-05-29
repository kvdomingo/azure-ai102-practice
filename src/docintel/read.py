import asyncio

from azure.ai.formrecognizer import AnalyzeResult
from azure.ai.formrecognizer.aio import DocumentAnalysisClient
from azure.core.polling import AsyncLROPoller
from loguru import logger

from src.common.credentials import multiservice_credential
from src.settings import settings


async def read():
    file_uri = "https://github.com/MicrosoftLearning/mslearn-ai-document-intelligence/blob/main/Labfiles/01-prebuild-models/sample-invoice/sample-invoice.pdf?raw=true"
    _file_locale = "en-US"
    file_model_id = "prebuilt-invoice"

    async with DocumentAnalysisClient(
        endpoint=str(settings.AZURE_AI_ENDPOINT), credential=multiservice_credential
    ) as client:
        poller: AsyncLROPoller[
            AnalyzeResult
        ] = await client.begin_classify_document_from_url(
            classifier_id=file_model_id, document_url=file_uri
        )
        receipts = await poller.result()

        for receipt in receipts.documents:
            vendor_name = receipt.fields.get("VendorName")
            logger.info(f"{vendor_name.value=}, {vendor_name.confidence=}")


if __name__ == "__main__":
    asyncio.run(read())
