import asyncio

from aiofiles import open
from azure.ai.textanalytics.aio import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from loguru import logger

from src.settings import settings


async def analyze_text():
    credential = AzureKeyCredential(key=settings.AZURE_AI_SUBSCRIPTION_KEY)
    client = TextAnalyticsClient(
        endpoint=str(settings.AZURE_AI_ENDPOINT), credential=credential
    )

    reviews_path = settings.BASE_DIR / "src" / "language" / "reviews"
    for file_path in reviews_path.glob("*.txt"):
        logger.info(f"\n-------------\n{file_path}")

        async with open(file_path, "r") as f:
            text = await f.read()

        logger.info(text)

        # Get language
        detected_language = await client.detect_language(documents=[text])
        logger.info(f"{detected_language[0].primary_language.name=}")

        # Get sentiment
        sentiment_analysis = await client.analyze_sentiment(documents=[text])
        logger.info(f"{sentiment_analysis[0].sentiment=}")

        # Get key phrases
        key_phrases = await client.extract_key_phrases(documents=[text])
        if len(key_phrases) == 0:
            logger.warning("No key phrases detected.")
        else:
            for phrase in key_phrases[0].key_phrases:
                logger.info(phrase)

        # Get entities
        entities = await client.recognize_entities(documents=[text])
        if len(entities) == 0:
            logger.warning("No entities detected.")
        else:
            for entity in entities[0].entities:
                logger.info(f"{entity.text=} {entity.category=}")

        # Get linked entities
        linked_entities = await client.recognize_linked_entities(documents=[text])
        if len(linked_entities) == 0:
            logger.warning("No linked entities detected.")
        else:
            for linked_entity in linked_entities[0].entities:
                logger.info(f"{linked_entity.name=} {linked_entity.url=}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(analyze_text())
