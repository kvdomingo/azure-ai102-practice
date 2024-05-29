import asyncio

from loguru import logger
from openai import AsyncOpenAI

from src.settings import settings


async def access():
    async with AsyncOpenAI(
        api_key=settings.OPENAI_SECRET_KEY,
        organization=settings.OPENAI_ORGANIZATION_ID,
    ) as client:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Azure OpenAI?"},
            ],
        )
        generated = response.choices[0].message.content
        logger.info(generated)


if __name__ == "__main__":
    asyncio.run(access())
