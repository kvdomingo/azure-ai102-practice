import asyncio

from azure.ai.language.questionanswering.aio import QuestionAnsweringClient
from azure.core.credentials import AzureKeyCredential

from src.settings import settings


async def qna():
    credential = AzureKeyCredential(settings.AZURE_AI_LANGUAGE_SUBSCRIPTION_KEY)

    async with QuestionAnsweringClient(
        endpoint=str(settings.AZURE_AI_LANGUAGE_ENDPOINT), credential=credential
    ) as client:
        user_question = ""

        while user_question.lower() != "quit":
            user_question = input("\nQuestion: ")
            res = await client.get_answers(
                question=user_question,
                deployment_name=settings.AZURE_AI_LANGUAGE_DEPLOYMENT_NAME,
                project_name=settings.AZURE_AI_LANGUAGE_PROJECT_NAME,
            )

            for answer in res.answers:
                print(answer.answer)

        if user_question.lower() == "quit":
            print("Bye!")


if __name__ == "__main__":
    asyncio.run(qna())
