import asyncio

from azure.ai.language.conversations.aio import ConversationAnalysisClient

from src.common.credentials import language_credential
from src.settings import settings


async def clu():
    project_name = "Clock"
    deployment_name = "production"

    async with ConversationAnalysisClient(
        endpoint=str(settings.AZURE_AI_LANGUAGE_ENDPOINT),
        credential=language_credential,
    ) as client:
        user_text = ""
        while user_text.lower() != "quit":
            user_text = input('\nEnter some text ("quit" to stop)\n')

            if user_text.lower() != "quit":
                result = await client.analyze_conversation(
                    task={
                        "kind": "Conversation",
                        "analysisInput": {
                            "conversationItem": {
                                "participantId": "1",
                                "id": "1",
                                "modality": "text",
                                "language": "en",
                                "text": user_text,
                            },
                            "isLoggingEnabled": False,
                        },
                        "parameters": {
                            "projectName": project_name,
                            "deploymentName": deployment_name,
                            "verbose": True,
                        },
                    }
                )

                entities = result["result"]["prediction"]["entities"]
                print("view top intent:")
                print(f'\ttop intent: {result["result"]["prediction"]["topIntent"]}')
                print(
                    f'\tcategory: {result["result"]["prediction"]["intents"][0]["category"]}'
                )
                print(
                    f'\tconfidence score: {result["result"]["prediction"]["intents"][0]["confidenceScore"]}\n'
                )

                print("view entities:")
                for entity in entities:
                    print(f'\tcategory: {entity["category"]}')
                    print(f'\ttext: {entity["text"]}')
                    print(f'\tconfidence score: {entity["confidenceScore"]}')

                print(f'query: {result["result"]["query"]}')


if __name__ == "__main__":
    asyncio.run(clu())
