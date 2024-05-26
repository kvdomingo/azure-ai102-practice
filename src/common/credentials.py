from azure.core.credentials import AzureKeyCredential

from src.settings import settings

multiservice_credential = AzureKeyCredential(settings.AZURE_AI_SUBSCRIPTION_KEY)

language_credential = AzureKeyCredential(settings.AZURE_AI_LANGUAGE_SUBSCRIPTION_KEY)
