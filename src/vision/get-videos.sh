#!/bin/bash

set -euxo pipefail

location="trial"

# Call the AccessToken method with the API key in the header to get an access token
raw_token=$(curl "https://api.videoindexer.ai/auth/$location/Accounts/$AZURE_AI_VIDEO_ACCOUNT_ID/AccessToken" -H "Ocp-Apim-Subscription-Key: $AZURE_AI_VIDEO_INDEXER_KEY")
token=$(eval echo "$raw_token")

# Use the access token to make an authenticated call to the Videos method to get a list of videos in the account
curl "https://api.videoindexer.ai/$location/Accounts/$AZURE_AI_VIDEO_ACCOUNT_ID/Videos?accessToken=$token" | jq
