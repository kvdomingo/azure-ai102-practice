#!/bin/bash

set -euxo pipefail

sed -i "s/<storageAccount>/$STORAGE_ACCOUNT_NAME/g" ./training-images/training_labels.json
