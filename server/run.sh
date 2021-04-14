# Runs the server app

#!/bin/bash

source config.sh

# Expose ports and run
if [ $ENVIRONMENT = "PROD" ]
then
  gunicorn3 --bind $HOST_PROD model_server:app
else
  gunicorn3 --bind $HOST_DEV model_server:app
fi
