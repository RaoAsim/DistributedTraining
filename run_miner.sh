#!/bin/bash

script="neurons/miner.py"
proc_name="distributed_training_miner" 
args=()

# Check if pm2 is installed
if ! command -v pm2 &> /dev/null
then
    echo "pm2 could not be found. To install see: https://pm2.keymetrics.io/docs/usage/quick-start/"
    exit 1
fi

# Loop through all command line arguments
while [[ $# -gt 0 ]]; do
  arg="$1"

  if [[ "$arg" == -* ]]; then
    if [[ $# -gt 1 && "$2" != -* ]]; then
      if [[ "$arg" == "--script" ]]; then
        script="$2"
        shift 2
      else
        args+=("'$arg'")
        args+=("'$2'")
        shift 2
      fi
    else
      args+=("'$arg'")
      shift
    fi
  else
    args+=("'$arg '")
    shift
  fi
done

echo "Running $script with the following pm2 config:"

joined_args=$(printf "%s," "${args[@]}")
joined_args=${joined_args%,}

echo "module.exports = {
  apps : [{
    name   : '$proc_name',
    script : '$script',
    interpreter: 'python3',
    min_uptime: '5m',
    max_restarts: 5,
    args: [$joined_args]
  }]
}" > app.config.js

cat app.config.js

# Start the miner process
pm2 start app.config.js --update-env
