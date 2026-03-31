#!/bin/sh
set -eu

ollama serve &
pid=$!

sleep "${OLLAMA_STARTUP_DELAY:-8}"

ready=0
for _ in 1 2 3 4 5; do
    if ollama list >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 2
done

if [ "$ready" -ne 1 ]; then
    echo "Ollama API did not become ready in time." >&2
    wait "$pid"
    exit 1
fi


# there's an issue with ollama pul where it will fail and not continue from last checkpoint.
# TODO: Best option for now is to pull and cancel every n-minutes until it finishes.
if ! ollama show "${OLLAMA_CHAT_MODEL:-llama3.2:3b}" >/dev/null 2>&1; then
    ollama pull "${OLLAMA_CHAT_MODEL:-llama3.2:3b}"
fi

if ! ollama show "${OLLAMA_EMBED_MODEL:-nomic-embed-text}" >/dev/null 2>&1; then
    ollama pull "${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
fi

wait "$pid"
