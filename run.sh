#!/bin/bash

BASE_DIR="/home/ec2-user/mPLUG-Owl-RAG"
STREAMLIT_DIR="${BASE_DIR}/src/client"
RAG_DIR="${BASE_DIR}/src/services/rag"
CHAT_DIR="${BASE_DIR}/src/services/chat"
ORIGINAL_DIR=$(pwd)

start_service() {
    local dir=$1
    local command=$2
    local service_name=$3
    
    echo "Starting ${service_name}..."
    cd "$dir" || exit 1
    echo "Running in $(pwd)"
    eval "$command"
}

mkfifo /tmp/streamlit_output /tmp/rag_output /tmp/chat_output

(start_service "$STREAMLIT_DIR" "streamlit run 1_💬_Chat.py" "Streamlit" | tee /tmp/streamlit_output) &
STREAMLIT_PID=$!

(start_service "$RAG_DIR" "uvicorn app:app --port 8080 --reload" "RAG service" | tee /tmp/rag_output) &
RAG_PID=$!

(start_service "$CHAT_DIR" "uvicorn app:app --port 8000 --reload" "Chat service" | tee /tmp/chat_output) &
CHAT_PID=$!

cd "$ORIGINAL_DIR"

trap "echo 'Stopping all services...'; kill $STREAMLIT_PID $RAG_PID $CHAT_PID; rm -f /tmp/streamlit_output /tmp/rag_output /tmp/chat_output; exit" SIGINT SIGTERM

echo "All services started. Press Ctrl+C to terminate all."

cat /tmp/streamlit_output &
cat /tmp/rag_output &
cat /tmp/chat_output &

wait $STREAMLIT_PID $RAG_PID $CHAT_PID