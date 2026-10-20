#!/bin/sh
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &
sleep 3
exec streamlit run app/ui/streamlit_app.py --server.port ${PORT:-7860} --server.address 0.0.0.0
