FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV API_URL=http://localhost:8000
EXPOSE 7860
CMD ["./start.sh"]
