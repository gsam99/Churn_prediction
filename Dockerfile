FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN chmod +x start.sh
EXPOSE 8000
EXPOSE 7860

CMD ["bash", "start.sh"]





