FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Génère les données et entraîne le modèle au moment du build de l'image,
# pour livrer une image "prête à l'emploi" (modèle déjà entraîné à l'intérieur).
RUN python3 data/generate_data.py && python3 src/train.py

EXPOSE 5000
CMD ["python3", "src/predict.py"]
