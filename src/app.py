"""Inference de producción para las reseñas de WeLoveReviews."""

from pathlib import Path

import pandas as pd
from transformers import pipeline


# Usamos un commit concreto para que el resultado no dependa de cambios futuros
# en la rama principal del repositorio del modelo.
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"


def rating_to_sentiment(predicted_rating: int) -> str:
    """Convierte las estrellas predichas en una banda de sentimiento."""
    if predicted_rating <= 2:
        return "negativo"
    if predicted_rating == 3:
        return "neutral"
    return "positivo"


def run_inference(input_path: Path, output_path: Path) -> pd.DataFrame:
	"""Predice sentimiento para todas las reseñas y guarda el CSV enriquecido."""
	reviews = pd.read_csv(input_path)

	# El pipeline se crea una sola vez y después se reutiliza para todo el lote.
	classifier = pipeline(
		task="sentiment-analysis",
		model=MODEL_NAME,
		revision=MODEL_REVISION,
	)

	# Procesamos los textos juntos para evitar cargar el modelo una vez por fila.
	predictions = classifier(
		reviews["review_text"].tolist(),
		truncation=True,
		batch_size=16,
	)
	reviews["predicted_stars"] = [
		int(item["label"].split()[0]) for item in predictions
	]
	reviews["prediction_confidence"] = [item["score"] for item in predictions]
	reviews["sentiment_band"] = reviews["predicted_stars"].map(rating_to_sentiment)

	# Creamos la carpeta de salida si todavía no existe.
	output_path.parent.mkdir(parents=True, exist_ok=True)
	reviews.to_csv(output_path, index=False)
	return reviews


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "data" / "raw" / "reviews.csv"
    output_file = project_root / "data" / "processed" / "reviews_with_sentiment.csv"
    result = run_inference(input_file, output_file)
    print(f"Reseñas procesadas: {len(result)}")
    print(f"Archivo generado: {output_file}")
