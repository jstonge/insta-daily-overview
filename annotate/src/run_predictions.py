"""Run a predictor over Label Studio tasks and upload predictions.

Typical flow:
    1. Pull all tasks from project PROJECT_ID.
    2. Skip tasks that already have a prediction tagged with this model's
       `model_version` (idempotent re-runs, safe to mix predictors).
    3. Resolve each task's $image URL back to an on-disk path under
       LOCAL_FILES_ROOT (the Label Studio server's LOCAL_FILES_DOCUMENT_ROOT),
       load the image from disk, run the predictor, upload the prediction.

Designed to run both locally (prototyping) and on the HPC, where
/mnt/compethicslab/cc/dailyoverview is mounted and GPUs are available.

Example:
    uv run python annotate/src/run_predictions.py --predictor clip_zeroshot
    uv run python annotate/src/run_predictions.py --predictor clip_zeroshot --limit 5 --dry-run
"""
import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Make sibling modules importable without packaging gymnastics.
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image
from label_studio_sdk.label_interface.objects import PredictionValue

from labelstudio import client, PROJECT_ID
from predictors.base import Predictor


# Root that the Label Studio server resolves `?d=…` against. Override with
# LOCAL_FILES_ROOT in .env when the mount point differs (e.g. on HPC).
LOCAL_FILES_ROOT = Path(os.getenv("LOCAL_FILES_ROOT", "/mnt/compethicslab/cc"))

# Name of the <Choices> control in the project labeling config.
CHOICES_CONTROL = "choice"


def load_predictor(name: str) -> Predictor:
    if name == "clip_zeroshot":
        from predictors.clip_zeroshot import ClipZeroShotPredictor
        return ClipZeroShotPredictor()
    raise ValueError(f"Unknown predictor: {name!r}")


def task_image_path(task) -> Path:
    """Resolve a task's $image URL to an on-disk path under LOCAL_FILES_ROOT.

    URLs look like `/data/local-files/?d=dailyoverview/posts/202506/<id>.jpg`;
    the `d` query param is relative to the server's document root.
    """
    url = task.data["image"]
    parsed = urlparse(url)
    d = parse_qs(parsed.query).get("d", [""])[0]
    return LOCAL_FILES_ROOT / d


def already_predicted(task, model_version: str) -> bool:
    for p in getattr(task, "predictions", None) or []:
        mv = p.get("model_version") if isinstance(p, dict) else getattr(p, "model_version", None)
        if mv == model_version:
            return True
    return False


def batched(iterable, n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", required=True, help="Predictor name, e.g. clip_zeroshot")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="Max tasks to process")
    ap.add_argument("--dry-run", action="store_true", help="Don't upload predictions")
    args = ap.parse_args()

    predictor = load_predictor(args.predictor)
    print(f"Loaded predictor: {predictor.model_version}")

    label_interface = client.projects.get(id=PROJECT_ID).get_label_interface()
    control = label_interface.get_control(CHOICES_CONTROL)

    # Collect tasks that still need predictions from this model.
    pending = []
    for task in client.tasks.list(project=PROJECT_ID):
        if already_predicted(task, predictor.model_version):
            continue
        pending.append(task)
        if args.limit and len(pending) >= args.limit:
            break

    print(f"{len(pending)} task(s) to predict")

    uploaded = 0
    for batch in batched(pending, args.batch_size):
        images: list[Image.Image] = []
        valid_tasks = []
        for task in batch:
            path = task_image_path(task)
            if not path.exists():
                print(f"  skip (missing file): {path}")
                continue
            try:
                images.append(Image.open(path).convert("RGB"))
                valid_tasks.append(task)
            except Exception as e:
                print(f"  skip (load error): {path} -> {e}")

        if not images:
            continue

        results = predictor.predict_batch(images)

        for task, choices in zip(valid_tasks, results):
            prediction = PredictionValue(
                model_version=predictor.model_version,
                result=[control.label(choices)],
            )
            if args.dry_run:
                print(f"[dry-run] task={task.id} -> {choices}")
            else:
                client.predictions.create(task=task.id, **prediction.model_dump())
            uploaded += 1

    print(f"Done. {uploaded} prediction(s) {'previewed' if args.dry_run else 'uploaded'}.")


if __name__ == "__main__":
    main()
