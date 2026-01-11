import os

from celery import Celery

broker = os.environ["CELERY_BROKER_URL"]
backend = os.environ.get("CELERY_RESULT_BACKEND", broker)

celery_app = Celery("ocr", broker=broker, backend=backend)

celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_time_limit=120,
    task_soft_time_limit=110,
    result_expires=3600,  # 1 hour
)
