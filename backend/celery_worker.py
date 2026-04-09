from tasks import celery


celery.conf.update(
    task_routes={
        "tasks.process_creative": {"queue": "creatives"},
    },
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

app = celery
