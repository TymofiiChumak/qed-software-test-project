import json
import traceback
import mlflow


from contextlib import contextmanager


@contextmanager
def log_errors():
    try:
        yield
    except Exception as error:
        mlflow.log_text(
            json.dumps({
                "error": str(error),
                "traceback": traceback.format_exc(),
            }),
            "errors.json"
        )
        raise
