import logging
import os
import datetime

class DriveSafeFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()
        os.fsync(self.stream.fileno())

def setup_logger(args):
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if not any(isinstance(h, DriveSafeFileHandler) for h in logger.handlers):
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        os.makedirs(args.logdir, exist_ok=True)
        fh_path = os.path.join(args.logdir, f"log_{ts}.log")
        fh = DriveSafeFileHandler(fh_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
