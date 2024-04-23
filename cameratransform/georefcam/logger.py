import logging
import os
import sys

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s | %(pathname)s | %(funcName)s():%(lineno)s] %(levelname)s: %(message)s"
    )
)
handler.setLevel(logging.DEBUG)
logger = logging.getLogger(f"{__name__}")
logger.addHandler(handler)
logger.propagate = False


class ProfilingLogger:
    """A class used for code profiling logs. Allows to save messages to a logfile.

    Parameters
    ----------
    logging_dirpath : str | None, default: None
        If not None, profiling messages will be written to the "profiling.log"
        file in the provided logging directory.

    Attributes
    ----------
    logger : logging.Logger
        The logger used to write profiling messages.
    """

    def __init__(self, logging_dirpath: str | None = None):

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(
            logging.Formatter("\n==========\nPROFILING: %(message)s\n")
        )
        stdout_handler.setLevel(logging.INFO)
        self.logger = logging.getLogger(f"profiling_logger")
        self.logger.propagate = False
        self.logger.addHandler(stdout_handler)

        if logging_dirpath is not None:
            stdout_handler.setLevel(logging.INFO)
            file_handler = logging.FileHandler(
                os.path.join(logging_dirpath, "profiling.log")
            )
            file_handler.setFormatter(logging.Formatter("[%(funcName)s()] %(message)s"))
            self.logger.addHandler(file_handler)


def format_processing_speed_message(task_time: float, n_items: int) -> str:

    if task_time > n_items:
        speed_count_str = f"{task_time / n_items:.2f} s/it"
    else:
        speed_count_str = f"{n_items / task_time:.2f} it/s"

    return speed_count_str
