
   
import datetime
import enum
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import colorlog
import numpy as np
import torch
from tensorboardX import SummaryWriter


class MetricType(enum.IntEnum):
    """
    Enum represents metric type
    """

    Number = 1
    Loss = 2
    Time = 3


class Metric(object):
    """
    Metric class for logger
    """

    mtype_list: List[int] = list(map(int, MetricType))

    def __init__(self, mtype: MetricType, priority: int):
        if mtype not in self.mtype_list:
            raise Exception("mtype is invalid, %s".format(self.mtype_list))

        self.mtype: MetricType = mtype
        self.params: Dict[str, Any] = {}
        self.priority: int = priority
        self.value: Any = 0


def new_logging_module(name: str, log_file: Path) -> logging.Logger:
    # specify format
    log_format: str = "%(asctime)s - " "%(message)s"
    bold_seq: str = "\033[1m"
    colorlog_format: str = f"{bold_seq} " "%(log_color)s " f"{log_format}"
    colorlog.basicConfig(format=colorlog_format)

    # init module
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # add handler to output file setting
    fh = logging.FileHandler(str(log_file))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


class Logger(object):
    """
    Logger for watchting some metrics involving training
    """

    def __init__(self, out_path: Path, tb_path: Path):
        # initialize logging module
        self._logger: logging.Logger = new_logging_module(__name__, out_path / "log")
        self.path = out_path

        # logging metrics
        self.metrics: OrderedDict[str, Metric] = OrderedDict()

        # tensorboard writer
        self.tf_writer: SummaryWriter = SummaryWriter(str(tb_path))

        # automatically add elapsed_time metric
        self.define("epoch", MetricType.Number, 5)
        self.define("iteration", MetricType.Number, 4)
        self.define("elapsed_time", MetricType.Time, -1)

    def define(self, name: str, mtype: MetricType, priority=0):
        metric: Metric = Metric(mtype, priority)
        if mtype == MetricType.Number:
            metric.value = 0
        elif mtype == MetricType.Loss:
            metric.value = []
        elif mtype == MetricType.Time:
            metric.value = 0
            metric.params["start_time"] = time.time()
        self.metrics[name] = metric

        self.metrics = OrderedDict(
            sorted(self.metrics.items(), key=lambda m: m[1].priority, reverse=True)
        )

    def metric_keys(self) -> List[str]:
        return list(self.metrics.keys())

    def clear(self):
        for _, metric in self.metrics.items():
            if metric.mtype != MetricType.Loss:
                metric.value = []

    def update(self, name: str, value: Any):
        m = self.metrics[name]
        if m.mtype == MetricType.Number:
            m.value = value
        elif m.mtype == MetricType.Loss:
            m.value.append(value)
        elif m.mtype == MetricType.Time:
            m.value = value - m.params["start_time"]

    def print_header(self):
        log_string = ""
        for name in self.metrics.keys():
            log_string += "{:>13} ".format(name)
        self._logger.info(log_string)

    def log(self):
        # display and save logs
        self.update("elapsed_time", time.time())

        log_strings: List[str] = []
        for k, m in self.metrics.items():
            if m.mtype == MetricType.Number:
                s = "{}".format(m.value)
            elif m.mtype == MetricType.Loss:
                if len(m.value) == 0:
                    raise Exception(f"Metric {k} has no values.")
                s = "{:0.3f}".format(sum(m.value) / len(m.value))
            elif m.mtype == MetricType.Time:
                _value = int(m.value)
                s = str(datetime.timedelta(seconds=_value))

            log_strings.append(s)

        log_string: str = ""
        for s in log_strings:
            log_string += "{:>13} ".format(s)

        self._logger.info(log_string)

    def log_tensorboard(self, x_axis_metric: str):
        # log MetricType.Loss metrics only
        if x_axis_metric not in self.metric_keys():
            raise Exception(f"No such metric: {x_axis_metric}")

        x_metric = self.metrics[x_axis_metric]
        if x_metric.mtype != MetricType.Number:
            raise Exception(f"Invalid metric type: {repr(x_metric.mtype)}")

        step = x_metric.value
        for name, metric in self.metrics.items():
            if metric.mtype != MetricType.Loss:
                continue

            if len(metric.value) == 0:
                raise Exception(f"Metric {name} has no values.")
            mean: float = sum(metric.value) / len(metric.value)
            self.tf_writer.add_scalar(name, mean, step)

    def tf_log_histgram(self, var, tag, step):
        var = var.clone().cpu().data.numpy()
        self.tf_writer.add_histogram(tag, var, step)

    def tf_log_image(self, x: torch.Tensor, step: int, tag: str):
        self.tf_writer.add_image(tag, x, step)

    def info(self, msg: str):
        self._logger.info(msg)

    def debug(self, msg: str):
        self._logger.debug(msg)


if __name__ == "__main__":
    import random
    from os.path import expanduser

    # init logger
    home = Path(expanduser("~"))
    out_path = home / "tmp/log"
    tfb_path = home / "tmp/log/tf"
    logger = Logger(out_path, tfb_path)
    print(logger.metric_keys())

    # add dummy metric
    logger.define("foo", MetricType.Number)
    logger.define("bar", MetricType.Loss)
    print(logger.metric_keys())

    logger.print_header()

    # update metric value and print log
    for i in range(10):
        logger.update("iteration", i % 2)
        logger.update("epoch", i // 2)
        logger.update("foo", random.randint(0, 100))
        logger.update("bar", random.randint(0, 10))
        logger.update("bar", random.randint(0, 10))
        logger.update("bar", random.randint(0, 10))
        time.sleep(1.0)
        logger.log()