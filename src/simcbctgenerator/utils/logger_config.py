###############################################################################
# simcbctgenerator
#
# Copyright 2025 Lukas Zimmermann and Michael Rauter
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import logging
import logging.handlers
import time
from functools import wraps
import pprint
import os
import datetime

def log_config(logger, config_dict, message="Current configuration:"):
    logger.info(f"\n{message}\n" + pprint.pformat(config_dict, indent=2))

def log_time(logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            logger.info(f"Starting {func.__name__}")
            result = func(*args, **kwargs)

            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Finished {func.__name__} in {duration:.2f} seconds")

            return result
        return wrapper
    return decorator

def setup_logger():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
# Clear any existing handlers from the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatters and handlers
    formatter = logging.Formatter(
        fmt='[%(asctime)s %(name)s %(levelname)s] %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, f'{datetime.date.today()}.log'),
        maxBytes=1024*1024,
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Configure the root logger
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Return a logger for the calling module
    return logging.getLogger()
