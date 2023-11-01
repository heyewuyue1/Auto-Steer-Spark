# Copyright 2022 Intel Corporation
# SPDX-License-Identifier: MIT
#
"""Base class for AutoSteer-G connectors"""
from typing import Type
from inference.preprocessing.preprocessor import QueryPlanPreprocessor


class DBConnector:
    """The basic connector class for a database under test"""

    class TimedResult:
        def __init__(self, result: str, time: int):
            self.result = result
            self.time_usecs = time

    def __init__(self):
        pass

    def connect(self) -> None:
        """Set up a new connection to the database"""
        raise NotImplementedError()

    def close(self) -> None:
        """Close the connection to the database"""
        raise NotImplementedError()

    def set_disabled_knobs(self, knobs: list) -> None:
        """Disable the provided list of knobs"""
        raise NotImplementedError()

    def get_knob(self, knob: str) -> bool:
        """Get current status of a knob"""
        raise NotImplementedError()

    def explain(self, query: str) -> str:
        """Explain a query and return its plan"""
        raise NotImplementedError()

    def execute(self, query: str) -> TimedResult:
        """Execute the query and return its timed result"""
        raise NotImplementedError()

    @staticmethod
    def get_plan_preprocessor() -> Type[QueryPlanPreprocessor]:
        """Return the type of the query plan preprocessor. The preprocessor transforms query plans into a form accepted by TCNNs."""
        raise NotImplementedError()

    @staticmethod
    def get_name() -> str:
        """Return the name of the database connector"""
        raise NotImplementedError()

    @staticmethod
    def get_knobs() -> list:
        """Static method returning all knobs defined for this connector"""
        raise NotImplementedError()
