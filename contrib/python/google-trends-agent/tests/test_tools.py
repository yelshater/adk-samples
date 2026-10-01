# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the BigQuery tool's result and error handling."""

import os
from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import BadRequest, Forbidden

# Importing app.tools runs app/__init__.py, which imports the agent and calls
# google.auth.default(); seed the same env as the runnability test.
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
os.environ.setdefault("MODEL_NAME_AGENT", "gemini-3.8-flash")
os.environ.setdefault("MODEL_NAME_TOOL", "gemini-3.8-flash")
with patch("google.auth.default", return_value=(MagicMock(), "test-project")):
    from app import tools


def _client_raising(exc: Exception) -> MagicMock:
    client = MagicMock()
    client.query.side_effect = exc
    return client


def _client_returning(rows: list[dict]) -> MagicMock:
    client = MagicMock()
    client.query.return_value.result.return_value = rows
    return client


@pytest.fixture
def mock_client():
    with patch.object(tools.bigquery, "Client") as client_cls:
        yield client_cls


def test_permission_denied_returns_hint(mock_client) -> None:
    mock_client.return_value = _client_raising(
        Forbidden(
            "Access Denied: User does not have bigquery.jobs.create",
            errors=[{"reason": "accessDenied"}],
        )
    )
    assert tools.execute_bigquery_sql("SELECT 1") == tools._PERMISSION_HINT


def test_non_iam_forbidden_returns_generic_error(mock_client) -> None:
    mock_client.return_value = _client_raising(
        Forbidden("Quota exceeded", errors=[{"reason": "quotaExceeded"}])
    )
    assert tools.execute_bigquery_sql("SELECT 1") == tools._GENERIC_ERROR


def test_other_error_returns_generic_error(mock_client) -> None:
    mock_client.return_value = _client_raising(BadRequest("Syntax error"))
    assert tools.execute_bigquery_sql("SELECT 1") == tools._GENERIC_ERROR


def test_no_rows(mock_client) -> None:
    mock_client.return_value = _client_returning([])
    assert (
        tools.execute_bigquery_sql("SELECT 1") == "Query returned no results."
    )


def test_rows_are_serialised(mock_client) -> None:
    mock_client.return_value = _client_returning([{"term": "x", "rank": 1}])
    assert (
        tools.execute_bigquery_sql("SELECT 1") == '[{"term": "x", "rank": 1}]'
    )
