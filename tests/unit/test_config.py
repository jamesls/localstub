from __future__ import annotations

import json
import pytest
from pathlib import Path

from localstub.config import (
    ResponseConfig,
    load_config,
    _parse_config,
    _parse_response_spec,
)
from localstub.server import HTTPResponse


class TestParseResponseSpec:
    def test_parse_json_response_with_body_and_status(self) -> None:
        spec = {"type": "json", "body": {"key": "value"}, "status": 201}
        response = _parse_response_spec(spec)

        assert response.status == 201
        assert response.headers["Content-Type"] == "application/json"
        assert json.loads(response.body) == {"key": "value"}

    def test_parse_json_response_uses_defaults(self) -> None:
        spec = {"body": {"data": 123}}
        response = _parse_response_spec(spec)

        assert response.status == 200
        assert response.headers["Content-Type"] == "application/json"

    def test_parse_json_response_with_custom_headers(self) -> None:
        spec = {
            "type": "json",
            "body": {},
            "headers": {"X-Custom": "test"},
        }
        response = _parse_response_spec(spec)

        assert response.headers["X-Custom"] == "test"
        assert response.headers["Content-Type"] == "application/json"

    def test_parse_text_response_with_body(self) -> None:
        spec = {"type": "text", "body": "hello world", "status": 200}
        response = _parse_response_spec(spec)

        assert response.status == 200
        assert response.headers["Content-Type"] == "text/plain; charset=utf-8"
        assert response.body == b"hello world"

    def test_parse_text_response_with_none_body_returns_empty(self) -> None:
        spec = {"type": "text"}
        response = _parse_response_spec(spec)

        assert response.body == b""

    def test_parse_text_response_converts_non_string_to_string(self) -> None:
        spec = {"type": "text", "body": 12345}
        response = _parse_response_spec(spec)

        assert response.body == b"12345"

    def test_parse_raw_response_with_utf8_body(self) -> None:
        spec = {"type": "raw", "body": "raw data", "encoding": "utf-8"}
        response = _parse_response_spec(spec)

        assert response.body == b"raw data"

    def test_parse_raw_response_with_base64_body(self) -> None:
        # "hello" in base64
        spec = {"type": "raw", "body": "aGVsbG8=", "encoding": "base64"}
        response = _parse_response_spec(spec)

        assert response.body == b"hello"

    def test_parse_raw_response_with_none_body_returns_empty(self) -> None:
        spec = {"type": "raw"}
        response = _parse_response_spec(spec)

        assert response.body == b""

    def test_parse_raw_response_with_custom_headers(self) -> None:
        spec = {
            "type": "raw",
            "body": "data",
            "headers": {"Content-Type": "application/octet-stream"},
        }
        response = _parse_response_spec(spec)

        assert response.headers["Content-Type"] == "application/octet-stream"

    def test_parse_raw_response_base64_requires_string(self) -> None:
        spec = {"type": "raw", "body": 123, "encoding": "base64"}

        with pytest.raises(ValueError, match="base64 body must be a string"):
            _parse_response_spec(spec)

    def test_parse_unknown_type_raises_error(self) -> None:
        spec = {"type": "unknown", "body": "data"}

        with pytest.raises(ValueError, match="Unknown response type"):
            _parse_response_spec(spec)


class TestParseConfig:
    def test_parse_config_with_single_response(self) -> None:
        data = {"response": {"type": "json", "body": {"ok": True}}}
        config = _parse_config(data)

        assert config.single_response is not None
        assert config.response_sequence is None
        assert config.single_response.status == 200

    def test_parse_config_with_response_sequence(self) -> None:
        data = {
            "responses": [
                {"type": "json", "body": {"error": "fail"}, "status": 500},
                {"type": "json", "body": {"ok": True}, "status": 200},
            ]
        }
        config = _parse_config(data)

        assert config.single_response is None
        assert config.response_sequence is not None
        assert len(config.response_sequence) == 2
        assert config.response_sequence[0].status == 500
        assert config.response_sequence[1].status == 200

    def test_parse_config_missing_keys_raises_error(self) -> None:
        data = {"invalid": "data"}

        with pytest.raises(ValueError, match="must have either"):
            _parse_config(data)


class TestLoadConfig:
    def test_load_config_from_valid_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps({"response": {"type": "text", "body": "test"}})
        )

        config = load_config(config_file)

        assert config.single_response is not None
        assert config.single_response.body == b"test"

    def test_load_config_from_sequence_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps({
                "responses": [
                    {"status": 503},
                    {"status": 200, "body": {"success": True}},
                ]
            })
        )

        config = load_config(config_file)

        assert config.response_sequence is not None
        assert len(config.response_sequence) == 2

    def test_load_config_file_not_found_raises_error(
        self, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_config(config_file)

    def test_load_config_invalid_json_raises_error(
        self, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text("not valid json")

        with pytest.raises(json.JSONDecodeError):
            load_config(config_file)


class TestResponseConfig:
    def test_response_config_dataclass_attributes(self) -> None:
        response = HTTPResponse.json({"test": True})
        config = ResponseConfig(
            single_response=response,
            response_sequence=None,
        )

        assert config.single_response is response
        assert config.response_sequence is None

    def test_response_config_with_sequence(self) -> None:
        responses = [
            HTTPResponse.json({"n": 1}),
            HTTPResponse.json({"n": 2}),
        ]
        config = ResponseConfig(
            single_response=None,
            response_sequence=responses,
        )

        assert config.single_response is None
        assert config.response_sequence is responses
