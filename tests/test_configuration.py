import textwrap

from dea_burn_severity.configuration import DEFAULT_CONFIG_DICT, RuntimeConfig, build_runtime_config


def test_build_runtime_config_prefers_cli_over_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRE_DB_HOSTNAME", "db.example")
    monkeypatch.setenv("FIRE_DB_NAME", "fires")
    monkeypatch.setenv("FIRE_DB_USERNAME", "burns_user")
    monkeypatch.setenv("FIRE_DB_PASSWORD", "secret")
    monkeypatch.setenv("DB_PORT", "5439")

    yaml_cfg = tmp_path / "config.yaml"
    yaml_cfg.write_text(
        textwrap.dedent(
            """
            output_dir: /from_yaml
            upload_to_s3: false
            """
        )
    )

    runtime = build_runtime_config(
        {"output_dir": "cli_override", "upload_to_s3": True},
        config_path=yaml_cfg,
    )

    assert runtime.output_dir == "cli_override"
    assert runtime.upload_to_s3 is True
    assert runtime.db_host == "db.example"
    assert runtime.db_port == 5439
    assert isinstance(runtime.s2_measurements, tuple)
    assert len(runtime.s2_measurements) == len(DEFAULT_CONFIG_DICT["s2_measurements"])


def test_runtime_config_from_defaults_creates_expected_types():
    runtime = RuntimeConfig.from_dict(DEFAULT_CONFIG_DICT)

    assert isinstance(runtime.s2_products, tuple)
    assert isinstance(runtime.grass_classes, tuple)
    assert isinstance(runtime.resolution, tuple)
    assert runtime.resolution == (
        float(DEFAULT_CONFIG_DICT["resolution"][0]),
        float(DEFAULT_CONFIG_DICT["resolution"][1]),
    )
