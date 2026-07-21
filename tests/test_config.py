import os
import pytest
from litagent.config import load_config
from litagent.exceptions import ConfigError


@pytest.fixture
def temp_project(tmp_path):
    """创建模拟项目结构：config/default.yaml + pyproject.toml"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "default.yaml"
    config_file.write_text("""
agent:
  max_loops: 15
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
""")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'litagent'\n")
    return tmp_path, config_file


class TestLoadConfig:
    def test_load_default(self, temp_project):
        tmp, cfg_file = temp_project
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp))
            cfg = load_config(str(cfg_file))
            assert cfg.agent.max_loops == 15
            assert cfg.logging.level == "INFO"
        finally:
            os.chdir(old_cwd)

    def test_env_var_override_int(self, temp_project, monkeypatch):
        tmp, cfg_file = temp_project
        monkeypatch.setenv("LITAGENT_AGENT_MAX_LOOPS", "20")
        cfg = load_config(str(cfg_file))
        assert cfg.agent.max_loops == 20
        assert isinstance(cfg.agent.max_loops, int)

    def test_env_var_override_str(self, temp_project, monkeypatch):
        tmp, cfg_file = temp_project
        monkeypatch.setenv("LITAGENT_LOGGING_LEVEL", "DEBUG")
        cfg = load_config(str(cfg_file))
        assert cfg.logging.level == "DEBUG"

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_invalid_yaml(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("agent: not_a_dict\nlogging:\n  level: INFO\n  format: ''")
        with pytest.raises((ConfigError, TypeError)):
            load_config(str(bad_file))


class TestApplyEnvOverrides:
    """直接测 _apply_env_overrides 的类型转换与边界（纯函数，构造 dict 精确覆盖各分支）。"""

    def _run(self, data, env, monkeypatch):
        from litagent.config import _apply_env_overrides
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return _apply_env_overrides(data)

    def test_bool_override_does_not_crash(self, monkeypatch):
        """P0 回归：bool 值被 env 覆盖时不该走 int('true') 崩溃。"""
        data = {"observability": {"enabled": False}}
        out = self._run(data, {"LITAGENT_OBSERVABILITY_ENABLED": "true"}, monkeypatch)
        assert out["observability"]["enabled"] is True   # 正确解析为 bool

    def test_bool_falsy_values(self, monkeypatch):
        """bool 的假值：false/0/no/off 都解析为 False。"""
        data = {"observability": {"enabled": True}}
        out = self._run(data, {"LITAGENT_OBSERVABILITY_ENABLED": "false"}, monkeypatch)
        assert out["observability"]["enabled"] is False

    def test_unknown_key_is_skipped(self, monkeypatch):
        """P1 回归：section 存在但 key 是 typo → 跳过，不创建垃圾键。"""
        data = {"agent": {"max_loops": 15}}
        out = self._run(data, {"LITAGENT_AGENT_TYPO": "x"}, monkeypatch)
        assert "typo" not in out["agent"]                # 垃圾键未被创建
        assert out["agent"]["max_loops"] == 15           # 合法键不受影响

    def test_int_and_float_conversion(self, monkeypatch):
        data = {"agent": {"max_loops": 15}, "llm": {"temperature": 0.1}}
        out = self._run(data, {
            "LITAGENT_AGENT_MAX_LOOPS": "20",
            "LITAGENT_LLM_TEMPERATURE": "0.7",
        }, monkeypatch)
        assert out["agent"]["max_loops"] == 20 and isinstance(out["agent"]["max_loops"], int)
        assert out["llm"]["temperature"] == 0.7 and isinstance(out["llm"]["temperature"], float)

    def test_unknown_section_is_skipped(self, monkeypatch):
        """section 本身不存在 → 跳过，不崩。"""
        data = {"agent": {"max_loops": 15}}
        out = self._run(data, {"LITAGENT_NOSUCH_KEY": "x"}, monkeypatch)
        assert "nosuch" not in out
