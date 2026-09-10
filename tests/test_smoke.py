from config import TOP_K


def test_config_module_is_importable_via_pythonpath():
    assert TOP_K == 5
