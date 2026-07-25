from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import Registry

from mmmac import registry


def test_all_registries_are_mmmac_scoped():
    for name in registry.__all__:
        reg = getattr(registry, name)
        assert isinstance(reg, Registry), name
        assert reg.scope == 'mmmac', name


def test_models_registry_parent_is_mmengine_root():
    assert registry.MODELS.parent is MMENGINE_MODELS


def test_cross_scope_lookup_from_configs_works():
    # configs reference upstream components as e.g. 'mmpretrain.LeNet5'
    cls = registry.MODELS.get('mmpretrain.LeNet5')
    assert cls is not None
    assert cls.__name__ == 'LeNet5'
