"""mmmac registries.

Every registry here is a child of the corresponding mmengine *root* registry
and lives in the ``mmmac`` scope. Components implemented in this repository
are registered into these registries; components from other OpenMMLab
libraries are referenced from configs with an explicit scope prefix, e.g.
``type='mmpretrain.ImageClassifier'`` (mmengine imports ``<scope>.registry``
on demand when it sees the prefix).

``locations`` lets mmengine lazily import the modules that contain the
registered classes, so configs work without manual imports in entry points.
"""
from mmengine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import EVALUATOR as MMENGINE_EVALUATOR
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import LOG_PROCESSORS as MMENGINE_LOG_PROCESSORS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODEL_WRAPPERS as MMENGINE_MODEL_WRAPPERS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import \
    OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmengine.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmengine.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mmengine.registry import \
    RUNNER_CONSTRUCTORS as MMENGINE_RUNNER_CONSTRUCTORS
from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine.registry import \
    WEIGHT_INITIALIZERS as MMENGINE_WEIGHT_INITIALIZERS
from mmengine.registry import Registry

__all__ = [
    'RUNNERS', 'RUNNER_CONSTRUCTORS', 'LOOPS', 'HOOKS', 'DATASETS',
    'DATA_SAMPLERS', 'TRANSFORMS', 'MODELS', 'MODEL_WRAPPERS',
    'WEIGHT_INITIALIZERS', 'OPTIMIZERS', 'OPTIM_WRAPPERS',
    'OPTIM_WRAPPER_CONSTRUCTORS', 'PARAM_SCHEDULERS', 'METRICS', 'EVALUATOR',
    'TASK_UTILS', 'VISUALIZERS', 'VISBACKENDS', 'LOG_PROCESSORS',
]

RUNNERS = Registry('runner', parent=MMENGINE_RUNNERS, scope='mmmac',
                   locations=['mmmac.engine'])
RUNNER_CONSTRUCTORS = Registry('runner constructor',
                               parent=MMENGINE_RUNNER_CONSTRUCTORS,
                               scope='mmmac', locations=['mmmac.engine'])
LOOPS = Registry('loop', parent=MMENGINE_LOOPS, scope='mmmac',
                 locations=['mmmac.engine'])
HOOKS = Registry('hook', parent=MMENGINE_HOOKS, scope='mmmac',
                 locations=['mmmac.engine'])

DATASETS = Registry('dataset', parent=MMENGINE_DATASETS, scope='mmmac',
                    locations=['mmmac.datasets'])
DATA_SAMPLERS = Registry('data sampler', parent=MMENGINE_DATA_SAMPLERS,
                         scope='mmmac', locations=['mmmac.datasets'])
TRANSFORMS = Registry('transform', parent=MMENGINE_TRANSFORMS, scope='mmmac',
                      locations=['mmmac.datasets'])

MODELS = Registry('model', parent=MMENGINE_MODELS, scope='mmmac',
                  locations=['mmmac.models'])
MODEL_WRAPPERS = Registry('model wrapper', parent=MMENGINE_MODEL_WRAPPERS,
                          scope='mmmac', locations=['mmmac.models'])
WEIGHT_INITIALIZERS = Registry('weight initializer',
                               parent=MMENGINE_WEIGHT_INITIALIZERS,
                               scope='mmmac', locations=['mmmac.models'])

OPTIMIZERS = Registry('optimizer', parent=MMENGINE_OPTIMIZERS, scope='mmmac',
                      locations=['mmmac.engine'])
OPTIM_WRAPPERS = Registry('optimizer wrapper', parent=MMENGINE_OPTIM_WRAPPERS,
                          scope='mmmac', locations=['mmmac.engine'])
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer wrapper constructor',
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS, scope='mmmac',
    locations=['mmmac.engine'])
PARAM_SCHEDULERS = Registry('parameter scheduler',
                            parent=MMENGINE_PARAM_SCHEDULERS, scope='mmmac',
                            locations=['mmmac.engine'])

METRICS = Registry('metric', parent=MMENGINE_METRICS, scope='mmmac',
                   locations=['mmmac.evaluation'])
EVALUATOR = Registry('evaluator', parent=MMENGINE_EVALUATOR, scope='mmmac',
                     locations=['mmmac.evaluation'])

TASK_UTILS = Registry('task util', parent=MMENGINE_TASK_UTILS, scope='mmmac',
                      locations=['mmmac.models'])

VISUALIZERS = Registry('visualizer', parent=MMENGINE_VISUALIZERS,
                       scope='mmmac', locations=['mmmac.visualization'])
VISBACKENDS = Registry('vis backend', parent=MMENGINE_VISBACKENDS,
                       scope='mmmac', locations=['mmmac.visualization'])

LOG_PROCESSORS = Registry('log processor', parent=MMENGINE_LOG_PROCESSORS,
                          scope='mmmac', locations=['mmmac.engine'])
