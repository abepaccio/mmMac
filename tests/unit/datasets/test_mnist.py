from mmpretrain.datasets import MNIST as MMPretrainMNIST

from mmmac.datasets import MNIST
from mmmac.registry import DATASETS


def test_registered_in_mmmac_scope():
    assert DATASETS.get('MNIST') is MNIST


def test_is_a_drop_in_mmpretrain_subclass():
    assert issubclass(MNIST, MMPretrainMNIST)


def test_uses_reachable_download_mirror():
    assert MNIST.url_prefix == 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    # inherited file list / checksums must stay untouched
    assert MNIST.train_list == MMPretrainMNIST.train_list
    assert MNIST.test_list == MMPretrainMNIST.test_list


def test_lazy_init_constructs_without_data(tmp_path):
    ds = MNIST(data_root=str(tmp_path), split='train', download=False,
               lazy_init=True)
    assert ds is not None


def test_split_drives_file_selection(tmp_path):
    # upstream keys train/test file selection off `test_mode` and ignores
    # `split`; our subclass derives test_mode from split (unless given)
    train = MNIST(data_root=str(tmp_path), split='train', download=False,
                  lazy_init=True)
    test = MNIST(data_root=str(tmp_path), split='test', download=False,
                 lazy_init=True)
    assert train.test_mode is False
    assert test.test_mode is True


def test_explicit_test_mode_wins_over_split(tmp_path):
    # e.g. overfit configs validate on the training split
    ds = MNIST(data_root=str(tmp_path), split='train', test_mode=True,
               download=False, lazy_init=True)
    assert ds.test_mode is True
