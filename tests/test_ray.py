import naive
import numpy as np
import numpy.testing as npt
import pytest

try:  # pragma: no cover
    import ray

    RAY_IMPORTED = True
except ImportError:  # pragma: no cover
    RAY_IMPORTED = False
from stumpy import aamp_stimped, aamped, maamped, mstumped, stimped, stumped


@pytest.fixture(scope="module")
def ray_cluster():
    try:
        if not ray.is_initialized():
            ray.init()
        yield None
        if ray.is_initialized():
            ray.shutdown()
    except NameError:  # pragma: no cover
        # Ray not installed
        yield None


test_data = [
    (
        np.array([9, 8100, -60, 7], dtype=np.float64),
        np.array([584, -11, 23, 79, 1001, 0, -19], dtype=np.float64),
    ),
    (
        np.random.uniform(-1000, 1000, [8]).astype(np.float64),
        np.random.uniform(-1000, 1000, [64]).astype(np.float64),
    ),
]

test_mdata = [
    (np.array([[584, -11, 23, 79, 1001, 0, -19]], dtype=np.float64), 3),
    (np.random.uniform(-1000, 1000, [5, 20]).astype(np.float64), 5),
]

T = [
    np.array([584, -11, 23, 79, 1001, 0, -19], dtype=np.float64),
    np.random.uniform(-1000, 1000, [64]).astype(np.float64),
]


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stumped_ray_self_join(T_A, T_B, ray_cluster):
    if not RAY_IMPORTED:  # pragma: no cover
        pytest.skip("Skipping Test Ray Not Installed")

    m = 3
    zone = int(np.ceil(m / 4))
    ref_mp = naive.stump(T_B, m, exclusion_zone=zone)
    comp_mp = stumped(ray, T_B, m, ignore_trivial=True)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_aamped_ray_self_join(T_A, T_B, ray_cluster):
    if not RAY_IMPORTED:  # pragma: no cover
        pytest.skip("Skipping Test Ray Not Installed")

    m = 3
    for p in [1.0, 2.0, 3.0]:
        ref_mp = naive.aamp(T_B, m, p=p)
        comp_mp = aamped(ray, T_B, m, p=p)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_mdata)
def test_mstumped_ray(T, m, ray_cluster):
    if not RAY_IMPORTED:  # pragma: no cover
        pytest.skip("Skipping Test Ray Not Installed")

    excl_zone = int(np.ceil(m / 4))

    ref_P, ref_I = naive.mstump(T, m, excl_zone)
    comp_P, comp_I = mstumped(ray, T, m)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_mdata)
def test_maamped_ray(T, m, ray_cluster):
    if not RAY_IMPORTED:  # pragma: no cover
        pytest.skip("Skipping Test Ray Not Installed")

    excl_zone = int(np.ceil(m / 4))

    ref_P, ref_I = naive.maamp(T, m, excl_zone)
    comp_P, comp_I = maamped(ray, T, m)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T", T)
def test_stimped_ray(T, ray_cluster):
    if not RAY_IMPORTED:  # pragma: no cover
        pytest.skip("Skipping Test Ray Not Installed")

    threshold = 0.2
    min_m = 3
    n = T.shape[0] - min_m + 1

    pan = stimped(
        ray,
        T,
        min_m=min_m,
        max_m=None,
        step=1,
        # normalize=True,
    )

    for i in range(n):
        pan.update()

    ref_PAN = np.full((pan.M_.shape[0], T.shape[0]), fill_value=np.inf)

    for idx, m in enumerate(pan.M_[:n]):
        zone = int(np.ceil(m / 4))
        ref_mp = naive.stump(T, m, T_B=None, exclusion_zone=zone)
        ref_PAN[pan._bfs_indices[idx], : ref_mp.shape[0]] = ref_mp[:, 0]

    # Compare raw pan
    cmp_PAN = pan._PAN

    naive.replace_inf(ref_PAN)
    naive.replace_inf(cmp_PAN)

    npt.assert_almost_equal(ref_PAN, cmp_PAN)

    # Compare transformed pan
    cmp_pan = pan.PAN_
    ref_pan = naive.transform_pan(
        pan._PAN, pan._M, threshold, pan._bfs_indices, pan._n_processed
    )

    naive.replace_inf(ref_pan)
    naive.replace_inf(cmp_pan)

    npt.assert_almost_equal(ref_pan, cmp_pan)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T", T)
def test_aamp_stimped_ray(T, ray_cluster):
    if not RAY_IMPORTED:  # pragma: no cover
        pytest.skip("Skipping Test Ray Not Installed")

    threshold = 0.2
    min_m = 3
    n = T.shape[0] - min_m + 1

    pan = aamp_stimped(
        ray,
        T,
        min_m=min_m,
        max_m=None,
        step=1,
    )

    for i in range(n):
        pan.update()

    ref_PAN = np.full((pan.M_.shape[0], T.shape[0]), fill_value=np.inf)

    for idx, m in enumerate(pan.M_[:n]):
        zone = int(np.ceil(m / 4))
        ref_mp = naive.aamp(T, m, T_B=None, exclusion_zone=zone)
        ref_PAN[pan._bfs_indices[idx], : ref_mp.shape[0]] = ref_mp[:, 0]

    # Compare raw pan
    cmp_PAN = pan._PAN

    naive.replace_inf(ref_PAN)
    naive.replace_inf(cmp_PAN)

    npt.assert_almost_equal(ref_PAN, cmp_PAN)

    # Compare transformed pan
    cmp_pan = pan.PAN_
    ref_pan = naive.transform_pan(
        pan._PAN,
        pan._M,
        threshold,
        pan._bfs_indices,
        pan._n_processed,
        np.min(T),
        np.max(T),
    )

    naive.replace_inf(ref_pan)
    naive.replace_inf(cmp_pan)

    npt.assert_almost_equal(ref_pan, cmp_pan)
