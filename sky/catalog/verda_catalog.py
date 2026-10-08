""" Verda Cloud | Catalog

This module loads the service catalog file and can be used to
query instance types and pricing information for Verda Cloud.
"""

import io
import threading
import time
import typing
from typing import Dict, List, Optional, Tuple, Union

from sky import sky_logging
from sky.adaptors import common as adaptors_common
from sky.adaptors import verda
from sky.catalog import common
from sky.catalog.data_fetchers import fetch_verda

if typing.TYPE_CHECKING:
    import pandas as pd

    from sky.clouds import cloud
else:
    pd = adaptors_common.LazyImport('pandas')

logger = sky_logging.init_logger(__name__)

# Verda Cloud has not set the update schedule for their catalog.
# We pull the catalog every 7 hours to make sure we have the
# latest information.
_PULL_FREQUENCY_HOURS = 7
_hosted_df = common.read_catalog('verda/vms.csv',
                                 pull_frequency_hours=_PULL_FREQUENCY_HOURS)
_OFFERINGS_TTL_SECONDS = 3600
_AVAILABILITY_TTL_SECONDS = 60

_lock = threading.Lock()
_client = verda.VerdaClient()
_offerings: Optional[Tuple[float, 'pd.DataFrame']] = None
_available: Optional[Tuple[float, 'pd.DataFrame']] = None


def _is_fresh(cached, ttl_seconds: int):
    return cached is not None and time.time() - cached[0] < ttl_seconds


def _fetch_offerings():
    buffer = io.StringIO()
    fetch_verda.write_catalog(buffer, _client.instance_types_get(),
                              _client.locations_get())
    buffer.seek(0)
    return pd.read_csv(buffer)


def _fetch_available(offerings: 'pd.DataFrame'):
    df = offerings.copy()
    keys = list(zip(df['InstanceType'], df['Region']))
    on_demand = _client.instance_availability_get(is_spot=False)
    spot = _client.instance_availability_get(is_spot=True)
    df.loc[[key not in on_demand for key in keys], 'Price'] = None
    df.loc[[key not in spot for key in keys], 'SpotPrice'] = None
    return df


def _offerings_df():
    global _offerings
    if not verda.get_verda_configuration()[0]:
        return _hosted_df
    with _lock:
        if not _is_fresh(_offerings, _OFFERINGS_TTL_SECONDS):
            try:
                _offerings = (time.time(), _fetch_offerings())
            except Exception as e:  # pylint: disable=broad-except
                logger.debug(f'Failed to fetch the live Verda catalog: {e}')
                return _hosted_df
        assert _offerings is not None
        return _offerings[1]


def _available_df():
    global _available
    offerings = _offerings_df()
    if offerings is _hosted_df:
        return _hosted_df
    with _lock:
        if not _is_fresh(_available, _AVAILABILITY_TTL_SECONDS):
            try:
                _available = (time.time(), _fetch_available(offerings))
            except Exception as e:  # pylint: disable=broad-except
                logger.debug(f'Failed to fetch Verda availability: {e}')
                return offerings
        assert _available is not None
        return _available[1]


def instance_type_exists(instance_type: str) -> bool:
    return common.instance_type_exists_impl(_offerings_df(), instance_type)


def validate_region_zone(
        region: Optional[str],
        zone: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    return common.validate_region_zone_impl('verda', _offerings_df(), region,
                                            zone)


def get_hourly_cost(instance_type: str,
                    use_spot: bool = False,
                    region: Optional[str] = None,
                    zone: Optional[str] = None) -> float:
    """Returns the cost, or the cheapest cost among all zones for spot."""
    return common.get_hourly_cost_impl(_offerings_df(), instance_type, use_spot,
                                       region, zone)


def get_vcpus_mem_from_instance_type(
        instance_type: str) -> Tuple[Optional[float], Optional[float]]:
    return common.get_vcpus_mem_from_instance_type_impl(_offerings_df(),
                                                        instance_type)


def get_default_instance_type(
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
        disk_tier: Optional[str] = None,
        local_disk: Optional[str] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        use_spot: bool = False,
        max_hourly_cost: Optional[float] = None) -> Optional[str]:
    del disk_tier, local_disk  # Verda Cloud does not support disk tiers.
    # NOTE: After expanding catalog to multiple entries, you may
    # want to specify a default instance type or family.
    return common.get_instance_type_for_cpus_mem_impl(_available_df(), cpus,
                                                      memory, region, zone,
                                                      use_spot, max_hourly_cost)


def get_accelerators_from_instance_type(
        instance_type: str) -> Optional[Dict[str, Union[int, float]]]:
    return common.get_accelerators_from_instance_type_impl(
        _offerings_df(), instance_type)


def get_instance_type_for_accelerator(
    acc_name: str,
    acc_count: int,
    cpus: Optional[str] = None,
    memory: Optional[str] = None,
    use_spot: bool = False,
    local_disk: Optional[str] = None,
    region: Optional[str] = None,
    zone: Optional[str] = None,
    max_hourly_cost: Optional[float] = None
) -> Tuple[Optional[List[str]], List[str]]:
    """Returns a list of instance types that have the given accelerator."""
    del local_disk  # Verda Cloud does not support local disk.
    return common.get_instance_type_for_accelerator_impl(
        df=_available_df(),
        acc_name=acc_name,
        acc_count=acc_count,
        cpus=cpus,
        memory=memory,
        use_spot=use_spot,
        region=region,
        zone=zone,
        max_hourly_cost=max_hourly_cost)


def get_region_zones_for_instance_type(instance_type: str,
                                       use_spot: bool) -> List['cloud.Region']:
    df = _available_df()
    df = df[df['InstanceType'] == instance_type]
    return common.get_region_zones(df, use_spot)


def list_accelerators(
        gpus_only: bool,
        name_filter: Optional[str],
        region_filter: Optional[str],
        quantity_filter: Optional[int],
        case_sensitive: bool = True,
        all_regions: bool = False,
        require_price: bool = True) -> Dict[str, List[common.InstanceTypeInfo]]:
    """Returns all instance types in Verda Cloud offering accelerators."""
    del require_price  # Unused.
    return common.list_accelerators_impl('Verda', _available_df(), gpus_only,
                                         name_filter, region_filter,
                                         quantity_filter, case_sensitive,
                                         all_regions)
