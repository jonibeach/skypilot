""" Verda Cloud | Catalog

This module loads the service catalog file and can be used to
query instance types and pricing information for Verda Cloud.
"""

import io
import threading
import time
import typing
from typing import Dict, List, Optional, Set, Tuple, Union

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
_RETRY_SECONDS = 60

_lock = threading.Lock()
_client = verda.VerdaClient()
_offerings: Optional[Tuple[float, 'pd.DataFrame']] = None
_in_stock: Optional[Tuple[float, Dict[bool, Set[Tuple[str, str]]]]] = None


def _is_fresh(cached):
    return cached is not None and time.time() < cached[0]


def _fetch_offerings():
    buffer = io.StringIO()
    fetch_verda.write_catalog(buffer, _client.instance_types_get(),
                              _client.locations_get())
    buffer.seek(0)
    return pd.read_csv(buffer)


def _offerings_df():
    global _offerings
    if not verda.get_verda_configuration()[0]:
        return _hosted_df
    with _lock:
        if not _is_fresh(_offerings):
            try:
                offerings = _fetch_offerings()
                _offerings = (time.time() + _OFFERINGS_TTL_SECONDS, offerings)
            except Exception as e:  # pylint: disable=broad-except
                logger.debug(f'Failed to fetch the live Verda catalog: {e}')
                _, cached = _offerings or (0, _hosted_df)
                _offerings = (time.time() + _RETRY_SECONDS, cached)
        assert _offerings is not None
        return _offerings[1]


def _stock_for_mode(use_spot: bool):
    global _in_stock
    if not verda.get_verda_configuration()[0]:
        return set()
    with _lock:
        if not _is_fresh(_in_stock):
            try:
                stock = {
                    is_spot: _client.instance_availability_get(is_spot)
                    for is_spot in (False, True)
                }
                _in_stock = (time.time() + _AVAILABILITY_TTL_SECONDS, stock)
            except Exception as e:  # pylint: disable=broad-except
                logger.debug(f'Failed to fetch Verda availability: {e}')
                _, cached = _in_stock or (0, {False: set(), True: set()})
                _in_stock = (time.time() + _RETRY_SECONDS, cached)
        assert _in_stock is not None
        return _in_stock[1][use_spot]


def _in_stock_regions(instance_type: str, use_spot: bool):
    return {
        region for type_, region in _stock_for_mode(use_spot)
        if type_ == instance_type
    }


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


def _candidate_dfs(use_spot: bool):
    offerings = _offerings_df()
    price_column = 'SpotPrice' if use_spot else 'Price'
    offerings = offerings.dropna(subset=[price_column])
    stock = _stock_for_mode(use_spot)
    in_stock = offerings.loc[[(row.InstanceType, row.Region) in stock
                              for row in offerings.itertuples()]]
    return in_stock, offerings


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
    for candidates in _candidate_dfs(use_spot):
        instance_type = common.get_instance_type_for_cpus_mem_impl(
            candidates, cpus, memory, region, zone, use_spot, max_hourly_cost)
        if instance_type is not None:
            return instance_type
    return None


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
    result: Tuple[Optional[List[str]], List[str]] = ([], [])
    for candidates in _candidate_dfs(use_spot):
        result = common.get_instance_type_for_accelerator_impl(
            df=candidates,
            acc_name=acc_name,
            acc_count=acc_count,
            cpus=cpus,
            memory=memory,
            use_spot=use_spot,
            region=region,
            zone=zone,
            max_hourly_cost=max_hourly_cost)
        if result[0]:
            return result
    return result


def get_region_zones_for_instance_type(instance_type: str,
                                       use_spot: bool) -> List['cloud.Region']:
    df = _offerings_df()
    regions = common.get_region_zones(df[df['InstanceType'] == instance_type],
                                      use_spot)
    in_stock = _in_stock_regions(instance_type, use_spot)
    return sorted(regions, key=lambda region: region.name not in in_stock)


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
    return common.list_accelerators_impl('Verda', _offerings_df(), gpus_only,
                                         name_filter, region_filter,
                                         quantity_filter, case_sensitive,
                                         all_regions)
