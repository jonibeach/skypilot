"""Verda Cloud (formerly DataCrunch) instance provisioning."""

import time
from typing import Any, Dict, List, Optional, Tuple

from sky import exceptions
from sky import sky_logging
from sky.adaptors.verda import Instance
from sky.adaptors.verda import InstanceStatus
from sky.adaptors.verda import VerdaClient
from sky.adaptors.verda import VerdaException
from sky.clouds.verda import VERDA_DEFAULT_IMAGE
from sky.provision import common
from sky.resources import DEFAULT_DISK_SIZE_GB
from sky.utils import common_utils
from sky.utils import status_lib
from sky.utils import ux_utils

# The maximum number of times to poll for the status of an operation.
POLL_INTERVAL = 5
MAX_POLLS = 60 // POLL_INTERVAL
# Terminating instances can take several minutes, so we increase the timeout
MAX_POLLS_FOR_UP_OR_TERMINATE = MAX_POLLS * 16

logger = sky_logging.init_logger(__name__)

# SSH connection readiness polling constants
SSH_CONN_MAX_RETRIES = 6
SSH_CONN_RETRY_INTERVAL_SECONDS = 10

verda = VerdaClient()

# https://api.verda.com/v1/docs#tag/instances/GET/v1/instances
_PENDING_STATUSES = [
    InstanceStatus.NEW,
    InstanceStatus.ORDERED,
    InstanceStatus.VALIDATING,
    InstanceStatus.PROVISIONING,
]
_FAILED_STATUSES = [
    InstanceStatus.ERROR,
    InstanceStatus.NO_CAPACITY,
    InstanceStatus.INSTALLATION_FAILED,
    InstanceStatus.DISCONTINUED,
    InstanceStatus.NOTFOUND,
]
_STATUS_MAP = {
    InstanceStatus.NEW: status_lib.ClusterStatus.INIT,
    InstanceStatus.ORDERED: status_lib.ClusterStatus.INIT,
    InstanceStatus.VALIDATING: status_lib.ClusterStatus.INIT,
    InstanceStatus.PROVISIONING: status_lib.ClusterStatus.INIT,
    InstanceStatus.RESTORING: status_lib.ClusterStatus.INIT,
    InstanceStatus.UNKNOWN: status_lib.ClusterStatus.INIT,
    InstanceStatus.ERROR: status_lib.ClusterStatus.INIT,
    InstanceStatus.NO_CAPACITY: status_lib.ClusterStatus.INIT,
    InstanceStatus.INSTALLATION_FAILED: status_lib.ClusterStatus.INIT,
    InstanceStatus.RUNNING: status_lib.ClusterStatus.UP,
    # Shut down but not deleted. Verda keeps billing for it.
    InstanceStatus.OFFLINE: status_lib.ClusterStatus.STOPPED,
    InstanceStatus.STARTING_HIBERNATION: status_lib.ClusterStatus.STOPPED,
    InstanceStatus.HIBERNATING: status_lib.ClusterStatus.STOPPED,
    # Already terminated - should be filtered out
    InstanceStatus.DISCONTINUED: None,
    # Being deleted - should be filtered out
    InstanceStatus.DELETING: None,
    InstanceStatus.NOTFOUND: None,
}


def _filter_instances(
        cluster_name_on_cloud: str,
        status_filters: Optional[List[str]] = None) -> Dict[str, Instance]:
    instances = verda.instances_get()
    hostnames = {
        f'{cluster_name_on_cloud}-head', f'{cluster_name_on_cloud}-worker'
    }
    filtered_instances = {}
    for instance in instances:
        instance_id = instance.instance_id
        instance_name = instance.hostname
        # Filter by cluster name
        if instance_name not in hostnames:
            continue
        # Filter by status if status_filters is provided
        if status_filters is not None and instance.status not in status_filters:
            continue
        filtered_instances[instance_id] = instance
    return filtered_instances


def _get_instance_info(instance_id: str) -> Instance:
    return verda.instance_get(instance_id)


def _get_head_instance_id(instances: Dict[str, Instance]) -> Optional[str]:
    head_instance_id = None
    for inst_id, inst in instances.items():
        if inst.hostname.endswith('-head'):
            head_instance_id = inst_id
            break
    return head_instance_id


# https://api.verda.com/v1/docs#tag/os-images/GET/v1/images
def _fallback_image(instance_type: str):
    # /images lists the newest CUDA image first. Some types have no CUDA
    # image at all (1V100.6V only has 24.04.base, jupyter and 26.04.base), and
    # jupyter is then the only image with NVIDIA drivers. The
    # ubuntu-24.04-cuda-* names match the style of VERDA_DEFAULT_IMAGE, in
    # case /images returns them.
    images = verda.images_get(instance_type)
    for matches in (
            lambda i: _is_cuda_image(i) and i.endswith(('.docker', '-docker')),
            _is_cuda_image,
            lambda i: i == 'jupyter',
    ):
        image = next((i for i in images if matches(i)), None)
        if image is not None:
            logger.info(f'Default image is not valid for {instance_type}, '
                        f'using {image}.')
            return image
    raise exceptions.ResourcesUnavailableError(
        f'No supported Verda image for {instance_type}.')


def _is_cuda_image(image: str):
    return image.startswith(('24.04.cuda', 'ubuntu-24.04-cuda-'))


def find_ssh_key_id(public_key: str):
    ssh_keys = verda.ssh_keys_get()
    for ssh_key in ssh_keys:
        if ssh_key.public_key == public_key:
            return ssh_key.id
    raise Exception(
        f'SSH key {public_key} not found in your Verda Cloud account')


def run_instances(
    region: str,
    cluster_name: str,
    cluster_name_on_cloud: str,
    config: common.ProvisionConfig,
) -> common.ProvisionRecord:
    """Runs instances for the given cluster."""
    del cluster_name  # unused
    newly_started_instances = _filter_instances(cluster_name_on_cloud,
                                                _PENDING_STATUSES)
    for _ in range(MAX_POLLS_FOR_UP_OR_TERMINATE):
        instances = _filter_instances(cluster_name_on_cloud, _PENDING_STATUSES)
        if not instances:
            break
        instance_statuses = [instance.status for instance in instances.values()]
        logger.info(f'Waiting for {len(instances)} instances to be ready: '
                    f'{instance_statuses}')
        time.sleep(POLL_INTERVAL)
    else:
        raise exceptions.ResourcesUnavailableError(
            f'Timed out waiting for pending Verda instances in '
            f'cluster {cluster_name_on_cloud}.')

    exist_instances = _filter_instances(cluster_name_on_cloud,
                                        status_filters=[InstanceStatus.RUNNING])
    head_instance_id = _get_head_instance_id(exist_instances)
    to_start_count = config.count - len(exist_instances)
    if to_start_count < 0:
        raise RuntimeError(
            f'Cluster {cluster_name_on_cloud} already has '
            f'{len(exist_instances)} nodes, but {config.count} are required.')
    if to_start_count == 0:
        if head_instance_id is None:
            head_instance_id = list(exist_instances.keys())[0]
        assert head_instance_id is not None, (
            'head_instance_id should not be None')
        logger.info(f'Cluster {cluster_name_on_cloud} already has '
                    f'{len(exist_instances)} nodes, no need to start more.')
        return common.ProvisionRecord(
            provider_name='verda',
            cluster_name=cluster_name_on_cloud,
            region=region,
            zone=None,
            head_instance_id=head_instance_id,
            resumed_instance_ids=list(newly_started_instances.keys()),
            created_instance_ids=[],
        )

    # Get image from node_config (populated from template)
    image = config.node_config.get('ImageId', VERDA_DEFAULT_IMAGE)
    created_instance_ids = []
    for _ in range(to_start_count):
        node_type = 'head' if head_instance_id is None else 'worker'
        try:
            # Extract vCPUs and memory from instance type
            # Format: instance_type__vcpus__memory[__SPOT]
            instance_type = config.node_config['InstanceType']
            disk_size = config.node_config.get('DiskSize', DEFAULT_DISK_SIZE_GB)
            # Preemptible - fancy way to call it a spot instance
            is_spot = config.node_config.get('Preemptible', None)

            ssh_public_key = config.node_config['PublicKey']
            if ssh_public_key is None:
                raise ValueError('SSH public key is not set in the node config')
            ssh_key_id = find_ssh_key_id(ssh_public_key)

            instance_data = {
                'instance_type': instance_type,
                'hostname': f'{cluster_name_on_cloud}-{node_type}',
                'location_code': region,
                'is_spot': is_spot if is_spot is not None else False,
                'contract': 'PAY_AS_YOU_GO' if not is_spot else 'SPOT',
                'image': image,
                'description': 'Created by SkyPilot',
                'ssh_key_ids': [ssh_key_id],
                'os_volume': {
                    'name': f'{cluster_name_on_cloud}-{node_type}',
                    'size': disk_size,
                }
            }
            # https://api.verda.com/v1/docs#tag/instances/POST/v1/instances
            # https://api.verda.com/v1/docs#description/2026-02-03-spot-instance-volume-policy
            # The default, keep_detached, leaves the OS volume billing after
            # Verda evicts a spot instance.
            if is_spot:
                instance_data['os_volume'][
                    'on_spot_discontinue'] = 'delete_permanently'
            try:
                response = verda.instance_create(instance_data)
            except VerdaException as e:
                # https://api.verda.com/v1/docs#tag/instances/POST/v1/instances
                # The default image uses NVIDIA's open kernel modules, which
                # need a Turing or newer GPU, so Verda rejects it on V100 with
                # "Operating system is not valid for this instance type". The
                # API docs do not list this message.
                if (image != VERDA_DEFAULT_IMAGE or
                        'Operating system is not valid' not in e.message):
                    raise
                image = _fallback_image(instance_type)
                instance_data['image'] = image
                response = verda.instance_create(instance_data)
            instance_id = response.instance_id
        except Exception as e:  # pylint: disable=broad-except
            # API errors - provide specific message
            instance_type = config.node_config['InstanceType']
            region_str = (f' in region {region}'
                          if region != 'PLACEHOLDER' else '')
            # Check if it's a resource unavailability error
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in [
                    'no capacity',
                    'capacity',
                    'unavailable',
                    'out of stock',
                    'insufficient',
                    'not available',
                    'quota exceeded',
                    'limit exceeded',
            ]):
                error_msg = (
                    f'Resources are currently unavailable on Verda. '
                    f'No {instance_type} instances are available{region_str}. '
                    f'Please try again later or consider using a different '
                    f'instance type or region. Details: {str(e)}')
            else:
                error_msg = (
                    f'Failed to launch {instance_type} instance on Verda'
                    f'{region_str}. Details: {str(e)}')
            logger.warning(f'API error during instance launch: {e}')
            with ux_utils.print_exception_no_traceback():
                raise exceptions.ResourcesUnavailableError(error_msg) from e
        logger.info(f'Launched instance {instance_id}.')
        created_instance_ids.append(instance_id)
        if head_instance_id is None:
            head_instance_id = instance_id

    # Wait for instances to be ready.
    for _ in range(MAX_POLLS_FOR_UP_OR_TERMINATE):
        failed = {
            inst_id: inst
            for inst_id, inst in _filter_instances(cluster_name_on_cloud,
                                                   _FAILED_STATUSES).items()
            if inst_id in created_instance_ids
        }
        if failed:
            instance_type = config.node_config['InstanceType']
            statuses = sorted({inst.status for inst in failed.values()})
            with ux_utils.print_exception_no_traceback():
                raise exceptions.ResourcesUnavailableError(
                    f'Failed to launch {instance_type} on Verda in region '
                    f'{region}: instance status {", ".join(statuses)}.')
        instances = _filter_instances(cluster_name_on_cloud,
                                      [InstanceStatus.RUNNING])
        logger.info('Waiting for instances to be ready: '
                    f'({len(instances)}/{config.count}).')
        if len(instances) == config.count:
            break

        time.sleep(POLL_INTERVAL)
    else:
        # Failed to launch config.count of instances after max retries
        # Provide more specific error message
        instance_type = config.node_config['InstanceType']
        region_str = f' in region {region}' if region != 'PLACEHOLDER' else ''
        active_instances = len(
            _filter_instances(cluster_name_on_cloud, [InstanceStatus.RUNNING]))
        error_msg = (
            f'Timed out waiting for {instance_type} instances to become '
            f'ready on Verda Cloud{region_str}. Only {active_instances} '
            f'out of {config.count} instances became active. This may '
            f'indicate capacity issues or slow provisioning. Please try '
            f'again later or consider using a different instance type or '
            f'region.')
        logger.warning(error_msg)
        with ux_utils.print_exception_no_traceback():
            raise exceptions.ResourcesUnavailableError(error_msg)
    assert head_instance_id is not None, 'head_instance_id should not be None'
    return common.ProvisionRecord(
        provider_name='verda',
        cluster_name=cluster_name_on_cloud,
        region=region,
        zone=None,
        head_instance_id=head_instance_id,
        resumed_instance_ids=[],
        created_instance_ids=created_instance_ids,
    )


def wait_instances(
    region: str,
    cluster_name_on_cloud: str,
    state: Optional[status_lib.ClusterStatus],
) -> None:
    # Waiting for instances to be ready is already handled in run_instances.
    del region, cluster_name_on_cloud, state


def stop_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    raise NotImplementedError()


def terminate_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """See sky/provision/__init__.py"""
    del provider_config  # unused
    instances = _filter_instances(cluster_name_on_cloud, None)

    # Log if no instances found
    if not instances:
        logger.info(f'No instances found for cluster {cluster_name_on_cloud}')
        return

    # Filter out already terminated instances
    non_terminated_instances = {
        inst_id: inst for inst_id, inst in instances.items() if inst.status
        not in [InstanceStatus.DISCONTINUED, InstanceStatus.DELETING]
    }

    if not non_terminated_instances:
        logger.info(
            f'All instances for cluster {cluster_name_on_cloud} are already '
            f'terminated or being deleted')
        return

    # Log what we're about to terminate
    instance_names = [
        inst.hostname for inst in non_terminated_instances.values()
    ]
    logger.info(
        f'Terminating {len(non_terminated_instances)} instances for cluster '
        f'{cluster_name_on_cloud}: {instance_names}')

    # Terminate each instance
    terminated_instances = []
    for instance_id, inst in non_terminated_instances.items():
        status = inst.status
        logger.debug(f'Terminating instance {instance_id} (status: {status})')
        if worker_only and inst.hostname.endswith('-head'):
            continue
        try:
            # Without volume_ids, the OS volume survives detached and keeps
            # billing. Verda removes it a few minutes after the instance.
            verda.instance_action(
                instance_id=instance_id,
                action='delete',
                volume_ids=[inst.os_volume_id] if inst.os_volume_id else None)
            terminated_instances.append(instance_id)
            name = inst.hostname
            logger.info(
                f'Successfully initiated termination of instance {instance_id}'
                f' ({name})')
        except Exception as e:  # pylint: disable=broad-except
            with ux_utils.print_exception_no_traceback():
                raise RuntimeError(
                    f'Failed to terminate instance {instance_id}: '
                    f'{common_utils.format_exception(e, use_bracket=False)}'
                ) from e

    # Wait for instances to be terminated
    if not terminated_instances:
        logger.info(
            'No instances were terminated (worker_only=True and only head '
            'node found)')
        return

    logger.info(
        f'Waiting for {len(terminated_instances)} instances to be terminated...'
    )
    for _ in range(MAX_POLLS_FOR_UP_OR_TERMINATE):
        remaining_instances = _filter_instances(cluster_name_on_cloud, None)

        # Check if all terminated instances are gone
        still_exist = [
            inst_id for inst_id in terminated_instances
            if inst_id in remaining_instances and
            remaining_instances[inst_id].status != InstanceStatus.DISCONTINUED
        ]
        if not still_exist:
            logger.info('All instances have been successfully terminated')
            break

        # Log status of remaining instances
        remaining_statuses = [(inst_id, remaining_instances[inst_id].status)
                              for inst_id in still_exist]
        logger.info(
            f'Waiting for termination... {len(still_exist)} instances still '
            f'exist: {remaining_statuses}')
        time.sleep(POLL_INTERVAL)
    else:
        # Timeout reached
        remaining_instances = _filter_instances(cluster_name_on_cloud, None)
        still_exist = [
            inst_id for inst_id in terminated_instances
            if inst_id in remaining_instances
        ]
        if still_exist:
            logger.warning(
                f'Timeout reached. {len(still_exist)} instances may still be '
                f'terminating: {still_exist}')
        else:
            logger.info('All instances have been successfully terminated')


def get_cluster_info(
    region: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
) -> common.ClusterInfo:
    del region  # unused
    running_instances = _filter_instances(cluster_name_on_cloud,
                                          [InstanceStatus.RUNNING])
    instances: Dict[str, List[common.InstanceInfo]] = {}
    head_instance_id = None
    for instance_id, instance in running_instances.items():
        running_instances[instance_id] = _get_instance_info(instance_id)
        external_ip = instance.ip
        if isinstance(external_ip, list):
            external_ip = external_ip[0]

        instances[instance_id] = [
            common.InstanceInfo(
                instance_id=instance_id,
                internal_ip='NOT_SUPPORTED',
                external_ip=external_ip,
                ssh_port=22,
                tags={'provider': 'verda'},
            )
        ]
        if instance.hostname.endswith('-head'):
            head_instance_id = instance_id

    return common.ClusterInfo(
        instances=instances,
        head_instance_id=head_instance_id,
        provider_name='verda',
        provider_config=provider_config,
        ssh_user='root',
    )


def query_instances(
    cluster_name: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    non_terminated_only: bool = True,
    retry_if_missing: bool = False,
) -> Dict[str, Tuple[Optional['status_lib.ClusterStatus'], Optional[str]]]:
    """See sky/provision/__init__.py"""
    assert provider_config is not None, (cluster_name_on_cloud, provider_config)
    del cluster_name, retry_if_missing  # unused
    instances = _filter_instances(cluster_name_on_cloud, None)

    statuses: Dict[str, Tuple[Optional[status_lib.ClusterStatus],
                              Optional[str]]] = {}
    for inst_id, inst in instances.items():
        status = _STATUS_MAP.get(inst.status, status_lib.ClusterStatus.INIT)
        if non_terminated_only and status is None:
            continue
        reason = (f'Verda instance status: {inst.status}'
                  if inst.status in _FAILED_STATUSES else None)
        statuses[inst_id] = (status, reason)
    return statuses


def cleanup_ports(
    cluster_name_on_cloud: str,
    ports: List[str],
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    del cluster_name_on_cloud, ports, provider_config  # Unused.
