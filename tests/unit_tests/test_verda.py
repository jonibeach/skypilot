"""Tests for Verda Cloud provider."""

import concurrent.futures
import json
from pathlib import Path
import stat
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from sky import clouds
from sky import exceptions
from sky import task as task_lib
from sky.adaptors import verda as verda_adaptor
from sky.adaptors.verda import Instance
from sky.adaptors.verda import InstanceStatus
from sky.adaptors.verda import VerdaClient
from sky.catalog import verda_catalog
from sky.clouds import verda
from sky.data import storage as storage_lib
from sky.provision.verda import instance as verda_instance


def test_verda_cloud_basics():
    cloud = verda.Verda()
    assert cloud.name == "verda"
    assert cloud._REPR == "Verda"
    assert cloud.max_cluster_name_length() == 52


def test_verda_credential_file_mounts(monkeypatch, tmp_path):
    config = verda_adaptor.VerdaConfiguration('id', 'secret', 'https://api',
                                              'FIN-03')
    monkeypatch.setattr(verda_adaptor, 'get_verda_configuration', lambda:
                        (True, None, config))
    monkeypatch.setattr(verda_adaptor, '_GENERATED_S3_DIR', str(tmp_path))
    monkeypatch.setattr(verda_adaptor, 'get_s3_credential_file_mounts',
                        lambda: {})
    mounts = verda.Verda().get_credential_file_mounts()
    path = Path(mounts['~/.verda/config.json'])
    assert json.loads(path.read_text())['client_secret'] == 'secret'
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_verda_region_zone_validation_disallows_zones():
    cloud = verda.Verda()
    with pytest.raises(ValueError, match="does not support zones"):
        cloud.validate_region_zone("some-region", "zone-1")


@patch('sky.clouds.verda.get_verda_configuration')
def test_verda_check_credentials_missing(mock_get_config, monkeypatch,
                                         tmp_path):
    cloud = verda.Verda()
    # Mock configuration to return False (not configured)
    mock_get_config.return_value = (False, "Verda credentials not found", None)

    fake_path = tmp_path / "config.json"
    monkeypatch.setattr(verda.Verda, "CREDENTIALS_PATH", str(fake_path))

    valid, msg = cloud.check_credentials(clouds.CloudCapability.COMPUTE)
    assert not valid
    assert "Verda credentials not found" in msg


class TestVerdaClientInstanceCreation:
    """Test cases for VerdaClient instance creation through Verda adaptor."""

    @patch('sky.adaptors.verda.get_verda_configuration')
    @patch('sky.adaptors.verda.requests.post')
    @patch('sky.adaptors.verda.requests.get')
    @patch('sky.adaptors.verda.time.time')
    def test_instance_create_success(self, mock_time, mock_get, mock_post,
                                     mock_get_config):
        """Test successful instance creation."""
        # Mock time to control token expiration
        mock_time.return_value = 1000.0

        # Mock configuration
        mock_config = MagicMock()
        mock_config.client_id = "test-client-id"
        mock_config.client_secret = "test-client-secret"
        mock_config.base_url = "https://api.verda.com/v1"
        mock_get_config.return_value = (True, None, mock_config)

        # Mock authentication response (called during HTTPClient.__init__)
        auth_response = MagicMock()
        auth_response.ok = True
        auth_response.json.return_value = {
            'access_token': 'test-access-token',
            'refresh_token': 'test-refresh-token',
            'scope': 'read write',
            'token_type': 'Bearer',
            'expires_in': 3600
        }
        auth_response.status_code = 200

        # Mock instance creation response
        create_response = MagicMock()
        create_response.ok = True
        create_response.text = "instance-123"
        create_response.status_code = 200

        # Mock instance get response
        get_response = MagicMock()
        get_response.ok = True
        get_response.json.return_value = {
            'id': 'instance-123',
            'status': InstanceStatus.RUNNING,
            'ip': '10.0.0.1',
            'hostname': 'test-cluster-head'
        }
        get_response.status_code = 200

        # Set up mock call order: auth (during HTTPClient init), create, get
        mock_post.side_effect = [auth_response, create_response]
        mock_get.return_value = get_response

        # Create client and instance
        client = VerdaClient()
        payload = {
            'instance_type': 'gpu-h100-8gpu',
            'hostname': 'test-cluster-head',
            'location_code': 'FIN-03',
            'is_spot': False,
            'contract': 'PAY_AS_YOU_GO',
            'image': 'ubuntu-24.04-cuda-12.8-open-docker',
            'description': 'Created by SkyPilot',
            'ssh_key_ids': ['ssh-key-1'],
            'os_volume': {
                'name': 'test-cluster-head',
                'size': 50
            }
        }

        instance = client.instance_create(payload)

        # Verify instance creation
        assert isinstance(instance, Instance)
        assert instance.instance_id == 'instance-123'
        assert instance.status == InstanceStatus.RUNNING
        assert instance.ip == '10.0.0.1'
        assert instance.hostname == 'test-cluster-head'

        # Verify API calls
        # First post is auth (during HTTPClient.__init__), second is instance create
        assert mock_post.call_count == 2
        assert mock_get.call_count == 1

        # Verify auth call
        auth_call = mock_post.call_args_list[0]
        assert 'https://api.verda.com/v1/oauth2/token' in auth_call[0][0]
        assert auth_call[1]['json']['grant_type'] == 'client_credentials'

        # Verify create call - check URL and payload
        create_call = mock_post.call_args_list[1]
        assert 'https://api.verda.com/v1/instances' in create_call[0][0]
        assert create_call[1]['json'] == payload

        # Verify get call
        get_call = mock_get.call_args
        assert 'https://api.verda.com/v1/instances/instance-123' in get_call[0][
            0]

    @patch('sky.adaptors.verda.get_verda_configuration')
    @patch('sky.adaptors.verda.requests.post')
    @patch('sky.adaptors.verda.requests.get')
    @patch('sky.adaptors.verda.time.time')
    def test_instance_create_with_spot(self, mock_time, mock_get, mock_post,
                                       mock_get_config):
        """Test instance creation with spot/preemptible instance."""
        # Mock time to control token expiration
        mock_time.return_value = 1000.0

        # Mock configuration
        mock_config = MagicMock()
        mock_config.client_id = 'test-client-id'
        mock_config.client_secret = 'test-client-secret'
        mock_config.base_url = 'https://api.verda.com/v1'
        mock_get_config.return_value = (True, None, mock_config)

        # Mock authentication response (called during HTTPClient.__init__)
        auth_response = MagicMock()
        auth_response.ok = True
        auth_response.json.return_value = {
            'access_token': 'test-access-token',
            'refresh_token': 'test-refresh-token',
            'scope': 'read write',
            'token_type': 'Bearer',
            'expires_in': 3600
        }
        auth_response.status_code = 200

        # Mock instance creation response
        create_response = MagicMock()
        create_response.ok = True
        create_response.text = "instance-456"
        create_response.status_code = 200

        # Mock instance get response
        get_response = MagicMock()
        get_response.ok = True
        get_response.json.return_value = {
            'id': 'instance-456',
            'status': InstanceStatus.PROVISIONING,
            'ip': None,
            'hostname': 'test-cluster-worker'
        }
        get_response.status_code = 200

        # Set up mock call order: auth (during HTTPClient init), create, get
        mock_post.side_effect = [auth_response, create_response]
        mock_get.return_value = get_response

        # Create client and instance with spot
        client = VerdaClient()
        payload = {
            'instance_type': 'gpu-h100-8gpu',
            'hostname': 'test-cluster-worker',
            'location_code': 'FIN-03',
            'is_spot': True,
            'contract': 'SPOT',
            'image': 'ubuntu-24.04-cuda-12.8-open-docker',
            'description': 'Created by SkyPilot',
            'ssh_key_ids': ['ssh-key-1'],
            'os_volume': {
                'name': 'test-cluster-worker',
                'size': 100
            }
        }

        instance = client.instance_create(payload)

        # Verify instance creation
        assert isinstance(instance, Instance)
        assert instance.instance_id == 'instance-456'
        assert instance.status == InstanceStatus.PROVISIONING
        assert instance.ip is None
        assert instance.hostname == 'test-cluster-worker'

        # Verify create call has spot settings
        create_call = mock_post.call_args_list[1]
        assert create_call[1]['json']['is_spot'] is True
        assert create_call[1]['json']['contract'] == 'SPOT'

    @patch('sky.adaptors.verda.get_verda_configuration')
    @patch('sky.adaptors.verda.requests.post')
    @patch('sky.adaptors.verda.time.time')
    def test_instance_create_api_error(self, mock_time, mock_post,
                                       mock_get_config):
        """Test instance creation with API error."""
        # Mock time to control token expiration
        mock_time.return_value = 1000.0

        # Mock configuration
        mock_config = MagicMock()
        mock_config.client_id = "test-client-id"
        mock_config.client_secret = "test-client-secret"
        mock_config.base_url = "https://api.verda.com/v1"
        mock_get_config.return_value = (True, None, mock_config)

        # Mock authentication response (called during HTTPClient.__init__)
        auth_response = MagicMock()
        auth_response.ok = True
        auth_response.json.return_value = {
            'access_token': 'test-access-token',
            'refresh_token': 'test-refresh-token',
            'scope': 'read write',
            'token_type': 'Bearer',
            'expires_in': 3600
        }
        auth_response.status_code = 200

        # Mock instance creation error response
        error_response = MagicMock()
        error_response.ok = False
        error_response.status_code = 400
        error_response.text = '{"code": "INVALID_INPUT", "message": "Invalid instance type"}'

        # Set up mock call order: auth (during HTTPClient init), create (error)
        mock_post.side_effect = [auth_response, error_response]

        # Create client and attempt instance creation
        client = VerdaClient()
        payload = {
            'instance_type': 'invalid-instance-type',
            'hostname': 'test-cluster-head',
            'location_code': 'FIN-03',
            'is_spot': False,
            'contract': 'PAY_AS_YOU_GO',
            'image': 'ubuntu-24.04-cuda-12.8-open-docker',
            'description': 'Created by SkyPilot',
            'ssh_key_ids': ['ssh-key-1'],
            'os_volume': {
                'name': 'test-cluster-head',
                'size': 50
            }
        }

        # Verify APIException is raised
        from sky.adaptors.verda import VerdaException
        with pytest.raises(VerdaException) as exc_info:
            client.instance_create(payload)

        assert exc_info.value.code == 'INVALID_INPUT'
        assert 'Invalid instance type' in exc_info.value.message

    @patch('sky.adaptors.verda.get_verda_configuration')
    def test_instance_create_configuration_error(self, mock_get_config):
        """Test instance creation with configuration error."""
        # Mock configuration error
        mock_get_config.return_value = (False, "Configuration not found", None)

        # Verify RuntimeError is raised when creating VerdaClient
        # (HTTPClient is initialized lazily, but we can test by trying to use it)
        client = VerdaClient()
        # HTTPClient is created lazily when instance_create is called
        with pytest.raises(RuntimeError) as exc_info:
            client.instance_create({})

        assert "Can't connect to Verda Cloud" in str(exc_info.value)


def _enable_storage_clouds(monkeypatch, names):
    monkeypatch.setattr(storage_lib,
                        'get_cached_enabled_storage_cloud_names_or_refresh',
                        lambda raise_if_no_cloud_access=False: names)


def test_preferred_store_skips_verda(monkeypatch):
    _enable_storage_clouds(monkeypatch, ['Verda', 'AWS'])
    task = task_lib.Task(run='echo hi')
    task.best_resources = MagicMock(cloud=verda.Verda(), region='FIN-01')
    assert task._get_preferred_store() == (storage_lib.StoreType.S3, None)


def test_preferred_store_errors_when_only_verda(monkeypatch):
    _enable_storage_clouds(monkeypatch, ['Verda'])
    task = task_lib.Task(run='echo hi')
    task.best_resources = MagicMock(cloud=verda.Verda(), region='FIN-01')
    with pytest.raises(exceptions.NoCloudAccessError):
        task._get_preferred_store()


@pytest.fixture
def live_catalog(monkeypatch):
    monkeypatch.setattr(verda_catalog, '_offerings', None)
    monkeypatch.setattr(verda_catalog, '_in_stock', None)
    monkeypatch.setattr(verda_catalog.verda, 'get_verda_configuration', lambda:
                        (True, None, None))
    client = MagicMock()
    monkeypatch.setattr(verda_catalog, '_client', client)
    return client


def test_catalog_lists_in_stock_regions_first(monkeypatch, live_catalog):
    offerings = pd.DataFrame({
        'InstanceType': ['T', 'T'],
        'Region': ['FIN-01', 'FIN-03'],
        'Price': [1.0, 1.0],
        'SpotPrice': [None, None],
    })
    monkeypatch.setattr(verda_catalog, '_fetch_offerings', lambda: offerings)
    live_catalog.instance_availability_get.side_effect = (
        lambda is_spot: set() if is_spot else {('T', 'FIN-03')})
    regions = verda_catalog.get_region_zones_for_instance_type('T', False)
    assert [region.name for region in regions] == ['FIN-03', 'FIN-01']


def test_catalog_caches_failed_fetches(monkeypatch, live_catalog):
    fetch = MagicMock(side_effect=RuntimeError('down'))
    monkeypatch.setattr(verda_catalog, '_fetch_offerings', fetch)
    live_catalog.instance_availability_get.side_effect = RuntimeError('down')
    for _ in range(2):
        assert verda_catalog._offerings_df() is verda_catalog._hosted_df
        assert not verda_catalog._in_stock_regions('T', False)
    assert fetch.call_count == 1
    assert live_catalog.instance_availability_get.call_count == 1


@pytest.mark.parametrize('prefix', ['VERDA', 'DATACRUNCH'])
def test_environment_credentials_reach_remote(monkeypatch, tmp_path, prefix):
    for name in ('VERDA_CLIENT_ID', 'VERDA_CLIENT_SECRET',
                 'DATACRUNCH_CLIENT_ID', 'DATACRUNCH_CLIENT_SECRET'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(f'{prefix}_CLIENT_ID', 'env-id')
    monkeypatch.setenv(f'{prefix}_CLIENT_SECRET', 'env-secret')
    monkeypatch.setenv(f'{prefix}_BASE_URL', 'https://example.invalid/v1')
    monkeypatch.setenv(f'{prefix}_DEFAULT_REGION', 'FIN-03')
    monkeypatch.setattr(verda_adaptor, '_GENERATED_S3_DIR', str(tmp_path))
    monkeypatch.setattr(verda_adaptor, 'get_s3_credential_file_mounts',
                        lambda: {})
    mounts = verda.Verda().get_credential_file_mounts()
    config = json.loads(Path(mounts['~/.verda/config.json']).read_text())
    assert config == dict(client_id='env-id',
                          client_secret='env-secret',
                          base_url='https://example.invalid/v1',
                          default_region='FIN-03')


@pytest.fixture
def cli_storage_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(verda_adaptor, '_GENERATED_S3_DIR',
                        str(tmp_path / 'generated'))
    monkeypatch.setattr(verda_adaptor, 'VERDA_S3_CREDENTIALS_PATH',
                        str(tmp_path / 's3.credentials'))
    monkeypatch.setattr(verda_adaptor, 'VERDA_S3_CONFIG_PATH',
                        str(tmp_path / 's3.config'))
    monkeypatch.setattr(
        verda_adaptor, '_cli_s3_section', lambda: {
            'verda_s3_access_key': 'test-access',
            'verda_s3_secret_key': 'test-secret',
            'verda_s3_endpoint': 'https://objects.example.invalid',
        })
    return tmp_path / 'generated'


def test_storage_registration_does_not_write_credentials(
        cli_storage_credentials):
    storage_lib.register_s3_compatible_store(storage_lib.VerdaStore)
    assert storage_lib.StoreType.VERDA.store_prefix() == 'verda://'
    assert not cli_storage_credentials.exists()


def test_storage_initialization_prepares_cli_files(monkeypatch,
                                                   cli_storage_credentials):
    store = storage_lib.VerdaStore.__new__(storage_lib.VerdaStore)
    store.config = storage_lib.VerdaStore.get_config()

    def initialize(self):
        assert 'test-access' in Path(self.config.credentials_file).read_text()
        assert 'objects.example.invalid' in Path(
            self.config.config_file).read_text()

    monkeypatch.setattr(storage_lib.S3CompatibleStore, 'initialize', initialize)
    store.initialize()


def test_generated_storage_credentials_are_private(cli_storage_credentials):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(
            pool.map(lambda _: verda_adaptor.local_s3_files(), range(20)))
    assert len(set(paths)) == 1
    for path in paths[0]:
        assert Path(path).read_text().startswith('[')
        assert stat.S_IMODE(Path(path).stat().st_mode) == 0o600
    assert 'test-secret' in Path(paths[0][0]).read_text()


def test_atomic_credentials_keep_old_file_on_failure(monkeypatch, tmp_path):
    path = tmp_path / 'credentials'
    path.write_text('old')
    monkeypatch.setattr(verda_adaptor.os, 'replace',
                        MagicMock(side_effect=OSError('replace failed')))
    with pytest.raises(OSError, match='replace failed'):
        verda_adaptor._write_private(str(path), 'new')
    assert path.read_text() == 'old'
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    'subpath,target',
    [(None, 'bucket'),
     ('jobs/workspaces/team/run', 'bucket/jobs/workspaces/team/run')])
def test_cached_mount_keeps_subpath(monkeypatch, subpath, target):
    store = storage_lib.VerdaStore.__new__(storage_lib.VerdaStore)
    store.name = 'bucket'
    store.bucket = MagicMock(name='bucket')
    store.bucket.name = 'bucket'
    store._bucket_sub_path = subpath
    monkeypatch.setattr(storage_lib.data_utils.Rclone.RcloneStores,
                        'get_config', lambda *args, **kwargs: 'config')
    command = store.mount_cached_command('/data')
    assert f'sky-verda-bucket:{target} /data' in command


@pytest.mark.parametrize('image', [
    '24.04.cuda12.3.docker', 'ubuntu-24.04-cuda-12.3-docker',
    'ubuntu-24.04-cuda-12.3', 'jupyter'
])
def test_image_fallback_accepts_supported_names(monkeypatch, image):
    monkeypatch.setattr(verda_instance.verda, 'images_get', lambda _: [image])
    assert verda_instance._fallback_image('gpu') == image


def test_image_fallback_prefers_docker(monkeypatch):
    monkeypatch.setattr(
        verda_instance.verda, 'images_get', lambda _:
        ['jupyter', 'ubuntu-24.04-cuda-12.3', 'ubuntu-24.04-cuda-12.3-docker'])
    assert verda_instance._fallback_image('gpu').endswith('-docker')


def test_pending_instance_wait_times_out(monkeypatch):
    instance = Instance(
        dict(id='pending', hostname='cluster-head', status='new'))
    monkeypatch.setattr(verda_instance.verda, 'instances_get',
                        lambda: [instance])
    create = MagicMock()
    monkeypatch.setattr(verda_instance.verda, 'instance_create', create)
    monkeypatch.setattr(verda_instance, 'MAX_POLLS_FOR_UP_OR_TERMINATE', 2)
    monkeypatch.setattr(verda_instance.time, 'sleep', lambda _: None)
    with pytest.raises(exceptions.ResourcesUnavailableError, match='pending'):
        verda_instance.run_instances('FIN-03', 'cluster', 'cluster',
                                     MagicMock())
    create.assert_not_called()


def test_spot_cpu_selection_prefers_available_type(monkeypatch, live_catalog):
    offerings = pd.DataFrame({
        'InstanceType': ['cheap', 'available'],
        'Region': ['FIN-03', 'FIN-03'],
        'vCPUs': [4, 4],
        'MemoryGiB': [16, 16],
        'Price': [0.1, 0.2],
        'SpotPrice': [0.02, 0.05],
    })
    monkeypatch.setattr(verda_catalog, '_fetch_offerings', lambda: offerings)
    live_catalog.instance_availability_get.side_effect = (
        lambda is_spot: {('available', 'FIN-03')}
        if is_spot else {('cheap', 'FIN-03')})
    assert verda_catalog.get_default_instance_type(use_spot=True) == 'available'


def test_cpu_selection_falls_back_when_stock_unknown(monkeypatch, live_catalog):
    offerings = pd.DataFrame({
        'InstanceType': ['cpu'],
        'Region': ['FIN-03'],
        'vCPUs': [4],
        'MemoryGiB': [16],
        'Price': [0.1],
        'SpotPrice': [0.05],
    })
    monkeypatch.setattr(verda_catalog, '_fetch_offerings', lambda: offerings)
    live_catalog.instance_availability_get.side_effect = RuntimeError('offline')
    assert verda_catalog.get_default_instance_type(use_spot=True) == 'cpu'


def test_failed_refresh_keeps_last_good_catalog(monkeypatch, live_catalog):
    offerings = pd.DataFrame({'InstanceType': ['live-only']})
    monkeypatch.setattr(verda_catalog, '_offerings', (0, offerings))
    fetch = MagicMock(side_effect=RuntimeError('offline'))
    monkeypatch.setattr(verda_catalog, '_fetch_offerings', fetch)
    assert verda_catalog._offerings_df() is offerings
    assert verda_catalog._offerings_df() is offerings
    fetch.assert_called_once()


def test_cpu_selection_returns_none_without_spot_prices(monkeypatch,
                                                        live_catalog):
    offerings = pd.DataFrame({
        'InstanceType': ['cpu'],
        'Region': ['FIN-03'],
        'vCPUs': [4],
        'MemoryGiB': [16],
        'Price': [0.1],
        'SpotPrice': [None],
    })
    monkeypatch.setattr(verda_catalog, '_fetch_offerings', lambda: offerings)
    live_catalog.instance_availability_get.return_value = set()
    assert verda_catalog.get_default_instance_type(use_spot=True) is None


def test_pending_instance_is_reused_when_ready(monkeypatch):
    pending = Instance(
        dict(id='existing', hostname='cluster-head', status='new'))
    running = Instance(
        dict(id='existing', hostname='cluster-head', status='running'))
    instances = MagicMock(
        side_effect=[[pending], [pending], [running], [running]])
    monkeypatch.setattr(verda_instance.verda, 'instances_get', instances)
    create = MagicMock()
    monkeypatch.setattr(verda_instance.verda, 'instance_create', create)
    monkeypatch.setattr(verda_instance.time, 'sleep', lambda _: None)
    record = verda_instance.run_instances('FIN-03', 'cluster', 'cluster',
                                          MagicMock(count=1))
    assert record.head_instance_id == 'existing'
    assert record.zone is None
    create.assert_not_called()


def test_custom_image_failure_does_not_trigger_fallback(monkeypatch):
    monkeypatch.setattr(verda_instance.verda, 'instances_get', lambda: [])
    monkeypatch.setattr(verda_instance, 'find_ssh_key_id', lambda _: 'key')
    create = MagicMock(side_effect=verda_adaptor.VerdaException(
        'invalid_request',
        'Operating system is not valid for this instance type'))
    monkeypatch.setattr(verda_instance.verda, 'instance_create', create)
    fallback = MagicMock()
    monkeypatch.setattr(verda_instance, '_fallback_image', fallback)
    config = MagicMock(count=1,
                       node_config={
                           'InstanceType': 'gpu',
                           'ImageId': 'custom',
                           'PublicKey': 'public-key'
                       })
    with pytest.raises(exceptions.ResourcesUnavailableError):
        verda_instance.run_instances('FIN-03', 'cluster', 'cluster', config)
    fallback.assert_not_called()
    create.assert_called_once()


def test_storage_profiles_use_separate_generated_files(monkeypatch,
                                                       cli_storage_credentials):
    first = verda_adaptor.local_s3_files()
    monkeypatch.setattr(
        verda_adaptor, '_cli_s3_section', lambda: {
            'verda_s3_access_key': 'other-access',
            'verda_s3_secret_key': 'other-secret',
        })
    second = verda_adaptor.local_s3_files()
    assert first != second
    assert 'test-secret' in Path(first[0]).read_text()
    assert 'other-secret' in Path(second[0]).read_text()


@pytest.mark.parametrize('available,expected',
                         [({'available'}, ['available']),
                          (set(), ['cheap', 'available'])])
def test_gpu_selection_prefers_stock_with_fallback(monkeypatch, live_catalog,
                                                   available, expected):
    offerings = pd.DataFrame({
        'InstanceType': ['cheap', 'available'],
        'Region': ['FIN-03', 'FIN-03'],
        'vCPUs': [4, 4],
        'MemoryGiB': [16, 16],
        'AcceleratorName': ['H100', 'H100'],
        'AcceleratorCount': [1, 1],
        'Price': [0.1, 0.2],
        'SpotPrice': [0.02, 0.05],
    })
    monkeypatch.setattr(verda_catalog, '_fetch_offerings', lambda: offerings)
    live_catalog.instance_availability_get.return_value = {
        (name, 'FIN-03') for name in available
    }
    matches, _ = verda_catalog.get_instance_type_for_accelerator('H100',
                                                                 1,
                                                                 use_spot=True)
    assert matches == expected
