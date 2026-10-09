"""Tests for Verda Cloud provider."""

import concurrent.futures
from pathlib import Path
import stat
import tempfile
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from sky import clouds
from sky import exceptions
from sky import task as task_lib
from sky.adaptors import verda as verda_adaptor
from sky.adaptors.verda import Instance
from sky.adaptors.verda import InstanceStatus
from sky.adaptors.verda import VerdaClient
from sky.clouds import verda
from sky.data import storage as storage_lib


def test_verda_cloud_basics():
    cloud = verda.Verda()
    assert cloud.name == "verda"
    assert cloud._REPR == "Verda"
    assert cloud._MAX_CLUSTER_NAME_LEN_LIMIT == 120


def test_verda_credential_file_mounts():
    with tempfile.TemporaryDirectory() as tmpdir:
        cred_path = Path(tmpdir) / "config.yaml"
        cred_path.touch()
        with pytest.MonkeyPatch.context() as m:
            m.setattr(verda.Verda, "CREDENTIALS_PATH", str(cred_path))
            m.setattr(verda_adaptor, "get_s3_credential_file_mounts",
                      lambda: {})
            cloud = verda.Verda()
            mounts = cloud.get_credential_file_mounts()
            assert mounts["~/.verda/config.json"] == str(cred_path)


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
def cli_storage_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(verda_adaptor, '_GENERATED_DIR',
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


def test_endpoint_lookup_does_not_write_credentials(cli_storage_credentials):
    assert verda_adaptor.get_endpoint() == 'https://objects.example.invalid'
    assert not cli_storage_credentials.exists()
