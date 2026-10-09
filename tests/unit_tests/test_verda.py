"""Tests for Verda Cloud provider."""

import json
from pathlib import Path
import stat
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from sky import clouds
from sky import exceptions
from sky.adaptors import verda as verda_adaptor
from sky.adaptors.verda import Instance
from sky.adaptors.verda import InstanceStatus
from sky.adaptors.verda import VerdaClient
from sky.clouds import verda
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
    mounts = verda.Verda().get_credential_file_mounts()
    config = json.loads(Path(mounts['~/.verda/config.json']).read_text())
    assert config == dict(client_id='env-id',
                          client_secret='env-secret',
                          base_url='https://example.invalid/v1',
                          default_region='FIN-03')


def test_atomic_credentials_keep_old_file_on_failure(monkeypatch, tmp_path):
    path = tmp_path / 'credentials'
    path.write_text('old')
    monkeypatch.setattr(verda_adaptor.os, 'replace',
                        MagicMock(side_effect=OSError('replace failed')))
    with pytest.raises(OSError, match='replace failed'):
        verda_adaptor._write_private(str(path), 'new')
    assert path.read_text() == 'old'
    assert list(tmp_path.iterdir()) == [path]


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
