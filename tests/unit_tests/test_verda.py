"""Tests for Verda Cloud provider."""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from sky import clouds
from sky.adaptors.verda import Instance
from sky.adaptors.verda import InstanceStatus
from sky.adaptors.verda import VerdaClient
from sky.catalog import verda_catalog
from sky.clouds import verda


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
            cloud = verda.Verda()
            mounts = cloud.get_credential_file_mounts()
            assert str(cred_path) in mounts
            assert mounts[str(cred_path)] == "~/.verda/config.json"


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
