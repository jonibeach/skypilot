"""Verda Cloud adaptor."""

import configparser
import contextlib
import dataclasses
import hashlib
import json
from json import load as json_load
from json import loads
import os
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from sky import sky_logging
from sky.adaptors import common
from sky.utils import annotations
from sky.utils import ux_utils

logger = sky_logging.init_logger(__name__)

VERDA_S3_PROFILE_NAME = 'verda'
VERDA_S3_CREDENTIALS_PATH = '~/.verda/s3.credentials'
VERDA_S3_CONFIG_PATH = '~/.verda/s3.config'
_CLI_CREDENTIALS_PATH = '~/.verda/credentials'
_GENERATED_DIR = '~/.sky/generated/verda'
# https://docs.verda.com/cli/object-storage/#configure-credentials
DEFAULT_REGION = 'us-east-1'
_DEFAULT_ENDPOINT = 'https://objects.fin-03.verda.storage'

_IMPORT_ERROR_MESSAGE = ('Failed to import dependencies for Verda object '
                         'storage. Try pip install "skypilot[verda]"')

boto3 = common.LazyImport('boto3', import_error_message=_IMPORT_ERROR_MESSAGE)
botocore = common.LazyImport('botocore',
                             import_error_message=_IMPORT_ERROR_MESSAGE)

_LAZY_MODULES = (boto3, botocore)
_session_creation_lock = threading.RLock()


@dataclasses.dataclass
class VerdaConfiguration:
    """A configuration for the Verda Cloud API."""
    client_id: str
    client_secret: str
    base_url: str
    default_region: str


def get_verda_configuration(
) -> Tuple[bool, Optional[str], Optional[VerdaConfiguration]]:
    """Checks known config file exists and is valid.
    Also supports using env vars instead.
    """
    try:
        if ('VERDA_CLIENT_ID' in os.environ and
                'VERDA_CLIENT_SECRET' in os.environ):
            # Configured via new env vars
            return (
                True,
                None,
                VerdaConfiguration(
                    client_id=os.environ['VERDA_CLIENT_ID'],
                    client_secret=os.environ['VERDA_CLIENT_SECRET'],
                    base_url=os.environ.get('VERDA_BASE_URL',
                                            'https://api.verda.com/v1'),
                    default_region=os.environ.get('VERDA_DEFAULT_REGION',
                                                  'FIN-03'),
                ),
            )

        if ('DATACRUNCH_CLIENT_ID' in os.environ and
                'DATACRUNCH_CLIENT_SECRET' in os.environ):
            # Configured via old env vars
            return (
                True,
                None,
                VerdaConfiguration(
                    client_id=os.environ['DATACRUNCH_CLIENT_ID'],
                    client_secret=os.environ['DATACRUNCH_CLIENT_SECRET'],
                    base_url=os.environ.get('DATACRUNCH_BASE_URL',
                                            'https://api.verda.com/v1'),
                    default_region=os.environ.get('DATACRUNCH_DEFAULT_REGION',
                                                  'FIN-03'),
                ),
            )

        filename = '~/.verda/config.json'
        config_file_path = os.path.expanduser(filename)
        if not os.path.exists(config_file_path):
            return (
                False,
                (f'Verda Cloud configuration not found. '
                 f'Please save your config.json as {config_file_path}\n'
                 '    Credentials can be set up by:\n'
                 '        $ mkdir -p ~/.verda\n'
                 '        $ cat > ~/.verda/config.json << EOF\n'
                 '        {\n'
                 '          "client_id": "your-client-id",\n'
                 '          "client_secret": "your-client-secret",\n'
                 '          "base_url": "https://api.verda.com/v1",\n'
                 '          "default_region": "FIN-03"\n'
                 '        }\n'
                 '        EOF'),
                None,
            )

        # Try to read the API key
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = json_load(f)

            if 'client_id' not in config or not config.get('client_id'):
                return (
                    False,
                    ('Verda Cloud Client ID is missing or empty in '
                     f'{config_file_path}\n'
                     '    Please ensure your config.json contains:\n'
                     '        {\n'
                     '          "client_id": "your-api-key",\n'
                     '          "client_secret": "your-api-secret"\n'
                     '        }'),
                    None,
                )
            elif 'client_secret' not in config or not config.get(
                    'client_secret'):
                return (
                    False,
                    ('Verda Cloud Client Secret is missing or empty in '
                     f'{config_file_path}\n'
                     '    Please ensure your config.json contains:\n'
                     '        {\n'
                     '          "client_id": "your-api-key",\n'
                     '          "client_secret": "your-api-secret"\n'
                     '        }'),
                    None,
                )
            else:
                return True, None, VerdaConfiguration(
                    client_id=config.get('client_id'),
                    client_secret=config.get('client_secret'),
                    base_url=config.get('base_url', 'https://api.verda.com/v1'),
                    default_region=config.get('default_region', 'FIN-03'),
                ),

    except (OSError, IOError) as e:
        return (
            False,
            ('Error reading Verda Cloud credentials from '
             f'{config_file_path}: {str(e)}\n'
             '    Please ensure the file exists and is readable.'),
            None,
        )
    except (KeyError, ValueError) as e:
        # KeyError for missing keys, ValueError for JSON decode errors
        return (
            False,
            (f'Error parsing Verda Cloud credentials from '
             f'{config_file_path}: {str(e)}\n'
             '    Please ensure your config.json is valid JSON'),
            None,
        )


_TOKEN_ENDPOINT = '/oauth2/token'
_CLIENT_CREDENTIALS = 'client_credentials'
_REFRESH_TOKEN = 'refresh_token'


class VerdaException(Exception):
    """This exception is raised if there was an error from verda's API.

    Could be an invalid input, token etc.

    Raised when an API HTTP call response has a status code >= 400
    """

    def __init__(self, code: str, message: str) -> None:
        """API Exception.

        :param code: error code
        :type code: str
        :param message: error message
        :type message: str
        """
        self.code = code
        """Error code. should be available in VerdaClient.error_codes"""

        self.message = message
        """Error message
        """
        super().__init__(message)

    def __str__(self) -> str:
        msg = ''
        if self.code:
            msg = f'error code: {self.code}\n'

        msg += f'message: {self.message}'
        return msg


def handle_error(response: requests.Response) -> None:
    """Checks for the status code and is response.ok

    :param response: the API call response
    :raises APIException: an api exception with message and error type code
    """
    if not response.ok:
        data = loads(response.text)
        code = data.get('code', 'Unknown')
        message = data.get('message', 'Internal error')
        raise VerdaException(code, message)


# requests waits forever by default, and the live catalog makes these calls
# while holding a lock.
_TIMEOUT_SECONDS = 30


class _AuthenticationService:
    """A service for client authentication."""

    def __init__(self, client_id: str, client_secret: str,
                 base_url: str) -> None:
        self._base_url = base_url
        self._client_id = client_id
        self._client_secret = client_secret

    def authenticate(self) -> dict:
        """Authenticate the client and store the access & refresh tokens.

        returns an authentication data dictionary with the following schema:
        {
            "access_token": token str,
            "refresh_token": token str,
            "scope": scope str,
            "token_type": token type str,
            "expires_in": duration until expires in seconds
        }

        :return: authentication data (tokens, scope, token type, expires in)
        :rtype: dict
        """
        url = self._base_url + _TOKEN_ENDPOINT
        payload = {
            'grant_type': _CLIENT_CREDENTIALS,
            'client_id': self._client_id,
            'client_secret': self._client_secret,
        }

        response = requests.post(url,
                                 json=payload,
                                 headers=self.generate_headers(),
                                 timeout=_TIMEOUT_SECONDS)
        handle_error(response)

        auth_data = response.json()

        self._access_token = auth_data['access_token']
        self._refresh_token = auth_data['refresh_token']
        self._scope = auth_data['scope']
        self._token_type = auth_data['token_type']
        self._expires_at = time.time() + auth_data['expires_in']
        return auth_data

    def refresh(self) -> dict:
        """Authenticate the client using the refresh token.

        updates the object's tokens and returns an authentication
        data dictionary with the following schema:
        {
            "access_token": token str,
            "refresh_token": token str,
            "scope": scope str,
            "token_type": token type str,
            "expires_in": duration until expires in seconds
        }

        :return: authentication data (tokens, scope, token type, expires in)
        :rtype: dict
        """
        url = self._base_url + _TOKEN_ENDPOINT

        payload = {
            'grant_type': _REFRESH_TOKEN,
            'refresh_token': self._refresh_token
        }

        response = requests.post(url,
                                 json=payload,
                                 headers=self.generate_headers(),
                                 timeout=_TIMEOUT_SECONDS)

        # if refresh token is also expired, authenticate again:
        if response.status_code == 401 or response.status_code == 400:
            return self.authenticate()
        else:
            handle_error(response)

        auth_data = response.json()

        self._access_token = auth_data['access_token']
        self._refresh_token = auth_data['refresh_token']
        self._scope = auth_data['scope']
        self._token_type = auth_data['token_type']
        self._expires_at = time.time() + auth_data['expires_in']

        return auth_data

    def generate_headers(self):
        """Generate the headers for the API request.

        :return: headers for the API request
        :rtype: dict
        """
        client_id_truncated = self._client_id[:10]
        headers = {
            'User-Agent': f'verda-python-v1-skypilot-{client_id_truncated}',
        }

        if hasattr(self, '_access_token') and self._access_token:
            headers['Authorization'] = f'Bearer {self._access_token}'

        return headers

    def is_expired(self) -> bool:
        """Returns true if the access token is expired.

        :return: True if the access token is expired, otherwise False.
        :rtype: bool
        """
        return time.time() >= self._expires_at


class _HTTPClient:
    """An http client, a wrapper for the requests library.

    For each request, it adds the authentication header with an access token.
    If the access token has expired it is refreshed it before calling the API.
    Also checks the response status code and raises an exception if needed.
    """

    def __init__(self) -> None:
        configured, reason, config = get_verda_configuration()
        if not configured or not config:
            raise RuntimeError(f'Can\'t connect to Verda Cloud: {reason}')
        self._base_url = config.base_url
        self._auth_service = _AuthenticationService(config.client_id,
                                                    config.client_secret,
                                                    config.base_url)
        self._auth_service.authenticate()

    def post(self,
             url: str,
             body: Optional[dict] = None,
             params: Optional[dict] = None,
             **kwargs) -> requests.Response:
        """Sends a POST request.

        A wrapper for the requests.post method.

        Builds the url, uses custom headers, refresh tokens if needed.

        :param url: relative url of the API endpoint
        :type url: str
        :param json: Python object to send in the body of the Request
        :type json: dict, optional
        :param params: Dictionary of querystring data to attach to the Request
        :type params: dict, optional

        :raises APIException: an api exception with message and error type code

        :return: Response object
        :rtype: requests.Response
        """
        self._refresh_token_if_expired()

        url = self._add_base_url(url)
        headers = self._generate_headers()

        response = requests.post(url,
                                 json=body,
                                 headers=headers,
                                 params=params,
                                 timeout=_TIMEOUT_SECONDS,
                                 **kwargs)
        handle_error(response)

        return response

    def put(self,
            url: str,
            body: Optional[dict] = None,
            params: Optional[dict] = None,
            **kwargs) -> requests.Response:
        """Sends a PUT request.

        A wrapper for the requests.put method.

        Builds the url, uses custom headers, refresh tokens if needed.

        :param url: relative url of the API endpoint
        :type url: str
        :param json: Python object to send in the body of the Request
        :type json: dict, optional
        :param params: Dictionary of querystring data to attach to the Request
        :type params: dict, optional

        :raises APIException: an api exception with message and error type code

        :return: Response object
        :rtype: requests.Response
        """
        self._refresh_token_if_expired()

        url = self._add_base_url(url)
        headers = self._generate_headers()

        response = requests.put(url,
                                json=body,
                                headers=headers,
                                params=params,
                                timeout=_TIMEOUT_SECONDS,
                                **kwargs)
        handle_error(response)

        return response

    def get(self,
            url: str,
            params: Optional[dict] = None,
            **kwargs) -> requests.Response:
        """Sends a GET request.

        A wrapper for the requests.get method.

        Builds the url, uses custom headers, refresh tokens if needed.

        :param url: relative url of the API endpoint
        :type url: str
        :param params: Dictionary of querystring data to attach to the Request
        :type params: dict, optional

        :raises APIException: an api exception with message and error type code

        :return: Response object
        :rtype: requests.Response
        """
        self._refresh_token_if_expired()

        url = self._add_base_url(url)
        headers = self._generate_headers()

        response = requests.get(url,
                                params=params,
                                headers=headers,
                                timeout=_TIMEOUT_SECONDS,
                                **kwargs)
        handle_error(response)

        return response

    def patch(self, url: str, body: Optional[dict], params: Optional[dict],
              **kwargs) -> requests.Response:
        """Sends a PATCH request.

        A wrapper for the requests.patch method.

        Builds the url, uses custom headers, refresh tokens if needed.

        :param url: relative url of the API endpoint
        :type url: str
        :param json: Python object to send in the body of the Request
        :type json: dict, optional
        :param params: Dictionary of querystring data to attach to the Request
        :type params: dict, optional

        :raises APIException: an api exception with message and error type code

        :return: Response object
        :rtype: requests.Response
        """
        self._refresh_token_if_expired()

        url = self._add_base_url(url)
        headers = self._generate_headers()

        response = requests.patch(url,
                                  json=body,
                                  headers=headers,
                                  params=params,
                                  timeout=_TIMEOUT_SECONDS,
                                  **kwargs)
        handle_error(response)

        return response

    def delete(self,
               url: str,
               body: Optional[dict] = None,
               params: Optional[dict] = None,
               **kwargs) -> requests.Response:
        """Sends a DELETE request.

        A wrapper for the requests.delete method.

        Builds the url, uses custom headers, refresh tokens if needed.

        :param url: relative url of the API endpoint
        :type url: str
        :param json: Python object to send in the body of the Request
        :type json: dict, optional
        :param params: Dictionary of querystring data to attach to the Request
        :type params: dict, optional

        :raises APIException: an api exception with message and error type code

        :return: Response object
        :rtype: requests.Response
        """
        self._refresh_token_if_expired()

        url = self._add_base_url(url)
        headers = self._generate_headers()

        response = requests.delete(url,
                                   headers=headers,
                                   json=body,
                                   params=params,
                                   timeout=_TIMEOUT_SECONDS,
                                   **kwargs)
        handle_error(response)

        return response

    def _refresh_token_if_expired(self) -> None:
        """Refreshes the access token if it expired.

        Uses the refresh token to refresh, and if the refresh token is also
        expired, uses the client credentials to authenticate again.

        :raises APIException: an api exception with message and error type code
        """
        if self._auth_service.is_expired():
            # try to refresh. if refresh token has expired, reauthenticate
            try:
                self._auth_service.refresh()
            except Exception:  # pylint: disable=broad-except
                self._auth_service.authenticate()

    def _generate_headers(self) -> dict:
        """Generate the default headers for every request.

        :return: dict with request headers
        :rtype: dict
        """
        headers = self._auth_service.generate_headers()
        headers.update({
            'Content-Type': 'application/json',
        })
        return headers

    def _add_base_url(self, url: str) -> str:
        """Adds the base url to the relative url.

        Example:
        if the relative url is '/balance'
        and the base url is 'https://api.verda.com/v1'
        then this method will return 'https://api.verda.com/v1/balance'

        :param url: a relative url path
        :type url: str
        :return: the full url path
        :rtype: str
        """
        return self._base_url + url


# https://api.verda.com/v1/docs#tag/instances/GET/v1/instances
class InstanceStatus:
    """Instance status."""

    # The API documents 13 statuses. STARTING_HIBERNATION, HIBERNATING and
    # RESTORING come from the older DataCrunch SDK and are not documented.
    ORDERED = 'ordered'
    RUNNING = 'running'
    PROVISIONING = 'provisioning'
    OFFLINE = 'offline'
    STARTING_HIBERNATION = 'starting_hibernation'
    HIBERNATING = 'hibernating'
    RESTORING = 'restoring'
    ERROR = 'error'
    DISCONTINUED = 'discontinued'
    UNKNOWN = 'unknown'
    NOTFOUND = 'notfound'
    NEW = 'new'
    DELETING = 'deleting'
    VALIDATING = 'validating'
    NO_CAPACITY = 'no_capacity'
    INSTALLATION_FAILED = 'installation_failed'


class Instance:
    """Instance model class."""

    def __init__(self, data) -> None:
        self.instance_id = data['id']
        self.status = data['status']
        self.hostname = data['hostname']
        # For not yet provisioned instances, ip is not available
        self.ip = data.get('ip')
        # https://api.verda.com/v1/docs#tag/instances/GET/v1/instances
        self.os_volume_id = data.get('os_volume_id')


class SSHKey:
    """An SSH key model class."""

    def __init__(self, data) -> None:
        """Initialize a new SSH key object.

        :param data: JSON data
        :type id: dict
        """
        self.id = data['id']
        self.name = data['name']
        self.public_key = data['key']


class VerdaClient:
    """A client for the Verda Cloud API."""

    def __init__(self) -> None:
        self.http_client: Optional[_HTTPClient] = None

    def instances_get(self) -> List[Instance]:
        """Get all instances."""
        if self.http_client is None:
            self.http_client = _HTTPClient()
        response = self.http_client.get('/instances').json()
        return [Instance(o) for o in response]

    def instance_get(self, instance_id: str) -> Instance:
        """Get instance."""
        if self.http_client is None:
            self.http_client = _HTTPClient()
        response = self.http_client.get(f'/instances/{instance_id}').json()
        return Instance(response)

    # https://api.verda.com/v1/docs#tag/os-images/GET/v1/images
    def images_get(self, instance_type: str):
        if self.http_client is None:
            self.http_client = _HTTPClient()
        response = self.http_client.get('/images',
                                        params={'instance_type': instance_type})
        return [image['image_type'] for image in response.json()]

    # https://api.verda.com/v1/docs#tag/instance-types/GET/v1/instance-types
    def instance_types_get(self):
        if self.http_client is None:
            self.http_client = _HTTPClient()
        return self.http_client.get('/instance-types').json()

    # https://api.verda.com/v1/docs#tag/locations/GET/v1/locations
    def locations_get(self):
        if self.http_client is None:
            self.http_client = _HTTPClient()
        return [
            location['code']
            for location in self.http_client.get('/locations').json()
        ]

    # https://api.verda.com/v1/docs#tag/instance-availability/GET/v1/instance-availability
    def instance_availability_get(self, is_spot: bool):
        if self.http_client is None:
            self.http_client = _HTTPClient()
        params = {'is_spot': 'true' if is_spot else 'false'}
        response = self.http_client.get('/instance-availability',
                                        params=params).json()
        return {(instance_type, location['location_code'])
                for location in response
                for instance_type in location['availabilities']}

    def ssh_keys_get(self) -> List[SSHKey]:
        """Get all ssh keys."""
        if self.http_client is None:
            self.http_client = _HTTPClient()
        response = self.http_client.get('/ssh-keys').json()
        return [SSHKey(o) for o in response]

    def ssh_keys_create(self, name: str, key: str) -> SSHKey:
        """Create a new ssh key."""
        if self.http_client is None:
            self.http_client = _HTTPClient()
        payload = {'name': name, 'key': key}
        key_id = self.http_client.post('/ssh-keys', body=payload).text
        return SSHKey({'id': key_id, 'name': name, 'key': key})

    def instance_create(self, payload: dict) -> Instance:
        if self.http_client is None:
            self.http_client = _HTTPClient()
        instance_id = self.http_client.post('/instances', body=payload).text
        instance = self.instance_get(instance_id)
        return instance

    def instance_action(self,
                        instance_id: str,
                        action: str,
                        volume_ids: Optional[List[str]] = None) -> None:
        if self.http_client is None:
            self.http_client = _HTTPClient()
        payload: Dict[str, Any] = {'id': [instance_id], 'action': action}
        if volume_ids:
            # https://api.verda.com/v1/docs#tag/instances/PUT/v1/instances
            # https://api.verda.com/v1/docs#description/2026-02-03-delete-volumes-permanently-when-deleting-an-instance
            payload['volume_ids'] = volume_ids
            payload['delete_permanently'] = True
        self.http_client.put('/instances', body=payload)
        return None


@contextlib.contextmanager
def _load_verda_s3_credentials_env():
    """Context manager to temporarily change the AWS credentials file path."""
    prev_credentials_path = os.environ.get('AWS_SHARED_CREDENTIALS_FILE')
    prev_config_path = os.environ.get('AWS_CONFIG_FILE')
    credentials_path, config_path = local_s3_files()
    os.environ['AWS_SHARED_CREDENTIALS_FILE'] = credentials_path
    os.environ['AWS_CONFIG_FILE'] = config_path
    try:
        yield
    finally:
        if prev_credentials_path is None:
            del os.environ['AWS_SHARED_CREDENTIALS_FILE']
        else:
            os.environ['AWS_SHARED_CREDENTIALS_FILE'] = prev_credentials_path
        if prev_config_path is None:
            del os.environ['AWS_CONFIG_FILE']
        else:
            os.environ['AWS_CONFIG_FILE'] = prev_config_path


def get_verda_s3_credentials(boto3_session):
    """Gets the Verda object storage credentials from a boto3 session.

    Args:
        boto3_session: The boto3 session object.
    Returns:
        botocore.credentials.ReadOnlyCredentials object with the Verda
        object storage credentials.
    """
    with _load_verda_s3_credentials_env():
        verda_credentials = boto3_session.get_credentials()
        if verda_credentials is None:
            with ux_utils.print_exception_no_traceback():
                raise ValueError('Verda object storage credentials not found. '
                                 'Run `sky check` to verify credentials are '
                                 'correctly set up.')
        return verda_credentials.get_frozen_credentials()


@annotations.lru_cache(scope='global')
def session():
    """Create an AWS session for Verda object storage."""
    # Creating the session object is not thread-safe for boto3,
    # so we add a reentrant lock to synchronize the session creation.
    # Reference: https://github.com/boto/boto3/issues/1592
    with _session_creation_lock:
        with _load_verda_s3_credentials_env():
            session_ = boto3.session.Session(profile_name=VERDA_S3_PROFILE_NAME)
        return session_


@annotations.lru_cache(scope='global')
def resource(resource_name: str, **kwargs):
    """Create a Verda object storage resource.

    Args:
        resource_name: Verda resource name (e.g., 's3').
        kwargs: Other options.
    """
    # Need to use the resource retrieved from the per-thread session
    # to avoid thread-safety issues (Directly creating the client
    # with boto3.resource() is not thread-safe).
    # Reference: https://stackoverflow.com/a/59635814
    session_ = session()
    verda_credentials = get_verda_s3_credentials(session_)

    return session_.resource(
        resource_name,
        endpoint_url=get_endpoint(),
        aws_access_key_id=verda_credentials.access_key,
        aws_secret_access_key=verda_credentials.secret_key,
        region_name=DEFAULT_REGION,
        config=botocore.config.Config(s3={'addressing_style': 'path'}),
        **kwargs)


@annotations.lru_cache(scope='global')
def client(service_name: str):
    """Create a Verda object storage client of a certain service.

    Args:
        service_name: Verda service name (e.g., 's3').
    """
    # Need to use the client retrieved from the per-thread session
    # to avoid thread-safety issues (Directly creating the client
    # with boto3.client() is not thread-safe).
    # Reference: https://stackoverflow.com/a/59635814
    session_ = session()
    verda_credentials = get_verda_s3_credentials(session_)

    return session_.client(
        service_name,
        endpoint_url=get_endpoint(),
        aws_access_key_id=verda_credentials.access_key,
        aws_secret_access_key=verda_credentials.secret_key,
        region_name=DEFAULT_REGION,
        config=botocore.config.Config(s3={'addressing_style': 'path'}),
    )


@common.load_lazy_modules(_LAZY_MODULES)
def botocore_exceptions():
    """AWS botocore exception."""
    # pylint: disable=import-outside-toplevel
    from botocore import exceptions as boto_exceptions
    return boto_exceptions


def get_endpoint():
    """Parse the VERDA_S3_CONFIG_PATH to get the endpoint_url.

    The config file is an AWS-style config file with format:
        [profile verda]
        endpoint_url = https://objects.fin-03.verda.storage

    Returns:
        str: The endpoint URL from the config file, or the default endpoint
             if the file doesn't exist or doesn't contain the endpoint_url.
    """
    if not (verda_s3_profile_in_cred() and verda_s3_profile_in_config()):
        section = _cli_s3_section()
        if section is not None:
            return section.get('verda_s3_endpoint', _DEFAULT_ENDPOINT)
    config_path = os.path.expanduser(VERDA_S3_CONFIG_PATH)
    if not os.path.isfile(config_path):
        return _DEFAULT_ENDPOINT

    try:
        config = configparser.ConfigParser()
        config.read(config_path)

        profile_section = f'profile {VERDA_S3_PROFILE_NAME}'
        if config.has_section(profile_section):
            if config.has_option(profile_section, 'endpoint_url'):
                endpoint = config.get(profile_section, 'endpoint_url')
                return endpoint.strip()
    except (configparser.Error, OSError) as e:
        logger.warning(f'Failed to parse Verda object storage config file: '
                       f'{e}. Using default endpoint: {_DEFAULT_ENDPOINT}')

    return _DEFAULT_ENDPOINT


def _s3_profile_exists(file_path: str, header: str) -> bool:
    expanded = os.path.expanduser(file_path)
    if not os.path.isfile(expanded):
        return False
    with open(expanded, 'r', encoding='utf-8') as f:
        return any(header in line for line in f)


def verda_s3_profile_in_cred() -> bool:
    """Checks if the Verda profile is set in the S3 credentials file."""
    return _s3_profile_exists(VERDA_S3_CREDENTIALS_PATH,
                              f'[{VERDA_S3_PROFILE_NAME}]')


def verda_s3_profile_in_config() -> bool:
    """Checks if the Verda profile is set in the S3 config file."""
    return _s3_profile_exists(VERDA_S3_CONFIG_PATH,
                              f'[profile {VERDA_S3_PROFILE_NAME}]')


# https://docs.verda.com/cli/object-storage/#configure-credentials
# https://docs.verda.com/cli/object-storage/#environment-variables
def _cli_s3_section():
    path = os.path.expanduser(_CLI_CREDENTIALS_PATH)
    if not os.path.isfile(path):
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read(path)
    except configparser.Error:
        return None
    profile = os.environ.get('VERDA_PROFILE', 'default')
    if not parser.has_section(profile):
        return None
    section = parser[profile]
    if not (section.get('verda_s3_access_key') and
            section.get('verda_s3_secret_key')):
        return None
    return section


# Write to a temporary file and rename it, so a reader (the AWS CLI, goofys,
# rclone or another API server worker) never sees a partly written file.
def _write_private(path: str, content: str):
    directory = os.path.dirname(path)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',
                                     encoding='utf-8',
                                     dir=directory,
                                     delete=False) as f:
        temporary_path = f.name
        try:
            f.write(content)
            f.close()
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)


def get_compute_credential_file_mounts():
    # Credentials can come from VERDA_* or DATACRUNCH_* env vars instead of
    # ~/.verda/config.json, so write the resolved config to a file the remote
    # can mount. The hash in the path changes the mount when the credentials
    # change.
    configured, _, config = get_verda_configuration()
    if not configured or config is None:
        return {}
    content = json.dumps(dataclasses.asdict(config), sort_keys=True)
    digest = hashlib.sha256(content.encode()).hexdigest()
    path = os.path.expanduser(f'{_GENERATED_DIR}/compute/{digest}/config.json')
    _write_private(path, content)
    return {'~/.verda/config.json': path}


def local_s3_files(write: bool = True):
    if verda_s3_profile_in_cred() and verda_s3_profile_in_config():
        return VERDA_S3_CREDENTIALS_PATH, VERDA_S3_CONFIG_PATH
    section = _cli_s3_section()
    if section is None:
        return VERDA_S3_CREDENTIALS_PATH, VERDA_S3_CONFIG_PATH
    credentials = (
        f'[{VERDA_S3_PROFILE_NAME}]\n'
        f'aws_access_key_id = {section["verda_s3_access_key"]}\n'
        f'aws_secret_access_key = {section["verda_s3_secret_key"]}\n')
    config = (f'[profile {VERDA_S3_PROFILE_NAME}]\n'
              'endpoint_url = '
              f'{section.get("verda_s3_endpoint", _DEFAULT_ENDPOINT)}\n'
              f'region = {DEFAULT_REGION}\n')
    # One folder per set of keys, so profiles do not overwrite each other.
    digest = hashlib.sha256((credentials + config).encode()).hexdigest()
    directory = os.path.expanduser(f'{_GENERATED_DIR}/s3/{digest}')
    credentials_path = os.path.join(directory, 's3.credentials')
    config_path = os.path.join(directory, 's3.config')
    if write:
        _write_private(credentials_path, credentials)
        _write_private(config_path, config)
    return credentials_path, config_path


def s3_credentials_configured():
    return ((verda_s3_profile_in_cred() and verda_s3_profile_in_config()) or
            _cli_s3_section() is not None)


def get_s3_credential_file_mounts() -> Dict[str, str]:
    """Returns the Verda object storage credential file mounts."""
    if not s3_credentials_configured():
        return {}
    credentials_path, config_path = local_s3_files()
    return {
        VERDA_S3_CREDENTIALS_PATH: credentials_path,
        VERDA_S3_CONFIG_PATH: config_path,
    }
