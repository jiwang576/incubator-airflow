#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the 'License'); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import re

from airflow import settings
from airflow.operators import BaseOperator
from airflow.contrib.hooks.gcp_cloudml_hook import CloudMLHook

from airflow.utils.decorators import apply_defaults
from apiclient import errors

logging.getLogger('GoogleCloudML').setLevel(settings.LOGGING_LEVEL)


def _normalize_cloudml_job_id(job_id):
    """Replaces invalid CloudML job_id characters with '_'.

    This also adds a leading 'z' in case job_id starts with an invalid
    character.

    Args:
        job_id: A job_id str that may have invalid characters.

    Returns:
        A valid job_id representation.
    """
    match = re.search(r'\d', job_id)
    if match and match.start() is 0:
        job_id = 'z_{}'.format(job_id)
    return re.sub('[^0-9a-zA-Z]+', '_', job_id)


def _create_prediction_input(project_id,
                             region,
                             data_format,
                             input_paths,
                             output_path,
                             model_name=None,
                             version_name=None,
                             uri=None,
                             max_worker_count=None,
                             runtime_version=None):
    """Create the batch prediction input from the given parameters.

    Args:
        A subset of arguments documented in __init__ method of class
        CloudMLBatchPredictionOperator

    Returns:
        A dictionary mirroring the predictionInput object as documented
        in https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs.

    Raises:
        ValueError: if a legal predictionInput cannot be constructed
        from the inputs.
    """

    if data_format not in ['TEXT', 'TF_RECORD', 'TF_RECORD_GZIP']:
        logging.warning(
            'The input data format is not recognized. Default to '
            'DATA_FORMAT_UNSPECIFIED.')
        data_format = 'DATA_FORMAT_UNSPECIFIED'

    try:
        for input_path in input_paths:
            _validate_gcs_uri(input_path)
    except ValueError as e:
        raise ValueError('Illegal input path. ' + str(e))

    try:
        _validate_gcs_uri(output_path)
    except ValueError as e:
        raise ValueError('Illegal output path. ' + str(e))

    prediction_input = {
        'dataFormat': data_format,
        'inputPaths': input_paths,
        'outputPath': output_path,
        'region': region
    }

    if uri:
        if model_name or version_name:
            logging.error(
                'Ambiguous model origin: Both uri and model/version name are '
                'provided.')
            raise ValueError('Ambiguous model origin.')
        try:
            _validate_gcs_uri(uri)
            prediction_input['uri'] = uri
        except ValueError as e:
            logging.error('Illegal uri: ' + uri)
            raise e

    elif version_name:
        if not model_name:
            logging.error(
                'Incomplete model origin: A version name is provided but the '
                'corresponding model name is missing.')
            raise ValueError('Incomplete version origin.')
        try:
            prediction_input['versionName'] = _create_origin_name(
                project_id, model_name, version_name)
        except ValueError as e:
            logging.error('Error constructing the model version origin.')
            raise e

    elif model_name:
        try:
            prediction_input['modelName'] = _create_origin_name(
                project_id, model_name)
        except ValueError as e:
            logging.error('Error constructing the model origin.')
            raise e

    else:
        logging.error(
            'Missing model origin: Batch prediction expects a model, '
            'a version, or a URI to savedModel.')
        raise ValueError('Missing model version origin.')

    # Check for non-integer string or non-positive input
    if max_worker_count:
        try:
            count = int(max_worker_count)
            if count < 0:
                raise ValueError(
                    'The maximum worker count cannot be non-positive.')
            prediction_input['maxWorkerCount'] = max_worker_count
        except ValueError as e:
            raise e

    if runtime_version:
        prediction_input['runtimeVersion'] = runtime_version

    return prediction_input


def _validate_gcs_uri(uri):
    """Verifies the given uri is a legal GCS location.

    The validation criteria for GCS bucket and object names are
    documented at:
    https://cloud.google.com/storage/docs/naming.

    Args:
        uri: The URI to be validated; string

    Raises:
        ValueError: if the URI is invalid.
    """
    if not re.match(r'^gs://', uri):
        raise ValueError('Missing GCS scheme in the URI: {}.'.format(uri))

    try:
        bucket_name, object_name = uri[len(r'gs://'):].split('/', 1)
    except ValueError:
        raise ValueError('Illegal URI for a GCS object: {}.'.format(uri))

    # Validation for bucket name
    if not re.match(r'^[a-zA-Z0-9][-.\w]*[a-zA-Z0-9]$', bucket_name):
        raise ValueError('Illegal GCS bucket name: {}.'.format(bucket_name))
    if '.' in bucket_name and len(bucket_name) > 222:
        raise ValueError(
            'GCS bucket name is longer than 222 bytes: {}.'
            .format(bucket_name))
    for dot_separated_component in bucket_name.split('.'):
        if not 3 <= len(dot_separated_component) <= 63:
            raise ValueError(
                'Each dot-eseparated component in GCS bucket name should be '
                'within 3 and 63 characters: {}.'.format(bucket_name))
    if re.match(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', bucket_name):
        raise ValueError(
            'GCS bucket name must not represent a IPv4-like sequence: {}.'
            .format(bucket_name))
    if re.match(r'(google|^goog)', bucket_name):
        raise ValueError(
            'GCS bucket name should not start with "goog" prefix, or contain '
            '"google" inside the name: {}.'.format(bucket_name))
    if re.match(r'(\.[-.])|(-\.)', bucket_name):
        raise ValueError(
            'Bucket names must not contain a dot next to either a dot or a '
            'dash: {}.'.format(bucket_name))

    # Validation for object name
    try:
        object_name_utf8 = object_name.decode('utf-8')
    except AttributeError:
        object_name_utf8 = object_name

    if not 1 <= len(object_name_utf8) <= 1024:
        raise ValueError(
            'GCS object name should be within 1 and 1024 bytes in length: {}.'.
            format(uri))
    if r'\r' in object_name_utf8 or r'\n' in object_name_utf8:
        raise ValueError(
            'GCS object name must not contain {} characters: {}.'
            .format(
                'Carriage Return' if r'\r' in object_name_utf8 else 'Line Feed',
                uri))


def _create_origin_name(project_id, model_name, version_name=None):
    """Create the name of a CloudML model/version origin.

    Args:
        project_id: The GCP project name; string
        model_name: The Cloud ML model name; string
        version_name: The Cloud ML model version name; string, optional

    Returns:
        A model/version origin name in the form
        "projects/<project_id>/models/<model_name>[/versions/<version_name>]"

    Raises:
        ValueError, when any of the three inputs is not a valid
        CloudML resource name, as documented in
        https://cloud.google.com/ml-engine/docs/how-tos/managing-models-jobs.
    """
    try:
        _validate_project_id(project_id)
        origin_name = 'projects/{}'.format(project_id)
    except ValueError as e:
        raise ValueError('Error parsing the project id. ' + str(e))

    try:
        _validate_resource_name(model_name)
        origin_name += '/models/{}'.format(model_name)
    except ValueError as e:
        raise ValueError('Error parsing the model name. ' + str(e))

    if version_name:
        try:
            _validate_resource_name(version_name)
            origin_name += '/versions/{}'.format(version_name)
        except ValueError as e:
            raise ValueError('Error parsing the version name. ' + str(e))

    return origin_name


def _validate_project_id(project_id):
    """Verifies the given project_id is legal.

    The validation criteria is documented as "projectID" at:
    https://cloud.google.com/resource-manager/reference/rest/v1/projects#Project.

    Args:
        project_id: Name of the project to be verified; string

    Raises:
        ValueError: if the project name is invalid

    """
    if not 6 <= len(project_id) <= 30:
        raise ValueError(
            'Project id should be within 6 and 30 characters in length: {}.'
            .format(project_id))
    if not re.match(r'^[a-z][-a-z0-9]*[^-]$', project_id):
        raise ValueError('Illegal project id: {}.'.format(project_id))


def _validate_resource_name(resource_name):
    """Verifies the given resource name is legal.
    The name could refer to a model, a version, or a job.

    The validation criteria is documented at:
    https://cloud.google.com/ml-engine/docs/how-tos/managing-models-jobs

    Args:
        resource_name: The resource name to be verified; string

    Raises:
        ValueError: if the resource name is invalid.

    """
    if not 1 <= len(resource_name) <= 128:
        raise ValueError(
            'Resource name should be within 1 and 128 characters in length: {}.'
            .format(resource_name))
    if not re.match(r'^[a-zA-Z]\w*$', resource_name):
        raise ValueError('The resource name is illegal: {}.'.format(
            resource_name))


class CloudMLBatchPredictionOperator(BaseOperator):
    """Start a Cloud ML prediction job.

    NOTE: For model origin, users should consider exactly one from the
    three options below:
    1. Populate 'uri' field only, which should be a GCS location that
    points to a tensorflow savedModel directory.
    2. Populate 'model_name' field only, which refers to an existing
    model, and the default version of the model will be used.
    3. Populate both 'model_name' and 'version_name' fields, which
    refers to a specific version of a specific model.

    In options 2 and 3, both model and version name should contain the
    minimal identifier. For instance, call
    CloudMLBatchPredictionOperator(..., model_name='my_model',
    version_name='my_version', ...)
    if the desired version is
    "projects/my_project/models/my_model/versions/my_version".

    :param project_id: The Google Cloud project name where the
        prediction job is submitted.
    :type project_id: string

    :param job_id: A unique id for the prediction job on Google Cloud
        ML Engine.
    :type job_id: string

    :param data_format: The format of the input data.
        It will default to 'DATA_FORMAT_UNSPECIFIED' if is not provided
        or is not one of ["TEXT", "TF_RECORD", "TF_RECORD_GZIP"].
    :type data_format: string

    :param input_paths: A list of GCS paths of input data for batch
        prediction. Accepting wildcard operator *, but only at the end.
    :type input_paths: list of string

    :param output_path: The GCS path where the prediction results are
        written to.
    :type output_path: string

    :param region: The Google Compute Engine region to run the
        prediction job in.:
    :type region: string

    :param model_name: The Google Cloud ML model to use for prediction.
        If version_name is not provided, the default version of this
        model will be used.
        Should not be None if version_name is provided.
        Should be None if uri is provided.
    :type model_name: string

    :param version_name: The Google Cloud ML model version to use for
        prediction.
        Should be None if uri is provided.
    :type version_name: string

    :param uri: The GCS path of the saved model to use for prediction.
        Should be None if model_name is provided.
        It should be a GCS path pointing to a tensorflow SavedModel.
    :type uri: string

    :param max_worker_count: The maximum number of workers to be used
        for parallel processing. Defaults to 10 if not specified.
    :type max_worker_count: int

    :param runtime_version: The Google Cloud ML runtime version to use
        for batch prediction.
    :type runtime_version: string

    :param gcp_conn_id: The connection ID used for connection to Google
        Cloud Platform.
    :type gcp_conn_id: string

    :param delegate_to: The account to impersonate, if any.
        For this to work, the service account making the request must
        have doamin-wide delegation enabled.
    :type delegate_to: string

    Raises:
        ValueError: when wrong arguments are given.
    """

    template_fields = [
        'project_id', 'job_id', 'input_paths', 'output_path', 'model_name',
        'version_name', 'uri'
    ]

    @apply_defaults
    def __init__(self,
                 project_id,
                 job_id,
                 region,
                 data_format,
                 input_paths,
                 output_path,
                 model_name=None,
                 version_name=None,
                 uri=None,
                 max_worker_count=None,
                 runtime_version=None,
                 gcp_conn_id='google_cloud_default',
                 delegate_to=None,
                 *args,
                 **kwargs):
        super(CloudMLBatchPredictionOperator, self).__init__(*args, **kwargs)

        self.project_id = project_id
        self.job_id = job_id
        self.region = region
        self.data_format = data_format
        self.input_paths = input_paths
        self.output_path = output_path
        self.model_name = model_name
        self.version_name = version_name
        self.uri = uri
        self.max_worker_count = max_worker_count
        self.runtime_version = runtime_version
        self.gcp_conn_id = gcp_conn_id
        self.delegate_to = delegate_to

    def execute(self, context):
        try:
            prediction_job_request = self._create_valid_job_request()
        except ValueError as e:
            raise e

        hook = CloudMLHook(self.gcp_conn_id, self.delegate_to)
        job_id = _normalize_cloudml_job_id(prediction_job_request['jobId'])
        try:
            existing_job = hook.get_job(self.project_id, job_id)
            logging.info(
                'Job with job_id {} already exist: {}.'.format(
                    job_id,
                    existing_job))
            finished_prediction_job = hook.wait_for_job_done(
                self.project_id,
                job_id)
        except errors.HttpError as e:
            if e.resp.status == 404:
                logging.error(
                    'Job with job_id {} does not exist. Will create it.'
                    .format(job_id))
                finished_prediction_job = hook.create_job(
                    self.project_id,
                    prediction_job_request)
            else:
                raise e

        if finished_prediction_job['state'] != 'SUCCEEDED':
            logging.error(
                'Batch prediction job failed: %s',
                str(finished_prediction_job))
            raise RuntimeError(finished_prediction_job['errorMessage'])

        return finished_prediction_job['predictionOutput']

    def _create_valid_job_request(self):
        try:
            _validate_resource_name(self.job_id)
        except ValueError as e:
            logging.error(
                'Cannot create batch prediction job request due to: {}'
                .format(str(e)))
            raise ValueError('Illegal job id. ' + str(e))

        try:
            prediction_input = _create_prediction_input(
                self.project_id, self.region, self.data_format,
                self.input_paths, self.output_path, self.model_name,
                self.version_name, self.uri, self.max_worker_count,
                self.runtime_version)
        except ValueError as e:
            logging.error(
                'Cannot create batch prediction job request due to: {}'
                .format(str(e)))
            raise e

        return {
            'jobId': self.job_id,
            'predictionInput': prediction_input}
