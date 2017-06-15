#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import copy
import datetime
import json
import logging
import os
import re
import urlparse

import airflow
from airflow.contrib.hooks.gcs_hook import GoogleCloudStorageHook
from airflow.contrib.operators.dataflow_operator import DataFlowPythonOperator
from airflow.operators import BaseOperator

# hack to make import successful on both EnCom and Airflow repo
# TODO(jiwangmtv) remove as appropriate when pushing to github.
try:
  from airflow.operators.python_operator import PythonOperator
except ImportError:
  from airflow.operators import PythonOperator

# TODO(jiwangmtv) remove as appropriate when pushing to github.
try:
  from airflow.contrib.hooks.gcp_cloudml_hook import CloudMLHook
except ImportError:
  from gcp_cloudml_hook import CloudMLHook

from airflow.utils.decorators import apply_defaults
from apiclient import errors

logging.getLogger("GoogleCloudML").setLevel(logging.INFO)


def _normalize_cloudml_job_id(job_id):
  """Replaces invalid CloudML job_id characters with '_'.

  This also adds a leading 'z' in case job_id starts with an invalid character.

  Args:
    job_id: A job_id str that may have invalid characters.

  Returns:
    A valid job_id representation.
  """
  match = re.search(r"\d", job_id)
  if match and match.start() is 0:
    job_id = "z_{}".format(job_id)
  return re.sub("[^0-9a-zA-Z]+", "_", job_id)


def _verify_dataflow_task_id(task_id):
  """Verify whether the proper task_id for DataFlowPythonOperator is used.

  DataFlow allows alpha numeric characters and dash (-) but not underscore (_)
  in the --job_name, which is derived from task_id.

  Args:
    task_id: a string

  Raises:
    ValueError, when task_id includes invalid characters.
  """
  if not re.match(r"^[a-zA-Z][-A-Za-z0-9]*$", task_id):
    raise ValueError("Malformed task_id for DataFlowPythonOperator: " + task_id)


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
    A subset of arguments documented in CloudMLBatchPredictionOperator

  Returns:
    A dictionary mirroring the predictionInput object as documented in
    https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#predictioninput.

  Raises:
    ValueError: if a legal predictionInput cannot be constructed from the
    inputs.
  """

  if data_format not in ["TEXT", "TF_RECORD", "TF_RECORD_GZIP"]:
    logging.warning("The input data format is not recognized. Default to "
                    "DATA_FORMAT_UNSPECIFIED.")
    data_format = "DATA_FORMAT_UNSPECIFIED"

  try:
    for input_path in input_paths:
      _validate_gcs_uri(input_path)
  except ValueError as e:
    raise ValueError("Illegal input path. " + str(e))

  try:
    _validate_gcs_uri(output_path)
  except ValueError as e:
    raise ValueError("Illegal output path. " + str(e))

  prediction_input = {
      "dataFormat": data_format,
      "inputPaths": input_paths,
      "outputPath": output_path,
      "region": region
  }

  if uri:
    if model_name or version_name:
      logging.error(
          "Ambiguous model origin: Both uri and model/version name are "
          "provided.")
      raise ValueError("Ambiguous model origin.")
    try:
      _validate_gcs_uri(uri)
      prediction_input["uri"] = uri
    except ValueError as e:
      logging.error("Illegal uri: " + uri)
      raise e

  elif version_name:
    if not model_name:
      logging.error(
          "Incomplete model origin: A version name is provided but the "
          "corresponding model name is missing.")
      raise ValueError("Incomplete version origin.")
    try:
      prediction_input["versionName"] = _create_origin_name(
          project_id, model_name, version_name)
    except ValueError as e:
      logging.error("Error constructing the model version origin.")
      raise e

  elif model_name:
    try:
      prediction_input["modelName"] = _create_origin_name(
          project_id, model_name)
    except ValueError as e:
      logging.error("Error constructing the model origin.")
      raise e

  else:
    logging.error("Missing model origin: Batch prediction expects a model, a "
                  "version, or a URI to savedModel.")
    raise ValueError("Missing model version origin.")

  # Check for non-integer string or non-positive input
  if max_worker_count:
    try:
      count = int(max_worker_count)
      if count < 0:
        raise ValueError("The maximum worker count cannot be non-positive.")
      prediction_input["maxWorkerCount"] = max_worker_count
    except ValueError as e:
      raise e

  if runtime_version:
    prediction_input["runtimeVersion"] = runtime_version

  return prediction_input


def _validate_gcs_uri(uri):
  """Verifies the given uri is a legal GCS location.

  The validation criteria for GCS bucket and object names are documented at:
  https://cloud.google.com/storage/docs/naming.

  Args:
    uri: The URI to be validated; string

  Raises:
    ValueError: if the URI is invalid.
  """
  if not re.match(r"^gs://", uri):
    raise ValueError("Missing GCS scheme in the URI: {}.".format(uri))

  try:
    bucket_name, object_name = uri[len(r"gs://"):].split("/", 1)
  except ValueError:
    raise ValueError("Illegal URI for a GCS object: {}.".format(uri))

  # Validation for bucket name
  if not re.match(r"^[a-zA-Z0-9][-.\w]*[a-zA-Z0-9]$", bucket_name):
    raise ValueError("Illegal GCS bucket name: {}.".format(bucket_name))
  if "." in bucket_name and len(bucket_name) > 222:
    raise ValueError(
        "GCS bucket name is longer than 222 bytes: {}.".format(bucket_name))
  for dot_separated_component in bucket_name.split("."):
    if not 3 <= len(dot_separated_component) <= 63:
      raise ValueError(
          "Each dot-eseparated component in GCS bucket name should be within 3"
          " and 63 characters: {}.".format(bucket_name))
  if re.match(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", bucket_name):
    raise ValueError(
        "GCS bucket name should represent a IPv4-like sequence: {}.".format(
            bucket_name))
  if re.match(r"(google|^goog)", bucket_name):
    raise ValueError(
        "GCS bucket name should not start with 'goog' prefix, or contain "
        "'google' inside the name: {}.".format(bucket_name))
  if re.match(r"(\.[-.])|(-\.)", bucket_name):
    raise ValueError(
        "Bucket names must not contain a dot next to either a dot or a dash: {}.".
        format(bucket_name))

  # Validation for object name
  object_name_utf8 = object_name.decode("utf-8")
  if not 1 <= len(object_name_utf8) <= 1024:
    raise ValueError(
        "GCS object name should be within 1 and 1024 bytes in length: {}.".
        format(uri))
  if r"\r" in object_name_utf8 or r"\n" in object_name_utf8:
    raise ValueError("GCS object name must not contain {} characters: {}.".
                     format("Carriage Return"
                            if r"\r" in object_name_utf8 else "Line Feed", uri))


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
    ValueError, when any of the three inputs is not a valid CloudML resource
    name, as documented in
    https://cloud.google.com/ml-engine/docs/how-tos/managing-models-jobs.
  """
  try:
    _validate_project_id(project_id)
    origin_name = "projects/{}".format(project_id)
  except ValueError as e:
    raise ValueError("Error parsing the project id. " + str(e))

  try:
    _validate_resource_name(model_name)
    origin_name += "/models/{}".format(model_name)
  except ValueError as e:
    raise ValueError("Error parsing the model name. " + str(e))

  if version_name:
    try:
      _validate_resource_name(version_name)
      origin_name += "/versions/{}".format(version_name)
    except ValueError as e:
      raise ValueError("Error parsing the version name. " + str(e))

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
        "Project id should be within 6 and 30 characters in length: {}.".format(
            project_id))
  if not re.match(r"^[a-z][-a-z0-9]*[^-]$", project_id):
    raise ValueError("Illegal project id: {}.".format(project_id))


def _validate_resource_name(resource_name):
  """Verifies the given resource name is legal.
  The name could refer to a model, a version, or a job.

  The validation criteria is documented at:
  https://cloud.google.com/ml-engine/docs/how-tos/managing-models-jobs.

  Args:
    resource_name: A single piece of resource name to be verified; string

  Raises:
    ValueError: if the resource name is invalid.

  """
  if not 1 <= len(resource_name) <= 128:
    raise ValueError(
        "Resource name should be within 1 and 128 characters in length: {}.".
        format(resource_name))
  if not re.match(r"^[a-zA-Z]\w*$", resource_name):
    raise ValueError("The resource name is illegal: {}.".format(resource_name))


class CloudMLVersionOperator(BaseOperator):
  template_fields = [
      "_model_name",
      "_version",
  ]

  @apply_defaults
  def __init__(self,
               version=None,
               cloudml_default_options={},
               gcp_conn_id="google_cloud_default",
               operation="create",
               *args,
               **kwargs):

    super(CloudMLVersionOperator, self).__init__(*args, **kwargs)

    self._model_name = cloudml_default_options.get("model_name")
    self._version = version

    self._gcp_conn_id = gcp_conn_id
    self._delegate_to = cloudml_default_options.get("delegate_to")
    self._project_name = cloudml_default_options.get("project")
    self._operation = operation

  def execute(self, context):
    hook = CloudMLHook(
        gcp_conn_id=self._gcp_conn_id, delegate_to=self._delegate_to)

    if self._operation == "create":
      assert self._version is not None
      return hook.create_version(self._project_name, self._model_name,
                                 self._version)

    elif self._operation == "set_default":
      return hook.set_default_version(
          self._project_name, self._model_name,
          self._version["name"])  # TODO: Do we need to return this?

    elif self._operation == "list":
      return hook.list_versions(self._project_name, self._model_name)

    elif self._operation == "delete":
      return hook.delete_version(self._project_name, self._model_name,
                                 self._version["name"])

    else:
      raise ValueError("Unknown operation: {}".format(self._operation))


class CloudMLModelOperator(BaseOperator):

  template_fields = [
      "_model",
  ]

  @apply_defaults
  def __init__(self,
               model,
               gcp_conn_id="google_cloud_default",
               project_name=None,
               delegate_to=None,
               operation="create",
               cloudml_default_options={},
               *args,
               **kwargs):
    super(CloudMLModelOperator, self).__init__(*args, **kwargs)
    self._gcp_conn_id = gcp_conn_id
    self._delegate_to = delegate_to
    self._project_name = project_name or cloudml_default_options.get("project")
    self._model = model
    self._operation = operation

  def execute(self, context):
    hook = CloudMLHook(
        gcp_conn_id=self._gcp_conn_id, delegate_to=self._delegate_to)
    if self._operation == "create":
      hook.create_model(self._project_name, self._model)
    elif self._operation == "get":
      hook.get_model(self._project_name, self._model["name"])
    else:
      raise ValueError("Unknown operation: {}".format(self._operation))


class CloudMLBatchPredictionOperator(BaseOperator):
  """Start a Cloud ML prediction job.

  NOTE: For model origin, users should consider exactly one from the three
  options below:
  1. Populate 'uri' field only, which should be a GCS location that points to a
  tensorflow savedModel directory.
  2. Populate 'model_name' field only, which refers to an existing model, and
  the default version of the model will be used.
  3. Populate both 'model_name' and 'version_name' fields, which refers to a
  specific version of a specific model.

  In options 2 and 3, both model and version name should contain the minimal
  identifier. For instance, call
  CloudMLBatchPredictionOperator(..., model_name='my_model',
  version_name='my_version', ...)
  if the desired version is
  "projects/my_project/models/my_model/versions/my_version".

  :param project_id: The Google Cloud project name where the prediction job is
  submitted
  :type project_id: string

  :param job_id: A unique id for the prediction job on Google Cloud ML Engine.
  :type job_id: string

  :param data_format: The format of the input data.
      It will default to 'DATA_FORMAT_UNSPECIFIED' if is not provided or is not
      one of ["TEXT", "TF_RECORD", "TF_RECORD_GZIP"].
  :type data_format: string

  :param input_paths: A list of GCS paths of input data for batch prediction.
      Accepting wildcard operator *, but only at the end.
  :type input_paths: list of string

  :param output_path: The GCS path where the prediction results are written to.
  :type output_path: string

  :param region: The Google Compute Engine region to run the prediction job in.:
  :type region: string

  :param model_name: The Google Cloud ML model to use for prediction.
      If version_name is not provided, the default version of this model will be
      used.
      Should not be None if version_name is provided.
      Should be None if uri is provided.
  :type model_name: string

  :param version_name: The Google Cloud ML model version to use for prediction.
      Should be None if uri is provided.
  :type version_name: string

  :param uri: The GCS path of the saved model to be used for prediction.
      Should be None if model_name is provided.
      It should be a GCS path pointing to a tensorflow SavedModel.
  :type uri: string

  :param max_worker_count: The maximum number of workers to be used for parallel
      processing. Defaults to 10 if not specified.
  :type max_worker_count: int

  :param runtime_version: The Google Cloud ML runtime version to use for this
      batch prediction.
  :type runtime_version: string

  :param gcp_conn_id: The connection ID to use connecting to Google Cloud
      Platform.
  :type gcp_conn_id: string

  :param delegate_to: The account to impersonate, if any.
      For this to work, the service account making the request must have
      doamin-wide delegation enabled.
  :type delegate_to: string

  Raises:
    ValueError: when wrong arguments are given.
  """

  template_fields = [
      "project_id", "job_id", "input_paths", "output_path", "model_name",
      "version_name", "uri"
  ]

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
               gcp_conn_id="google_cloud_default",
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
    job_id = _normalize_cloudml_job_id(prediction_job_request["jobId"])
    try:
      existing_job = hook.get_job(self.project_id, job_id)
      logging.info(
          "Job with job_id {} already exist: {}.".format(job_id, existing_job))
      finished_prediction_job = hook.wait_for_job_done(self.project_id, job_id)
    except errors.HttpError as e:
      if e.resp.status == 404:
        logging.error(
            "Job with job_id {} does not exist. Will create it.".format(job_id))
        finished_prediction_job = hook.create_job(self.project_id,
                                                  prediction_job_request)
      else:
        raise e

    if finished_prediction_job["state"] != "SUCCEEDED":
      logging.error("Batch prediction job failed: %s",
                    str(finished_prediction_job))
      raise RuntimeError(finished_prediction_job["errorMessage"])

    return finished_prediction_job["predictionOutput"]

  def _create_valid_job_request(self):
    try:
      _validate_resource_name(self.job_id)
    except ValueError as e:
      logging.error("Cannot create batch prediction job request due to: {}".
                    format(str(e)))
      raise ValueError("Illegal job id. " + str(e))

    try:
      prediction_input = _create_prediction_input(
          self.project_id, self.region, self.data_format, self.input_paths,
          self.output_path, self.model_name, self.version_name, self.uri,
          self.max_worker_count, self.runtime_version)
    except ValueError as e:
      logging.error("Cannot create batch prediction job request due to: {}".
                    format(str(e)))
      raise e

    return {"jobId": self.job_id, "predictionInput": prediction_input}


# TODO(jiwangmtv): Remove the class altogether if we decided it's useless.
class CloudMLBatchPredictionDataflowOperator(DataFlowPythonOperator):
  """ Launch a Dataflow job to run batch prediciton.
  """

  template_fields = ["options"]

  @apply_defaults
  def __init__(self,
               task_id,
               dag,
               model_uri,
               project_name,
               job_id,
               input_paths,
               prediction_path,
               dataflow_options,
               input_file_format,
               return_input=True,
               *args,
               **kwargs):
    _verify_dataflow_task_id(task_id)
    super(CloudMLBatchPredictionDataflowOperator, self).__init__(
        task_id=task_id,
        dag=dag,
        py_file=os.path.join(
            os.getenv("DAGS_FOLDER"), "batch_prediction_dataflow.py"),
        dataflow_default_options=dataflow_options,
        options={
            # This is to get around old airflow implementation.
            # TODO: Remove this line once the change is released.
            "runner": "DataflowRunner",
            "model_uri": model_uri,
            "project_name": project_name,
            "job_id": job_id,
            "return_input": return_input,
            "input_file_format": input_file_format,
            "input_paths": ",".join(input_paths),
            "prediction_path": prediction_path,
        },
        *args,
        **kwargs)


class CloudMLSummarizePredictionOperator(DataFlowPythonOperator):
  """Get the summary of prediction_result and publish at the same path.

  :param input_paths: the list of input paths used for the prediction.
  :type input_paths: list of string
  :param prediction_path: GCS folder of the prediction results. The result files
      prediction.results-NNNNN-of-NNNNN should be under this folder,
      and the summary will be stored as prediction.summary-NNNNN-of-NNNNN in
      the same folder.
  :type prediction_path: string
  :param dataflow_options: options used for dataflow operation,
      including 'project', 'zone', 'staging_location', 'temp_location'.
  :type dataflow_options: dict (string: string)
  """

  template_fields = DataFlowPythonOperator.template_fields + [
      "options",
  ]

  @apply_defaults
  def __init__(self, task_id, dag, input_paths, prediction_path,
               dataflow_options, *args, **kwargs):
    _verify_dataflow_task_id(task_id)
    self._prediction_path = prediction_path
    super(CloudMLSummarizePredictionOperator, self).__init__(
        task_id=task_id,
        dag=dag,
        # TODO(youngheek): move to using a module once it's installed.
        py_file=os.path.join(
            os.getenv("DAGS_FOLDER"), "evaluate_version_sub.py"),
        dataflow_default_options=dataflow_options,
        options={
            # TODO(youngheek): remove this override, once airflow is fixed.
            "runner": "DataflowRunner",
            "input_paths": ",".join(input_paths),
            "prediction_path": prediction_path,
        },
        *args,
        **kwargs)


class CloudMLValidatePredictionOperator(PythonOperator):

  @apply_defaults
  def __init__(self, task_id, dag, prediction_path, validate_fn, *args,
               **kwargs):
    if not callable(validate_fn):
      raise airflow.exception.AirflowException(
          "`validate_fn` param must be callable")
    self._validate_fn = validate_fn

    def apply_validate_fn(*args, **kwargs):
      prediction_path = kwargs["templates_dict"]["prediction_path"]
      scheme, bucket, obj, _, _ = urlparse.urlsplit(prediction_path)
      if scheme != "gs" or not bucket or not obj:
        raise ValueError("Wrong format prediction_path: %s" % prediction_path)
      summary = os.path.join(
          obj.strip("/"), "prediction.summary-00000-of-00001")
      gcs_hook = GoogleCloudStorageHook()
      summary = json.loads(gcs_hook.download(bucket, summary))
      return self._validate_fn(summary)

    super(CloudMLValidatePredictionOperator, self).__init__(
        task_id=task_id,
        dag=dag,
        python_callable=apply_validate_fn,
        provide_context=True,
        templates_dict={"prediction_path": prediction_path},
        *args,
        **kwargs)


class CloudMLTrainingOperator(BaseOperator):
  """Operator for launching a CloudML training job."""

  template_fields = [
      "_job_id",
      "_package_uris",
      "_training_python_module",
      "_training_args",
  ]

  @apply_defaults
  def __init__(self,
               job_id,
               package_uris,
               training_python_module,
               training_args,
               project_name=None,
               scale_tier=None,
               region=None,
               gcp_conn_id="google_cloud_default",
               delegate_to=None,
               cloudml_default_options={},
               mode="PRODUCTION",
               *args,
               **kwargs):
    """Constructs a CloudMLTrainingOperator.

    Args:
      project_name: Name of the Google Cloud project.
      package_uris: A list of package locations, which include the training
        program + any additional dependencies.
      training_python_module: The Python module name to run within CloudML
        training job after installing the 'package_uris' packages.
      args: A list of command line arguments to pass to the CloudML
        training program.
      region: The Google Compute Engine region to run the training job in.
      mode: Can be one of {"PRODUCTION", "TEST", "DRY_RUN"}. If in "TEST" mode,
        a random ID will be appended to the training job ID. In "DRY_RUN" mode,
        the CloudML training job request will be printed out.

    Raises:
      ValueError: when wrong arguments are given.
    """
    super(CloudMLTrainingOperator, self).__init__(*args, **kwargs)
    self._job_id = job_id
    self._package_uris = package_uris
    self._training_python_module = training_python_module
    self._training_args = training_args
    self._project_name = project_name or cloudml_default_options.get("project")
    self._scale_tier = scale_tier or cloudml_default_options.get("scale_tier")
    self._region = region or cloudml_default_options.get("region")
    self._gcp_conn_id = gcp_conn_id
    self._delegate_to = delegate_to
    self._mode = mode

    if not package_uris or len(package_uris) < 1:
      raise ValueError(
          "CloudML Training job needs at least one python package.")
    if not training_python_module:
      raise ValueError(
          "CloudML Training job needs to know which python module to run.")

  def execute(self, context):
    job_id = _normalize_cloudml_job_id(self._job_id)
    training_request = {
        "jobId": job_id,
        "trainingInput": {
            "scaleTier": self._scale_tier,
            "packageUris": self._package_uris,
            "pythonModule": self._training_python_module,
            "region": self._region,
            "args": self._training_args,
        }
    }

    if self._mode == "DRY_RUN":
      logging.info("In dry_run mode.")
      logging.info("CloudML Training job request is: %s", training_request)
      return

    hook = CloudMLHook(
        gcp_conn_id=self._gcp_conn_id, delegate_to=self._delegate_to)
    finished_training_job = None
    try:
      existing_job = hook.get_job(self._project_name, job_id)
      logging.info(
          "Job with job_id {} already exist: {}".format(job_id, existing_job))
      finished_training_job = hook.wait_for_job_done(self._project_name, job_id)
    except errors.HttpError as e:
      if e.resp.status == 404:
        logging.error(
            "Job with job_id {} does not exist. Will create it.".format(job_id))
        finished_training_job = hook.create_job(self._project_name,
                                                training_request)
      else:
        raise e

    if finished_training_job["state"] != "SUCCEEDED":
      logging.error("Training job failed: %s", str(finished_training_job))
      raise RuntimeError(finished_training_job["errorMessage"])


class CloudMLTweakModelOperator(PythonOperator):

  @apply_defaults
  def __init__(self, task_id, dag, export_dir_base, export_dir_new, *args,
               **kwargs):

    def locate_model(export_dir_base):
      """Locate the actual model path from the base.

      Models are exported as '<base>/export/Servo/<timestamp>/saved_model.pb'.
      It finds the latest timestamp using GoogleCloudStorageHook.list, and
      returns the path (excluding /saved_model.pb part).
      """
      scheme, bucket, obj, _, _ = urlparse.urlsplit(export_dir_base)
      if scheme != "gs" or not bucket or not obj:
        raise ValueError("Wrong format export_dir_base: %s" % export_dir_base)
      prefix = os.path.join(obj, "export", "Servo").strip("/")
      gcs_hook = GoogleCloudStorageHook()
      for fname in sorted(gcs_hook.list(bucket, prefix=prefix), reverse=True):
        if fname.endswith("/saved_model.pb"):
          return urlparse.urlunsplit(
              [scheme, bucket, fname[:-len("/saved_model.pb")], "", ""])
      raise RuntimeError("No exported model is found. Please export using "
                         "Estimator.export_savedmodel()")

    def tweak_model(*args, **kwargs):
      """Modify the signature of the model to return the input as well."""

      export_dir = locate_model(kwargs["templates_dict"]["export_dir_base"])
      export_dir_new = kwargs["templates_dict"]["export_dir_new"]

      import subprocess
      subprocess.call(["pip install tensorflow --user"], shell=True)
      subprocess.call(["pip install tensorflow-transform --user"], shell=True)
      import tensorflow as tf
      import tensorflow_transform  # needed to load the criteo_tft model.
      if not tf.saved_model.loader.maybe_saved_model_directory(export_dir):
        raise RuntimeError("Exported model not found: %s" % export_dir)

      with tf.Session(graph=tf.Graph()) as sess:
        model = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        if len(model.asset_file_def) > 0:
          raise RuntimeError(
              "Tweaking models with asset_file_def is not supported yet.")
        # Add pass-through input to all signature_defs.
        for k in model.signature_def:
          new_input = tf.identity(
              sess.graph.get_tensor_by_name(model.signature_def[k].inputs[
                  "inputs"].name))
          model.signature_def[k].outputs["inputs"].CopyFrom(
              model.signature_def[k].inputs["inputs"])
          model.signature_def[k].outputs["inputs"].name = new_input.name
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir_new)
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map=model.signature_def)
        builder.save()

    super(CloudMLTweakModelOperator, self).__init__(
        task_id=task_id,
        dag=dag,
        python_callable=tweak_model,
        provide_context=True,
        templates_dict={
            "export_dir_base": export_dir_base,
            "export_dir_new": export_dir_new
        },
        *args,
        **kwargs)
