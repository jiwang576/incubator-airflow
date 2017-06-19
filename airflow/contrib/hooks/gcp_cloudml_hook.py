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


import logging
import random
import time
from apiclient.discovery import build
from apiclient import errors
from oauth2client.client import GoogleCredentials
from datetime import datetime
from airflow.contrib.hooks.gcp_api_base_hook import GoogleCloudBaseHook

logging.getLogger('GoogleCloudML').setLevel(logging.INFO)


class _CloudMLJob(object):

    def __init__(self, cloudml, project_name, job_spec):
        self._cloudml = cloudml
        self._project_name = 'projects/{}'.format(project_name)
        self._job_spec = job_spec

        self._job_id = self._job_spec['jobId']
        assert self._job_id is not None and self._job_id is not ""

        self._job = None
        self._create()

    def _get_job(self):
        name = '{}/jobs/{}'.format(self._project_name, self._job_id)
        request = self._cloudml.projects().jobs().get(name=name)
        try:
            self._job = request.execute()
            return True
        except errors.HttpError, e:
            logging.error('Something went wrong: %s', e)
            return False

    def _create(self):
        """Creates the Job on Cloud ML. Returns True if the job was successfully
        created, and False otherwise."""
        request = self._cloudml.projects().jobs().create(
            parent=self._project_name, body=self._job_spec)
        try:
            response = request.execute()
            return True
        except errors.HttpError, e:
            logging.error('Something went wrong: %s', e)
            return False

    def wait_for_done(self):
        """Waits for the Job to reach a terminal state, and returns the Job's
        status."""
        state = None
        while True:
            self._get_job()
            state = self._job['state']
            if state in ['FAILED', 'SUCCEEDED', 'CANCELLED']:
                break
            time.sleep(10)

    def get(self):
        return self._job


class _CloudMLVersion(object):

    def __init__(self, cloudml, project_name, model_name, version):
        self._cloudml = cloudml
        self._parent_name = 'projects/{}/models/{}'.format(
            project_name, model_name)
        self._project_name = project_name
        self._version_spec = version

    def create(self):
        """Creates the Version on Cloud ML. Returns True if the version was successfully
        created, and False otherwise."""
        request = self._cloudml.projects().models().versions().create(
            parent=self._parent_name,
            body=self._version_spec)
        try:
            response = request.execute()
            logging.info('received response: %s', response)
            self._operation_name = response['name']  # TODO(nedam) Verify this.
            return True
        except errors.HttpError, e:
            logging.error('Something went wrong: %s', e)
            raise e

    def wait_for_done(self):
        """Waits for the Operation to reach a terminal state, and returns the operations'
        response."""
        request = self._cloudml.projects().operations().get(name=self._operation_name)
        for n in range(0, 9):
            try:
                operation = request.execute()
                logging.info('%s-Received operation: %s',
                             datetime.now(), operation)
                if operation.get('error', None) is not None:
                    raise ValueError('Received an error: {}'.format(
                        operation.get('error')))
                if operation.get('done', False):
                    return operation
            except e:
                if e.resp.status != 429:
                    logging.error('Something went wrong. Not retrying: %s', e)
                    raise e

            time.sleep((2**n) + (random.randint(0, 1000) / 1000))

        logging.error('Still not done after 10 re-tries. Giving up.')
        

def _retry_with_exponential_delay(request, max_n, is_done_func, is_error_func):

    for i in range (0, max_n):
        try:
            response = request.execute()
            if is_error_func(response):
                raise ValueError('The response contained an error: {}'.format(response))
            elif is_done_func(response):
                logging.info('Operation is done: {}'.format(response))
                return response
            else:
                time.sleep((2**i) + (random.randint(0, 1000) / 1000))
        except errors.HttpError as e:
          if e.resp.status != 429:
              logging.info('Something went wrong. Not retrying: %s', e)
              raise e
          else:
              time.sleep((2**i) + (random.randint(0, 1000) / 1000))


class CloudMLHook(GoogleCloudBaseHook):

    def __init__(self, gcp_conn_id='google_cloud_default', delegate_to=None):
        super(CloudMLHook, self).__init__(gcp_conn_id, delegate_to)
        self._cloudml = self.get_conn()

    def get_conn(self):
        """
        Returns a Google CloudML service object.
        """
        credentials = GoogleCredentials.get_application_default()
        return build('ml', 'v1', credentials=credentials)

    def create_version(self, project_name, model_name, version_spec):
        """Creates the Version on Cloud ML.

        Returns the operation if the version was created successfully and raises
        an error otherwise.
        """
        parent_name = 'projects/{}/models/{}'.format(project_name, model_name)
        create_request = self._cloudml.projects().models().versions().create(
        parent=parent_name, body=version_spec)
        response = create_request.execute()
        get_request = self._cloudml.projects().operations().get(
                name=response['name'])

        return _retry_with_exponential_delay(
                request = get_request,
                max_n = 9,
                is_done_func = lambda resp: resp.get('done', False),
                is_error_func = lambda resp: resp.get('error', None) is not None)

    def set_default_version(self, project_name, model_name, version_name):
        """
        Sets a version to be the default. Blocks until finished.
        """
        full_version_name = 'projects/{}/models/{}/versions/{}'.format(
            project_name, model_name, version_name)
        request = self._cloudml.projects().models().versions().setDefault(
            name=full_version_name, body={})

        try:
            response = request.execute()
            logging.info('Successfully set version: %s to default', response)
            return response
        except errors.HttpError as e:
            logging.error('Something went wrong: %s', e)
            raise e

    def list_versions(self, project_name, model_name):
        """
        Lists all available versions of a model. Blocks until finished.
        """
        result = []
        full_parent_name = 'projects/{}/models/{}'.format(
                project_name, model_name)
        request = self._cloudml.projects().models().versions().list(
            parent=full_parent_name, pageSize=100)

        response = request.execute()
        next_page_token = response.get('nextPageToken', None)
        result.extend(response.get('versions', []))
        while next_page_token is not None:
            next_request = self._cloudml.projects().models().versions().list(
                parent=full_parent_name,
                pageToken=next_page_token,
                pageSize=100)
            response = next_request.execute()
            next_page_token = response.get('nextPageToken', None)
            result.extend(response.get('versions', []))
            time.sleep(5)
        return result

    def delete_version(self, project_name, model_name, version_name):
        """
        Deletes the given version of a model. Blocks until finished.
        """
        full_name = 'projects/{}/models/{}/versions/{}'.format(
            project_name, model_name, version_name)
        delete_request = self._cloudml.projects().models().versions().delete(
            name=full_name)
        response = delete_request.execute()
        get_request = self._cloudml.projects().operations().get(
                name=response['name'])

        return _retry_with_exponential_delay(
                request = get_request,
                max_n = 9,
                is_done_func = lambda resp: resp.get('done', False),
                is_error_func = lambda resp: resp.get('error', None) is not None)

    def create_model(self, project_name, model):
        """
        Create a Model. Blocks until finished.
        """
        assert model['name'] is not None and model['name'] is not ''
        project = 'projects/{}'.format(project_name)

        request = self._cloudml.projects().models().create(
            parent=project, body=model)
        return request.execute()

    def get_model(self, project_name, model_name):
        """
        Gets a Model. Blocks until finished.
        """
        assert model_name is not None and model_name is not ''
        full_model_name = 'projects/{}/models/{}'.format(
                project_name, model_name)
        request = self._cloudml.projects().models().get(name=full_model_name)
        try:
            return request.execute()
        except errors.HttpError as e:
            if e.resp.status == 404:
                logging.error('Model was not found: %s', e)
                return None
            raise e
