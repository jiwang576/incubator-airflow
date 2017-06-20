# Placeholder. See https://github.com/apache/incubator-airflow/blob/master/airflow/contrib/hooks/gcp_dataflow_hook.py
# for reference

import logging
import random
import time
from apiclient.discovery import build
from apiclient import errors
from datetime import datetime
from oauth2client.client import GoogleCredentials

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


class CloudMLHook(GoogleCloudBaseHook):

    def __init__(self,
                 gcp_conn_id='google_cloud_default',
                 delegate_to=None):
        super(CloudMLHook, self).__init__(gcp_conn_id, delegate_to)
        self._cloudml = self.get_conn()

    def get_conn(self):
        """
        Returns a Google CloudML service object.
        """
        # TODO(nedam): Make this work with self._authorize() instead.
        credentials = GoogleCredentials.get_application_default()
        return build('ml', 'v1', credentials=credentials)

    def create_job(self, project_name, job):
        """
        Creates a CloudML Job, and returns the Job object, which can be waited
        upon.

        project_name is the name of the project to use, such as
        'peterdolan-experimental'

        job is the complete Cloud ML Job object that should be provided to the Cloud
        ML API, such as

        {
          'jobId': 'my_job_id',
          'trainingInput': {
            'scaleTier': 'STANDARD_1',
            ...
          }
        }
        """
        cloudml_job = _CloudMLJob(self._cloudml, project_name, job)
        cloudml_job.wait_for_done()
        return cloudml_job.get()
