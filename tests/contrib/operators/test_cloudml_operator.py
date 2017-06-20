# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
from apiclient import errors
import httplib2
import unittest

import airflow
from airflow import configuration, DAG
from airflow.contrib.operators.cloudml_operator import CloudMLBatchPredictionOperator

from mock import patch

DEFAULT_DATE = datetime.datetime(2017, 6, 6)

INPUT_FOR_SUCCESS = {
    'dataFormat': 'TEXT',
    'inputPaths': ['gs://legal-bucket/fake-input-path/*'],
    'modelName': 'projects/experimental-project/models/fake_model',
    'outputPath': 'gs://legal-bucket/fake-output-path',
    'region': 'us-east1',
}

SUCCESS_MESSAGE = {
    'jobId': 'test_prediction',
    'predictionInput': INPUT_FOR_SUCCESS,
    'predictionOutput': {
        'outputPath': 'gs://fake-output-path',
        'predictionCount': 5000,
        'errorCount': 0,
        'nodeHours': 2.78
    },
    'state': 'SUCCEEDED'
}

DEFAULT_ARGS = {
    'project_id': 'experimental-project',
    'job_id': 'test_prediction',
    'region': 'us-east1',
    'data_format': 'TEXT',
    'input_paths': ['gs://legal-bucket-dash-Capital/legal-input-path/*'],
    'output_path': 'gs://12_legal_bucket_underscore_number/legal-output-path',
    'task_id': 'test-prediction'
}


class CloudMLBatchPredictionOperatorTest(unittest.TestCase):

    def setUp(self):
        super(CloudMLBatchPredictionOperatorTest, self).setUp()
        configuration.load_test_config()
        self.dag = DAG(
            'test_dag',
            default_args={
                'owner': 'airflow',
                'start_date': DEFAULT_DATE,
                'end_date': DEFAULT_DATE,
            },
            schedule_interval='@daily')

    def testSuccess(self):
        with patch('airflow.contrib.operators.cloudml_operator.CloudMLHook') \
                as mock_hook:
            hook_instance = mock_hook.return_value
            hook_instance.get_job.side_effect = errors.HttpError(
                resp=httplib2.Response({
                    'status': 404
                }), content='some bytes')
            hook_instance.create_job.return_value = SUCCESS_MESSAGE

            prediction_task = CloudMLBatchPredictionOperator(
                job_id='test_prediction',
                project_id='experimental-project',
                region=INPUT_FOR_SUCCESS['region'],
                data_format=INPUT_FOR_SUCCESS['dataFormat'],
                input_paths=INPUT_FOR_SUCCESS['inputPaths'],
                output_path=INPUT_FOR_SUCCESS['outputPath'],
                model_name=INPUT_FOR_SUCCESS['modelName'].split('/')[-1],
                dag=self.dag,
                task_id='test-prediction')
            prediction_output = prediction_task.execute(None)

            mock_hook.assert_called_with('google_cloud_default', None)
            hook_instance.create_job.assert_called_with(
                'experimental-project',
                {
                    'jobId': 'test_prediction',
                    'predictionInput': INPUT_FOR_SUCCESS
                })
            self.assertEquals(
                SUCCESS_MESSAGE['predictionOutput'],
                prediction_output)

    def testInvalidModelOrigin(self):
        task_args = DEFAULT_ARGS.copy()
        task_args['uri'] = 'non-gs-uri/saved_model'
        with self.assertRaises(ValueError) as context:
            CloudMLBatchPredictionOperator(**task_args).execute(None)
        self.assertEquals(
            'Missing GCS scheme in the URI: {}.'.format(task_args['uri']),
            str(context.exception))

        task_args = DEFAULT_ARGS.copy()
        task_args['uri'] = 'gs://fake-uri/saved_model'
        task_args['model_name'] = 'fake_model'
        with self.assertRaises(ValueError) as context:
            CloudMLBatchPredictionOperator(**task_args).execute(None)
        self.assertEquals('Ambiguous model origin.', str(context.exception))

        task_args = DEFAULT_ARGS.copy()
        with self.assertRaises(ValueError) as context:
            CloudMLBatchPredictionOperator(**task_args).execute(None)
        self.assertEquals(
            'Missing model version origin.',
            str(context.exception))

        task_args = DEFAULT_ARGS.copy()
        task_args['project_id'] = 'bad_project_underscore'
        task_args['model_name'] = 'legal_fake_model'
        with self.assertRaises(ValueError) as context:
            CloudMLBatchPredictionOperator(**task_args).execute(None)
        self.assertEquals(
            'Error parsing the project id. Illegal project id: {}.'.format(
                task_args['project_id']), str(context.exception))

        task_args = DEFAULT_ARGS.copy()
        task_args['project_id'] = 'bad_project_id_toolong_toolong_toolong'
        task_args['model_name'] = 'legal_fake_model'
        with self.assertRaises(ValueError) as context:
            CloudMLBatchPredictionOperator(**task_args).execute(None)
        self.assertEquals(
            'Error parsing the project id. Project id should be within 6 and '
            '30 characters in length: {}.'.format(task_args['project_id']),
            str(context.exception))

        task_args = DEFAULT_ARGS.copy()
        task_args['model_name'] = \
            'projects/experimental-project/models/fake_model'
        with self.assertRaises(ValueError) as context:
            CloudMLBatchPredictionOperator(**task_args).execute(None)
        self.assertEquals(
            'Error parsing the model name. The resource name is illegal: {}.'
                .format(task_args['model_name']),
            str(context.exception))

        task_args = DEFAULT_ARGS.copy()
        task_args['model_name'] = 'fake_model'
        task_args['version_name'] = 'fake_model/fake_version'
        with self.assertRaises(ValueError) as context:
            CloudMLBatchPredictionOperator(**task_args).execute(None)
        self.assertEquals(
            'Error parsing the version name. The resource name is illegal: {}.'
                .format(task_args['version_name']),
            str(context.exception))

        task_args = DEFAULT_ARGS.copy()
        task_args['version_name'] = 'fake_version'
        with self.assertRaises(ValueError) as context:
            CloudMLBatchPredictionOperator(**task_args).execute(None)
        self.assertEquals(
            'Incomplete version origin.',
            str(context.exception))

    def testInvalidJobId(self):
        task_args = DEFAULT_ARGS.copy()
        task_args['job_id'] = 'job-with-dash'
        task_args['model_name'] = 'fake_model'
        with self.assertRaises(ValueError) as context:
            CloudMLBatchPredictionOperator(**task_args).execute(None)
        self.assertEquals(
            'Illegal job id. The resource name is illegal: {}.'.format(
                task_args['job_id']), str(context.exception))

    def testInvalidIOPath(self):
        task_args = DEFAULT_ARGS.copy()
        task_args['input_paths'] = [r'gs://bucket/carriage-return\r']
        task_args['model_name'] = 'fake_model'
        with self.assertRaises(ValueError) as context:
            CloudMLBatchPredictionOperator(**task_args).execute(None)
        self.assertEquals(
            'Illegal input path. GCS object name must not contain Carriage '
                'Return characters: {}.'.format(task_args['input_paths'][0]),
            str(context.exception))

        task_args = DEFAULT_ARGS.copy()
        task_args['output_path'] = r'gs://bucket/linefeed\n'
        task_args['model_name'] = 'fake_model'
        with self.assertRaises(ValueError) as context:
            CloudMLBatchPredictionOperator(**task_args).execute(None)
        self.assertEquals(
            'Illegal output path. GCS object name must not contain Line Feed '
                'characters: {}.'.format(task_args['output_path']),
            str(context.exception))

    def testHttpError(self):
        http_error_code = 503
        self.assertNotEqual(404, http_error_code)

        with patch('airflow.contrib.operators.cloudml_operator.CloudMLHook') \
              as mock_hook:
            hook_instance = mock_hook.return_value
            hook_instance.get_job.side_effect = errors.HttpError(
                resp=httplib2.Response({
                    'status': http_error_code
                }), content='some bytes')

            with self.assertRaises(errors.HttpError) as context:
                prediction_task = CloudMLBatchPredictionOperator(
                    job_id='test_prediction',
                    project_id='experimental-project',
                    region=INPUT_FOR_SUCCESS['region'],
                    data_format=INPUT_FOR_SUCCESS['dataFormat'],
                    input_paths=INPUT_FOR_SUCCESS['inputPaths'],
                    output_path=INPUT_FOR_SUCCESS['outputPath'],
                    model_name=INPUT_FOR_SUCCESS['modelName'].split('/')[-1],
                    dag=self.dag,
                    task_id='test-prediction')
                prediction_task.execute(None)

                mock_hook.assert_called_with('google_cloud_default', None)
                hook_instance.create_job.assert_called_with(
                    'experimental-project',
                    {
                        'jobId': 'test_prediction',
                        'predictionInput': INPUT_FOR_SUCCESS
                    })

            self.assertEquals(http_error_code, context.exception.resp.status)


if __name__ == '__main__':
    unittest.main()
