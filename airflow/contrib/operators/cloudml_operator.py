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


import logging
import airflow
from airflow.contrib.hooks.gcp_cloudml_hook import CloudMLHook
from airflow.operators import BaseOperator
from airflow.utils.decorators import apply_defaults

logging.getLogger('GoogleCloudML').setLevel(logging.INFO)


class CloudMLVersionOperator(BaseOperator):
    template_fields = [
        '_model_name',
        '_version',
    ]

    @apply_defaults
    def __init__(self,
                 version=None,
                 cloudml_default_options={},
                 gcp_conn_id='google_cloud_default',
                 operation='create',
                 *args,
                 **kwargs):

        super(CloudMLVersionOperator, self).__init__(*args, **kwargs)

        self._model_name = cloudml_default_options.get('model_name')
        self._version = version
        self._gcp_conn_id = gcp_conn_id
        self._delegate_to = cloudml_default_options.get('delegate_to')
        self._project_name = cloudml_default_options.get('project')
        self._operation = operation

    def execute(self, context):
        hook = CloudMLHook(
            gcp_conn_id=self._gcp_conn_id, delegate_to=self._delegate_to)

        if self._operation == 'create':
            assert self._version is not None
            return hook.create_version(self._project_name, self._model_name,
                                       self._version)
        elif self._operation == 'set_default':
            return hook.set_default_version(
                self._project_name, self._model_name,
                self._version['name'])
        elif self._operation == 'list':
            return hook.list_versions(self._project_name, self._model_name)
        elif self._operation == 'delete':
            return hook.delete_version(self._project_name, self._model_name,
                                       self._version['name'])
        else:
            raise ValueError('Unknown operation: {}'.format(self._operation))


    class CloudMLModelOperator(BaseOperator):

        template_fields = [
            '_model',
        ]

        @apply_defaults
        def __init__(self,
                     model,
                     cloudml_default_options={},
                     gcp_conn_id='google_cloud_default',
                     operation='create',
                     *args,
                     **kwargs):
            super(CloudMLModelOperator, self).__init__(*args, **kwargs)
            self._model = model
            self._operation = operation
            self._gcp_conn_id = gcp_conn_id
            self._delegate_to = cloudml_default_options.get('delegate_to')
            self._project_name = cloudml_default_options.get('project')

        def execute(self, context):
            hook = CloudMLHook(
                gcp_conn_id=self._gcp_conn_id, delegate_to=self._delegate_to)
            if self._operation == 'create':
                hook.create_model(self._project_name, self._model)
            elif self._operation == 'get':
                hook.get_model(self._project_name, self._model['name'])
            else:
                raise ValueError('Unknown operation: {}'.format(self._operation))
