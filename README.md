# end_to_end_project_classifcation

import dagshub
dagshub.init(repo_owner='brandon030921', repo_name='end_to_end_project_classifcation', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

