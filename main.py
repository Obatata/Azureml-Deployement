from azureml.core import Workspace, Environment
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import AksWebservice
from azureml.core.model import Model
from azureml.core.model import InferenceConfig

"""
Acess to workspace
"""
ws = Workspace.from_config("./config/config.json")


"""
Create custom envirnmont
"""
# Create the environment
deploy_env = Environment.from_pip_requirements("deploy_env", "./requirement.txt")
deploy_env.register(ws)

"""
Create Azure kubernets cluster
check if cluster exists in the workspace
if exists ==> use it
    else ==> create a new one
"""
cluster_name = "aks-cluster-01"
if cluster_name in ws.compute_targets:
    print("Cluster {} exists in the workspace".format(cluster_name))
    production_cluster = ws.compute_targets[cluster_name]
else:
    print("Cluster {} does not exist in the workspace".format(cluster_name))
    print("Let's create a AKS !")
    aks_config = AksCompute.provisioning_configuration(
                                                       location="eastus",
                                                       vm_size="STANDARD_D11_V2",
                                                       agent_count=1,
                                                       cluster_purpose="DevTest"
                                                      )

    production_cluster = ComputeTarget.create(ws, cluster_name, aks_config)
    production_cluster.wait_for_completion(show_output=True)


"""
Create inference configuration
"""
inference_config = InferenceConfig(
                                    source_directory="./service_files",
                                    entry_script="predict.py",
                                    environment=deploy_env
                                  )

"""
Create service deloyement configuration 
"""
deploy_config = AksWebservice.deploy_configuration(
                                                    cpu_cores=1,
                                                    memory_gb=1
                                                  )

"""
Create and deploy the webservice
"""
model = ws.models["AdultIncome_model_local"]


service = Model.deploy(
                        workspace=ws,
                        name='adultincome-service',
                        models=[model],
                        inference_config=inference_config,
                        deployment_config=deploy_config,
                        deployment_target=production_cluster
                      )

service.wait_for_deployment(show_output=True)

