ansible-playbook deploy-docker-images.yml -i inventory.yml
ansible-playbook deploy-create-instance.yml -i inventory.yml --extra-vars cluster_state=present
ansible-playbook deploy-provision-instance.yml -i inventory.yml
ansible-playbook deploy-setup-containers.yml -i inventory.yml
ansible-playbook deploy-setup-webserver.yml -i inventory.yml
ansible-playbook update-k8s-cluster.yml -i inventory-prod.yml