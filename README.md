# AMI PROJECT 2022: GROUP05
The aim of this project is to build an image classification model to distinguish between cropped images of car damages. The model can classify the images into "scratch", "dent", "rim" and "other". 

The model is integrated in an easy-to-use web-interface.

## Web features

Users on the web page can make use of the following features :
- Image labeler
     - JSON file containing the labels is available for download.
- Damage classifier
     - In case of misclassification, users can correct the labels then the model will then be retrained.



## Usage

- ### Locally

```bash
docker login gitlab.ldv.ei.tum.de:5005 -u kube-puller -p sCAaXWzxzopG9BPwPCZy
docker pull gitlab.ldv.ei.tum.de:5005/ami2022/group05/web_app:latest
docker run -p 8888:8888 gitlab.ldv.ei.tum.de:5005/ami2022/group05/web_app:latest
```
 On local browser, access: [http://localhost:8888](http://localhost:8080)
- ### Kubernetes
  - Download  `registry-credentials.yml` and `deploy_project.yml` from gitlab registry.
  - Download [kubectl.exe](https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/)
  - Make folder called 'kube' in local computer and add 
`<directory/of/kube/folder>` to PATH.
  - Download `config05` file from [this](https://gitlab.ldv.ei.tum.de/ami2022/Group05/-/issues/13) gitlab issue.
  - Rename `config05` to `config`.
  - Move `config` , `kubectl.exe`, `registry-credentials.yml` and `deploy_project.yml` 
to `<directory/of/kube/folder>`.

```bash
cd <directory/of/kube/folder>
kubectl apply -f registry-credentials.yml --kubeconfig=<directory/of/kube/folder>
kubectl apply -f deploy_project.yml --kubeconfig=<directory/of/kube/folder>
```
 :fireworks: A Pod that runs our docker image should be now running on the Kubernetes cluster.


## Contributing
Pull requests are welcome.
