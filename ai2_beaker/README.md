To run on Ai2's infra:

1. Copy these yaml files into the presto root dir. All commands below are run from the root dir
2. Copy the new data from gcloud into WEKA using sync_weka.yaml:

```bash
beaker experiment create sync_weka.yaml
```

3. Create a docker image of the code
```bash
docker build -t presto .
```
4. Build a beaker image from the docker image
```bash
beaker image create --name gabi-presto presto
```
5. Update the `beaker-config.yaml` to reference the correct beaker image (in this example `gabi-presto`)
6. Kick off the experiment:
```bash
beaker experiment create beaker-config.yaml
```
