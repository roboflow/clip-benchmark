# Clip benchmarks

## Getting started

### Docker

```bash
docker build -t clip-benchmark .
```

```
docker run --gpus all --rm -it --ipc host --network host --shm-size 64g \
-u $(id -u ${USER}):$(id -g ${USER}) \
-v $(pwd):/workspace \
clip-benchmark run_benchmarks.sh
```