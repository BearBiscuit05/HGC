![img](http://img.shields.io/badge/c%2B%2B-14-brightgreen)

<style>table {margin: auto;}</style>

# Graph_algo_OpenCL

## TODO:

+ 整理OpenCL相关代码集成为类

+ 将代码中float类型->框架下的数据类型

+ 注释掉相关debug内容

---

renew on 11/23 21

## Graph calculation

### Algorithm Classification

#### SSSP

Bellman-ford

#### MST

Kruskal,Prim

### APSP

Floyd(weight)，Floyd-Warshall(unweight)



### Traversal

| Algorithm | SSSP(weighted) | SSSP(unweight) | DFS/BFS |
| :-------: | -------------- | -------------- | ------- |
|    CPU    |                |                |         |
|    GPU    |                |                |         |
|   FPGA    |                |                |         |

| Algorithm | APSP(weight) | APSP(unweight) | MST(Kruskal) | MST(Prim) |
| :-------: | ------------ | -------------- | ------------ | --------- |
|    CPU    |              |                |              |           |
|    GPU    |              |                |              |           |
|   FPGA    |              |                |              |           |

MST:Minimum spanning tree



### Components

| Algorithm | CC(weakly) | CC(strongly) | TC   | Reachability |
| :-------: | ---------- | ------------ | ---- | ------------ |
|    CPU    |            |              |      |              |
|    GPU    |            |              |      |              |
|   FPGA    |            |              |      |              |

cc:Connected components



### Centrality Mearuse

| Algorithm | PR   | BC   | A*   |      |
| :-------: | ---- | ---- | ---- | ---- |
|    CPU    |      |      |      |      |
|    GPU    |      |      |      |      |
|   FPGA    |      |      |      |      |



### Pattern Match

| Algorithm | MM   |      |      |      |
| :-------: | ---- | ---- | ---- | ---- |
|    CPU    |      |      |      |      |
|    GPU    |      |      |      |      |
|   FPGA    |      |      |      |      |



---

项目可以使用autoBuild.sh脚本进行构建

```bash
./autoBuild
```

