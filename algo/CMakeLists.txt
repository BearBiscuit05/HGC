add_library(algo
        Bellman/bellman.cpp Bellman/bellman.h
        BFS/BFS.cpp BFS/BFS.h
        Kruskal/Kruskal.cpp Kruskal/Kruskal.h
        WCC/WCC.cpp WCC/WCC.h)

add_library(basic_algo algo.cpp algo.h algoHead.h)
target_link_libraries(basic_algo env graph)
target_link_libraries(algo basic_algo)