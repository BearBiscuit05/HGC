//
// Created by 向向泳安 on 2022/3/11.
//

#ifndef HGC_PR_H
#define HGC_PR_H

#include "../algo.h"

class PR : public Algo
{
    PR(string GraphPath, string EnvPath, int deviceKind, int partition);

    void MSGGenMerge_CPU(Graph& g, vector<int>& mValue) override;

    void MSGGenMerge_GPU(Graph& g, vector<int>& mValue) override;
};


#endif //HGC_PR_H
