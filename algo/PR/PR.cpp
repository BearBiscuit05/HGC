//
// Created by 向向泳安 on 2022/3/11.
//

#include "PR.h"

PR::PR(string GraphPath, string EnvPath, int deviceKind, int partition) {

}

void PR::MSGGenMerge_CPU(Graph &g, vector<int> &mValue) {
    Algo::MSGGenMerge_CPU(g, mValue);
}

void PR::MSGGenMerge_GPU(Graph &g, vector<int> &mValue) {
    Algo::MSGGenMerge_GPU(g, mValue);
}
