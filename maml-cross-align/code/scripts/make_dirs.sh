#!/bin/usr/bash/env bash
corpus=$1
num_tasks=$2

mkdir ../output/$corpus ../ckpt/$corpus ../data/$corpus/processed ../data/$corpus/processed/${num_tasks}t
