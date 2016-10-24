#!/bin/bash

ALIGNMENT="/home/dh/program/SeetaFaceEngine/FaceAlignment"
IDENTIFICATION="/home/dh/program/SeetaFaceEngine/FaceIdentification"
IMAGE="/home/dh/program/SeetaFaceEngine/FaceAlignment/input"

cd $ALIGNMENT
EXE1="./build/fa_test $1"

$EXE1
mv $ALIGNMENT/data.txt $IDENTIFICATION/data.txt

cp $1 $IDENTIFICATION/input.jpg
cd $IDENTIFICATION

EXE2="./build/src/test/test_face_recognizer.bin $2"

$EXE2

