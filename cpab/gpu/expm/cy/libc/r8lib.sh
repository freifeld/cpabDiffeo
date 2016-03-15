#!/bin/bash
#
cp r8lib.h ./include
#
gcc -c -I./include r8lib.c
if [ $? -ne 0 ]; then
  echo "Errors compiling r8lib.c"
  exit
fi
#
mv r8lib.o ./$ARCH/r8lib.o
#
echo "Library installed as ./$ARCH/r8lib.o"
