#!/bin/bash
#
gcc -c -g -I./include my_test.c >& compiler.txt
if [ $? -ne 0 ]; then
  echo "Errors compiling my_test.c."
  exit
fi
rm compiler.txt
#
gcc my_test.o ./$ARCH/test_matrix_exponential.o \
                                  ./$ARCH/r8lib.o ./$ARCH/c8lib.o -lm
if [ $? -ne 0 ]; then
  echo "Errors linking and loading my_test.o."
  exit
fi
#
rm my_test.o
#
mv a.out my_test
./my_test > my_test_output.txt
if [ $? -ne 0 ]; then
  echo "Errors running my_test."
  exit
fi
#rm my_test
#
echo "Program output written to my_test_output.txt"
