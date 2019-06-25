ALBANY_ROOT=${WORK_DIR}/albany/albany-install/gcc/opt/branch

rm -rf CMakeFiles
rm -f  CMakeCache.txt

cmake                                   \
  -D CMAKE_BUILD_TYPE:STRING=RELEASE    \
  -D CMAKE_CXX_COMPILER:STRING=mpicxx   \
  -D ALBANY_DIR:PATH=${ALBANY_ROOT}     \
  ../src
