mkdir build

cd build

cmake .. -G "Visual Studio 16 2019" -DPARDISO_LIBRARIES="C:/Users/zizhou/Desktop/pardiso/libpardiso600-WIN-X86-64.lib" -DLAPACK_LIBRARIES="D:/LAPACK/lib/liblapack.dll.a" -DBLAS_LIBRARIES="D:/LAPACK/lib/libblas.dll.a"

cmake --build .

cmake --build . --config release