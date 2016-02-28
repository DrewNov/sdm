cmake_minimum_required (VERSION 2.8.11)
project (sdm)

set(SOURCE_EXE
        src/main.c
        src/sdm_jaekel.c
        src/sdm_jaekel.h
        src/bmp.c
        src/bmp.h
    )

add_executable(main ${SOURCE_EXE})