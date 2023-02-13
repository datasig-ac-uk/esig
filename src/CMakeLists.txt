cmake_minimum_required(VERSION 3.16)


set(ESIG_BUILD_PYMODULE "${CMAKE_CURRENT_LIST_DIR}/esig" CACHE PATH
        "esig pymodule build tree location")

add_subdirectory(common)
add_subdirectory(algebra)
#add_subdirectory(libalgebra_context)
add_subdirectory(paths)
add_subdirectory(pyesig)
