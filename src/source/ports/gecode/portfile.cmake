vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Piercelab-Caltech/gecode
    REF 125d82a9cf957f2d1e16c12f0ab164308d6ef3fb
    SHA512 32bfde2da46cb7095c99e8e598d57f595108c95e517aeab252590889a56cab9e639939c29a1d16b10dc582ea00c2c5b54f53c9da71141d93242459631a5d7f14
    HEAD_REF master
)

vcpkg_configure_cmake(
    SOURCE_PATH ${SOURCE_PATH}
    PREFER_NINJA
    # OPTIONS
)

vcpkg_install_cmake()

if(VCPKG_LIBRARY_LINKAGE STREQUAL "static")
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/bin" "${CURRENT_PACKAGES_DIR}/debug/bin")
endif()

file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug/include)
file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug/share)

vcpkg_copy_pdbs()

file(INSTALL ${SOURCE_PATH}/LICENSE DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)
