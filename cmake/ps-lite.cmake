include(ExternalProject)

set(pslite_URL https://github.com/dmlc/ps-lite.git)
set(pslite_TAG 2ce8b9a256207947acfa2cb9b09ab74b8de74547)

ExternalProject_Add(ps-lite
  PREFIX ps-lite
  GIT_REPOSITORY ${pslite_URL}
  GIT_TAG ${pslite_TAG}
  UPDATE_COMMAND ""
  BUILD_COMMAND make
)

