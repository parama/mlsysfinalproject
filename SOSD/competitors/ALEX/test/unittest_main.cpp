// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gtest/gtest.h"

#define private public
#include "unittest_alex.h"
#include "unittest_alex_map.h"
#include "unittest_alex_multimap.h"
#include "unittest_nodes.h"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}