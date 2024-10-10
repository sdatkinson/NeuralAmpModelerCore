// Entry point for tests

#include <iostream>
#include "test/test_dsp.cpp"

int main() {
    std::cout << "Running tests..." << std::endl;
    // TODO Automatically loop, catch exceptions, log results
    test_dsp::test_construct();
    test_dsp::test_get_input_level();
    test_dsp::test_get_output_level();
    test_dsp::test_has_input_level();
    test_dsp::test_has_output_level();
    test_dsp::test_set_input_level();
    test_dsp::test_set_output_level();

    std::cout << "Success!" << std::endl;
    return 0;
}